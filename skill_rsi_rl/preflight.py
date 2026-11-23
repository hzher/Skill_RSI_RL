"""Check the local H20 training setup and optional frozen inputs without loading weights."""

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path

from .contracts import MODEL_PATH, load_replay


def software():
    from packaging.requirements import Requirement

    requirements = Path(__file__).resolve().parents[1] / "env" / "train.txt"
    installed = {}
    issues = []
    for line in requirements.read_text(encoding="utf-8").splitlines():
        spec = line.partition("#")[0].strip()
        if not spec:
            continue
        requirement = Requirement(spec)
        try:
            version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            version = None
        installed[requirement.name] = {"installed": version, "required": str(requirement.specifier)}
        if version is None or not requirement.specifier.contains(version):
            issues.append(f"{requirement.name} {version} does not satisfy {requirement.specifier}")
    return {"ok": not issues, "python": sys.version.split()[0], "packages": installed, "issues": issues}


def cuda(expected_gpus):
    import torch

    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    devices = []
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        with torch.cuda.device(index):
            bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        devices.append({"index": index, "name": properties.name,
                        "memory_bytes": properties.total_memory, "native_bf16": bf16})
    nccl = torch.distributed.is_nccl_available()
    issues = []
    if count != expected_gpus:
        issues.append(f"expected {expected_gpus} visible GPUs, found {count}")
    if any("H20" not in device["name"] or not device["native_bf16"] for device in devices):
        issues.append("visible GPUs must be NVIDIA H20 with native BF16")
    if expected_gpus > 1 and not nccl:
        issues.append("the installed PyTorch does not provide an NCCL backend")
    return {"ok": not issues, "cuda_build": torch.version.cuda, "visible_gpus": count,
            "devices": devices, "nccl_available": nccl, "issues": issues}


def backend(name):
    if name == "easyr1":
        from .easyr1_backend import backend_provenance

        return backend_provenance()
    from .train import group_losses

    return {"name": name, "policy_loss": f"{group_losses.__module__}.{group_losses.__name__}"}


def read_json(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data


def checkpoint(path):
    config = read_json(path / "config.json")
    if config.get("model_type") != "qwen3":
        raise ValueError("expected a Qwen3 model")
    dimensions = {key: config.get(key) for key in ("vocab_size", "max_position_embeddings")}
    if any(type(value) is not int or value < 1 for value in dimensions.values()):
        raise ValueError("checkpoint must declare a valid vocabulary and maximum sequence length")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        weights = read_json(index)["weight_map"]
        if not isinstance(weights, dict) or not weights or any(not isinstance(value, str) for value in weights.values()):
            raise ValueError("weight index must name at least one shard")
        names = sorted(set(weights.values()))
    else:
        names = ["model.safetensors"]
    size = 0
    for name in names:
        shard = path / name
        if shard.name != name or shard.suffix != ".safetensors":
            raise ValueError(f"invalid weight shard name: {name}")
        with shard.open("rb") as stream:
            if not stream.read(1):
                raise ValueError(f"empty weight shard: {name}")
        size += shard.stat().st_size
    for required in ("tokenizer.json", "tokenizer_config.json"):
        with (path / required).open("rb") as stream:
            if not stream.read(1):
                raise ValueError(f"empty tokenizer file: {required}")
    return {"path": str(path), "model_type": config["model_type"], "weight_shards": len(names),
            "weight_bytes": size, **dimensions}


def archive(path):
    manifest = read_json(path / "manifest.json")
    index = read_json(path / "derived" / "skill-rsi-v1" / "batch_index.json")
    runs = index["runs"]
    if not isinstance(runs, list):
        raise ValueError("archive index must contain a list of runs")
    return {"path": str(path), "batch_id": index["batch_id"],
            "training_permission": manifest["training_permission"], "indexed_runs": len(runs)}


def replay(path, model, model_info, max_prompt_tokens):
    from transformers import AutoTokenizer

    from .train import prepare_groups

    manifest, rows = load_replay(path)
    if manifest["split"] != "train" or manifest.get("fixture_only"):
        raise ValueError("expected an actual train replay")
    if manifest.get("shard", {"rank": 0, "world_size": 1}) != {"rank": 0, "world_size": 1}:
        raise ValueError("merge all replay shards before preflight")
    identity = manifest.get("behavior_checkpoint_id")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("train replay requires an immutable behavior_checkpoint_id")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    limit = min(max_prompt_tokens, model_info["max_position_embeddings"])
    usable, skipped = prepare_groups(rows, limit, model_info["vocab_size"], tokenizer)
    return {"path": str(path), "dataset_id": manifest["dataset_id"], "status": manifest["split"],
            "states": len(rows), "sampled_actions": sum(len(group["samples"]) for group in rows),
            "training_groups": len(usable), "skipped_groups": skipped, "behavior_checkpoint_id": identity}


def check(function, *args):
    try:
        return {"ok": True, **function(*args)}
    except (OSError, ImportError, KeyError, ValueError, TypeError, RuntimeError) as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--backend", choices=("easyr1", "native"), default="easyr1")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.expected_gpus < 1 or args.max_prompt_tokens < 1:
        parser.error("expected positive GPU count and prompt-token limit")
    diagnostics = {
        "software": check(software),
        "backend": check(backend, args.backend),
        "cuda": check(cuda, args.expected_gpus),
        "checkpoint": check(checkpoint, args.model),
    }
    if args.archive:
        diagnostics["archive"] = check(archive, args.archive)
    if args.replay:
        diagnostics["replay"] = (check(replay, args.replay, args.model, diagnostics["checkpoint"],
                                       args.max_prompt_tokens)
                                 if diagnostics["checkpoint"]["ok"] else
                                 {"ok": False, "error": "checkpoint metadata is unavailable"})
    ready = all(result["ok"] for result in diagnostics.values())
    print(json.dumps({"ready": ready, "checks": diagnostics}, indent=2, ensure_ascii=False))
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
