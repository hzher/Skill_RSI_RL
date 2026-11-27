"""Train a memory-policy LoRA with native CL-Bench rollouts from each frozen policy version."""

import argparse
import gc
import json
import math
import os
import signal
from contextlib import contextmanager
from pathlib import Path
import subprocess
from threading import Thread

from .contracts import digest
from .poker_split import validate_development_split, verify_deal_ids
from .rollout_batch import check_group, update_policy

PROJECT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "group_size": 4, "groups_per_round": 1, "max_group_attempts": 4,
    "seed": 17, "learning_rate": 8e-6, "clip_ratio": 0.2, "beta": 0.01, "lora_rank": 16,
    "device": "cuda:0", "model_id": "Qwen3-8B", "max_prompt_tokens": 4096,
    "logp_tolerance": 0.002, "rollout_timeout_seconds": 7200,
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rollout(path, expected):
    row = read(path)
    verify_deal_ids(expected["task"], expected["environment_seed"], row["outcomes"])
    return row


def configuration(path, device):
    raw = read(path)
    unknown = set(raw) - set(DEFAULTS) - {"split", "benchmark", "environment_seeds", "bench_python"}
    if unknown:
        raise ValueError(f"unknown online config fields: {sorted(unknown)}")
    cfg = {**DEFAULTS, **raw}
    from .skill_rsi_bridge import pin_benchmark

    cfg["benchmark"] = pin_benchmark({**cfg["benchmark"], "split": cfg.get("split")})
    if device:
        cfg["device"] = device
    if cfg.get("split") != "train":
        raise ValueError("online optimization requires split=train")
    validate_development_split(cfg)
    bench = cfg["benchmark"]
    if bench["harness"] not in {"dsh", "codex", "claude"} or not bench["task"]:
        raise ValueError("select a native harness and task")
    if "api_key" in bench.get("system_params", {}):
        raise ValueError("provide native API credentials through the environment")
    for key in ("group_size", "groups_per_round", "max_group_attempts", "lora_rank",
                "max_prompt_tokens", "rollout_timeout_seconds"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["group_size"] < 2 or cfg["max_group_attempts"] < cfg["groups_per_round"]:
        raise ValueError("insufficient group size or attempt budget")
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    for key in ("learning_rate", "beta", "clip_ratio", "logp_tolerance"):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]):
            raise ValueError(f"invalid {key}")
    if not (cfg["learning_rate"] > 0 and cfg["beta"] >= 0 and 0 < cfg["clip_ratio"] < 1
            and 0 < cfg["logp_tolerance"] < 0.1):
        raise ValueError("invalid optimization settings")
    return cfg


@contextmanager
def policy_service(config, model, adapter, policy_id):
    from .serve import QwenPolicy, make_server

    policy = QwenPolicy(model, adapter, config["device"], config["max_prompt_tokens"])
    with make_server("127.0.0.1", 0, config["model_id"], policy_id, policy.decide, sample=policy.sample) as server:
        thread = Thread(target=server.serve_forever, name="frozen-online-policy", daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


def group_identity(config, round_id, attempt, bench_commit):
    benchmark = {**config["benchmark"], "split": "train", "policy_model": config["model_id"]}
    seeds = config["environment_seeds"]
    header = {"bench_commit": bench_commit, "benchmark_identity": digest(
        {k: v for k, v in benchmark.items() if k != "endpoint"}),
        "harness": benchmark["harness"], "task": benchmark["task"], "run_index": benchmark.get("run_index", 0),
        "environment_seed": seeds[(round_id - 1 + attempt) % len(seeds)]}
    streams = [int(digest([config["seed"], round_id, attempt, i])[:15], 16) for i in range(config["group_size"])]
    return header, streams


def run_native(command, log, timeout):
    # A timed-out runner can own a native CLI subprocess. Terminate the entire replica process group.
    with subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True) as proc:
        try:
            status = proc.wait(timeout)
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise
        if status:
            raise subprocess.CalledProcessError(status, command)


def collect(cfg, round_path, round_id, policy_id, endpoint, bench_python, identity, bench_commit):
    from .training_artifacts import _write_run_json, file_sha256

    refs, rejected = [], []
    bench = {**cfg["benchmark"], "endpoint": endpoint, "policy_model": cfg["model_id"], "split": "train"}
    bench_file = round_path / "benchmark.json"
    _write_run_json(bench_file, bench)
    for attempt in range(cfg["max_group_attempts"]):
        group_dir = round_path / f"group-{attempt + 1:04d}"
        group_dir.mkdir(exist_ok=True)
        expected, streams = group_identity(cfg, round_id, attempt, bench_commit)
        environment_seed = expected["environment_seed"]
        receipt = group_dir / "collection.json"
        if not receipt.exists():
            paths = []
            try:
                for i, sampling_seed in enumerate(streams):
                    # Workspaces and sampling streams differ even when environment seed and policy are identical.
                    out = group_dir / f"replica-{i:04d}"
                    result = out / "rollout.json"
                    if not result.is_file():
                        if out.exists():
                            raise ValueError(f"interrupted or failed native rollout: {out.name}")
                        cmd = [str(bench_python), "-m", "skill_rsi_rl.harnesses.runner",
                               "--config", str(bench_file), "--output", str(out),
                               "--environment-seed", str(environment_seed), "--sampling-seed", str(sampling_seed),
                               "--policy-id", policy_id, "--sample"]
                        with (group_dir / f"{out.name}.log").open("w") as log:
                            run_native(cmd, log, cfg["rollout_timeout_seconds"])
                    paths.append(result)
                scores = check_group([read_rollout(p, expected) for p in paths], policy_id, cfg["group_size"],
                                     expected=expected, sampling_seeds=streams)
                group = {"accepted": True, "scores": scores, "rollouts": [
                    {"path": str(p.relative_to(round_path)), "sha256": file_sha256(p)} for p in paths]}
            except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as exc:
                group = {"accepted": False, "reason": str(exc)}
            group.update(environment_seed=environment_seed, parent_policy_id=policy_id, round=round_id,
                         group=attempt + 1, identity=identity)
            _write_run_json(receipt, group)
        group = read(receipt)
        if group["identity"] != identity or group["round"] != round_id or group["parent_policy_id"] != policy_id:
            raise ValueError("collection belongs to a different parent or round")
        if group["accepted"]:
            refs.append(group)
        else:
            rejected.append(group)
        if len(refs) == cfg["groups_per_round"]:
            break
    batch = dict(round=round_id, parent_policy_id=policy_id, identity=identity, groups=refs, rejected=rejected)
    if len(refs) != cfg["groups_per_round"]:
        _write_run_json(round_path / "no-update.json", batch)
        raise RuntimeError(f"online sampling budget exhausted without enough eligible groups: {round_path}")
    _write_run_json(round_path / "batch.json", batch)


def load_batch(path, identity, parent_id, round_id, config, bench_commit):
    from .training_artifacts import file_sha256

    batch = read(path)
    if (batch["identity"] != identity or batch["parent_policy_id"] != parent_id or batch["round"] != round_id
            or len(batch["groups"]) != config["groups_per_round"]):
        raise ValueError("sealed batch identity, group count or parent policy mismatch")
    groups = []
    used_files, group_ids = set(), set()
    for group in batch["groups"]:
        number = group["group"]
        if (type(number) is not int or not 1 <= number <= config["max_group_attempts"] or number in group_ids):
            raise ValueError("duplicate or unexpected sealed group")
        group_ids.add(number)
        expected, streams = group_identity(config, round_id, number - 1, bench_commit)
        rows = []
        for i, ref in enumerate(group["rollouts"]):
            if ref["path"] != f"group-{number:04d}/replica-{i:04d}/rollout.json" or ref["sha256"] in used_files:
                raise ValueError("duplicate or misplaced rollout in sealed batch")
            source = (path.parent / ref["path"]).resolve(strict=True)
            if not source.is_relative_to(path.parent) or file_sha256(source) != ref["sha256"]:
                raise ValueError("sealed rollout reference changed")
            used_files.add(ref["sha256"])
            rows.append(read_rollout(source, expected))
        if check_group(rows, parent_id, config["group_size"], expected=expected, sampling_seeds=streams) != group["scores"]:
            raise ValueError("rollout scores differ from sealed batch")
        groups.append(rows)
    return groups


def run(config, model, initial, output, rounds, bench_python):
    import torch

    from .easyr1_backend import backend_provenance
    from .serve import checkpoint_identities, checkpoint_identity
    from .sources import verify_source
    from .training_artifacts import _write_run_json, file_sha256, load_training_state, lock_output

    if rounds < 1:
        raise ValueError("rounds must be positive")
    if not bench_python.is_file():
        raise FileNotFoundError(f"benchmark interpreter missing: {bench_python}")
    base_id, parent_id = checkpoint_identities(model, initial)
    backend = backend_provenance()
    provenance = {k: v for k, v in backend.items() if k != "checkout"}
    from .harnesses.native_spec import benchmark_commit
    bench_commit = benchmark_commit(config["benchmark"])
    sources = {p.name: file_sha256(p) for p in (
        Path(__file__), Path(__file__).with_name("rollout_batch.py"), Path(__file__).with_name("serve.py"),
        Path(__file__).with_name("train.py"), Path(__file__).with_name("poker_split.py"))}
    harness_root = Path(__file__).parent / "harnesses"
    sources.update({str(p.relative_to(PROJECT)): file_sha256(p) for p in sorted(harness_root.rglob("*"))
                    if p.is_file() and p.suffix in {".py", ".mjs", ".json"}})
    manifest = {"schema_version": 1, "config": {k: v for k, v in config.items() if k != "bench_python"},
                "benchmark_commit": bench_commit,
                "benchmark_kind": config["benchmark"].get("kind", "clbench"),
                "clbench_commit": bench_commit if config["benchmark"].get("kind", "clbench") == "clbench" else None,
                "source_lock_sha256": file_sha256(PROJECT / "sources.lock.json"),
                "base_policy_id": base_id, "initial_policy_id": parent_id, "backend": provenance, "sources": sources}
    identity = digest(manifest)
    parent, optimizer = initial, None
    with lock_output(output):
        path = output / "manifest.json"
        if path.exists() and read(path) != manifest:
            raise ValueError("output belongs to a different online experiment")
        _write_run_json(path, manifest)
        for nr in range(1, rounds + 1):
            rd = output / f"round-{nr:04d}"
            rd.mkdir(exist_ok=True)
            batch_file = rd / "batch.json"
            if not batch_file.is_file():
                with policy_service(config, model, parent, parent_id) as endpoint:
                    collect(config, rd, nr, parent_id, endpoint, bench_python, identity, bench_commit)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            groups = load_batch(batch_file, identity, parent_id, nr, config, bench_commit)
            settings = {"identity": identity, "round": nr, "parent_policy_id": parent_id,
                        "batch_sha256": file_sha256(batch_file)}
            checkpoint = rd / "checkpoint"
            if not checkpoint.exists():
                update_policy(config, model, parent, optimizer, groups, parent_id, nr, checkpoint, settings)
            # An interrupted publication resumes from the atomically committed optimizer and adapter.
            state_file = checkpoint / "training_state.pt"
            raw = torch.load(state_file, map_location="cpu", weights_only=True)
            saved = dict(raw["settings"])
            metrics = saved.pop("metrics")
            if saved != settings or metrics["optimizer_step"] != nr:
                raise ValueError("checkpoint parent, optimization step, or sealed batch has changed")
            del raw
            progress = load_training_state(checkpoint, {**settings, "metrics": metrics}, epochs=None)
            if progress.step != nr or progress.next_epoch != nr:
                raise ValueError("checkpoint step is inconsistent with round")
            new_id = checkpoint_identity(model, checkpoint)
            if new_id == parent_id:
                raise ValueError("checkpoint did not produce a new policy identity")
            result = {**settings, "policy_id": new_id, "checkpoint": "checkpoint", "metrics": metrics,
                      "training_state_sha256": file_sha256(state_file)}
            result_path = rd / "result.json"
            if result_path.exists() and read(result_path) != result:
                raise ValueError("published round no longer matches its weights, optimizer or rollout batch")
            _write_run_json(result_path, result)
            parent, parent_id, optimizer = checkpoint, new_id, progress.optimizer
            print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", default=os.getenv("SKILL_RSI_MODEL") or os.getenv("MODEL_DIR"), type=Path)
    parser.add_argument("--adapter", type=Path, help="optional existing LoRA for the first round")
    parser.add_argument("--output", type=Path, default=Path("runs/poker-online"))
    parser.add_argument("--device")
    parser.add_argument("--bench-python", type=Path)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if args.model is None:
        parser.error("supply --model or set SKILL_RSI_MODEL or MODEL_DIR")
    config = configuration(args.config, args.device)
    bench_python = args.bench_python or Path(config.get("bench_python", PROJECT / ".venv-bench/bin/python"))
    # Keep a virtualenv executable's symlink intact so the child uses its own installed packages.
    run(config, args.model.expanduser().resolve(strict=True), args.adapter, args.output.resolve(), args.rounds,
        bench_python.expanduser().absolute())


if __name__ == "__main__":
    main()
