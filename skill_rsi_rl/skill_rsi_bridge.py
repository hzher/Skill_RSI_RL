"""Accept Codex decision states from Skill_RSI; require separate branch scoring.

This is an evidence handoff, never a conversion of executor API traces into
student sampling tokens or fabricated preference rewards.
"""

import argparse
import fcntl
import hashlib
from importlib import import_module
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from skill_rsi_rl.contracts import digest


def framework_identity(root=None):
    """Use the explicitly installed package; never inject adjacent source paths."""
    try:
        package = import_module("skill_rsi")
    except ImportError as exc:
        raise ImportError("Install Skill_RSI in the benchmark environment (pip install -e /path/to/Skill_RSI)") from exc
    installed = Path(package.__file__).resolve().parent
    if root is not None and installed.parent != Path(root).resolve(strict=True):
        raise ImportError("installed skill_rsi does not belong to the selected checkout")
    files = {str(path.relative_to(installed)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(installed.rglob("*.py"))}
    return installed.parent, "sha256:" + digest(files)


def pin_benchmark(benchmark):
    """Pin the optional gateway executor's installed implementation before collection."""
    from .harnesses.native_spec import prepare
    benchmark = prepare(benchmark)
    params = benchmark.get("system_params", {})
    if benchmark.get("harness") != "codex" or params.get("backend") != "skill_rsi":
        return benchmark
    _, identity = framework_identity(params.get("skill_rsi_root"))
    if params.get("source_id", identity) != identity:
        raise ValueError("configured Skill_RSI source identity has changed")
    return {**benchmark, "system_params": {**params, "source_id": identity}}


def receive(bundle, output, benchmark, framework_root=None):
    if benchmark not in {"past-bench", "cl-bench"}:
        raise ValueError("initial training datasets are past-bench and cl-bench")
    _, framework_id = framework_identity(framework_root)
    from skill_rsi.exchange.training import verify_export, verify_receipt

    bundle, output = Path(bundle).resolve(strict=True), Path(output).resolve()
    manifest = verify_export(bundle)
    origins = {row["state_id"]: row for row in (
        json.loads(line) for line in (bundle / "sources.jsonl").read_text().splitlines() if line.strip())}
    accepted, excluded, states = {}, [], []
    for split in manifest["partitions"]:
        accepted[split] = 0
        for line in (bundle / f"{split}/states.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            state = json.loads(line)
            if state["host"] != "codex":
                excluded.append({"state_id": state["state_id"], "reason": "different_host"})
                continue
            origin = origins[state["state_id"]]
            states.append({**state, "split": split, "benchmark": benchmark,
                           "lineage_id": origin["lineage_id"],
                           "scenario_id": digest([manifest["dataset_id"], origin["lineage_id"], state["task_id"]]),
                           "source_projection_sha256": origin["projection_sha256"],
                           "status": "awaiting_counterfactual_native_scores"})
            accepted[split] += 1
    if not states:
        raise ValueError("bundle contains no restorable Codex decision states")
    receipt = {"schema": "skill-rsi.training-receipt.v1", "bundle_sha256": manifest["content_sha256"],
               "dataset_id": manifest["dataset_id"], "accepted": accepted,
               "receiver": "Skill_RSI_RL", "skill_rsi_source_id": framework_id,
               "benchmark": benchmark, "excluded": excluded,
               "status": "evidence_accepted_not_scored_or_sampled"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name(output.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            raise FileExistsError(output)
        with TemporaryDirectory(prefix=".skill-rsi-handoff-", dir=output.parent) as temporary:
            stage = Path(temporary) / "bundle"
            stage.mkdir(mode=0o700)
            for name in ["manifest.json", *manifest["files"]]:
                destination = stage / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((bundle / name).read_bytes())
                destination.chmod(0o600)
            receipt_path = stage / "receiver.json"
            receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
            (stage / "pending-states.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in states))
            # Verify copied bytes against the original contract before committing.
            verify_receipt(stage, receipt_path)
            stage.rename(output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("past-bench", "cl-bench"), required=True)
    parser.add_argument("--skill-rsi-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(receive(args.bundle, args.output, args.benchmark, args.skill_rsi_root), ensure_ascii=False))


if __name__ == "__main__":
    main()
