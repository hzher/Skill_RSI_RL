"""Verify read-only decision boundaries from completed DSH Poker and Spectrum runs."""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import gzip
import hashlib
import json

from .run import BENCH, BENCH_COMMIT, canonical_hash, sha256, valid_trace
from .poker_resume import _query_payload, replay_prefix


TASKS = ("blind_spectrum_monitoring", "exploitable_poker")
IDENTITY_FIELDS = (
    "attempt_id", "run_id", "job_id", "phase", "task", "index", "expected_instance_count",
    "fingerprint", "execution_fingerprint", "mode", "schedule", "seed", "stage_seeds",
)


@dataclass(frozen=True)
class CompletedSource:
    batch: Path
    attempt: dict
    trace: dict
    job: dict
    receipt: dict


def _verify_checkout(manifest):
    from skill_rsi_rl.sources import verify_source

    root, commit = verify_source("clbench")
    recorded = manifest["runtime"]
    if root != BENCH or recorded["bench_commit"] != commit or commit != BENCH_COMMIT:
        raise ValueError("Archive and the verified CL-Bench snapshot use different commits")
    if recorded.get("bench_changed_sha256"):
        raise ValueError("Archived run used modified official CL-Bench sources")
    return commit


def _verify_resources(manifest, task):
    resources = {name: value for name, value in manifest["benchmark_resources"].items()
                 if name.startswith(f"data/{task}/")}
    if task == "blind_spectrum_monitoring" and not resources:
        raise ValueError("Spectrum archive has no recorded corpus resource hashes")
    for name, recorded in resources.items():
        path = BENCH / name
        if (not path.resolve().is_relative_to(BENCH / "data" / task) or not path.is_file() or
                path.stat().st_size != recorded.get("bytes") or sha256(path) != recorded.get("sha256")):
            raise ValueError(f"Official benchmark resource differs from the archive: {Path(name).name}")
    return canonical_hash(resources)


def _recorded_file(path):
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("Historical source must be a file at its recorded archive path")
    return path


def _read_manifest(path):
    raw = _recorded_file(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _load_source(batch, manifest, batch_sha, commit, resource_sha, task, index):
    task_info = manifest["tasks"][task]
    if task_info["status"] != "completed" or type(index) is not int or not 0 <= index < task_info["rollout_count"]:
        raise ValueError("Expected a completed task and an existing rollout index")
    checkpoint = batch / "checkpoints" / task
    run = f"run-{index}"
    expected = task_info["instances_per_rollout"]
    alias_sha = sha256(_recorded_file(checkpoint / f"{run}.json.gz"))
    accepted = {manifest["fingerprint"], *manifest.get("execution_revisions", {})}
    candidates = []
    for path in sorted((checkpoint / "attempts" / run).glob("*/manifest.json")):
        attempt, attempt_sha = _read_manifest(path)
        if attempt.get("status") == "completed" and attempt.get("trace_sha256") == alias_sha:
            candidates.append((path.parent, attempt, attempt_sha))
    if len(candidates) != 1:
        raise ValueError("Published checkpoint must match one sealed completed attempt")
    attempt_dir, attempt, attempt_sha = candidates[0]
    trace_path = attempt_dir / "trace.json.gz"
    relative_trace = trace_path.relative_to(batch).as_posix()
    fingerprint = manifest["fingerprint"]
    trace_raw = _recorded_file(trace_path).read_bytes()
    if (attempt.get("attempt_id") != attempt_dir.name or attempt.get("job_id") != f"{task}/{run}" or
            (attempt.get("task"), attempt.get("phase"), attempt.get("index")) != (task, "run", index) or
            attempt.get("trace_path") != relative_trace or attempt.get("fingerprint") != fingerprint or
            attempt.get("execution_fingerprint") not in accepted or hashlib.sha256(trace_raw).hexdigest() != alias_sha):
        raise ValueError("Completed attempt provenance or trace digest differs")

    trace = json.loads(gzip.decompress(trace_raw))
    identity = trace["capture_identity"]
    if (any(identity.get(key) != attempt.get(key) for key in IDENTITY_FIELDS) or
            identity["expected_instance_count"] != expected or identity["mode"] != task_info["run_mode"] or
            identity["seed"] != task_info["seed"] or identity["stage_seeds"] != task_info["stage_seeds"]):
        raise ValueError("Completed trace identity differs from the archived task and attempt")
    params = {key: value for key, value in trace["task"]["params"].items() if key != "expected_num_instances"}
    job = {
        "task": task, "phase": "run", "index": index, "expected_instances": expected,
        "task_params": params, "group": f"{manifest['run_name']}-{task}",
    }
    if (not valid_trace(job, trace) or params.get("schedule") != task_info["schedule"] or
            trace["execution"]["total_interactions"] != len(trace["interactions"])):
        raise ValueError("Completed trace does not match the official rollout")
    receipt = {
        "batch_id": manifest["run_name"], "task": task, "run_index": index,
        "source_training_permission": manifest.get("training_permission"),
        "benchmark_commit": commit, "benchmark_resources_sha256": resource_sha,
        "batch_manifest_sha256": batch_sha, "attempt_manifest_sha256": attempt_sha,
        "trace_path": relative_trace, "trace_sha256": alias_sha,
    }
    return CompletedSource(batch, attempt, trace, job, receipt)


def load_completed_source(batch, task, run_index):
    """Bind a completed rollout to its official checkout, attempt and trace digest."""
    if task not in TASKS:
        raise ValueError("Archived boundary replay supports Poker and Spectrum")
    batch, manifest, batch_sha, commit = _load_archive(batch)
    resource_sha = _verify_resources(manifest, task)
    return _load_source(batch, manifest, batch_sha, commit, resource_sha, task, run_index)


def build_prefix(source, retained_interactions):
    """Slice a verified completed rollout before its first excluded Query."""
    trace = source.trace
    interactions = trace["interactions"]
    if type(retained_interactions) is not int or not 0 < retained_interactions < len(interactions):
        raise ValueError("Boundary requires a nonempty prefix and at least one remaining interaction")
    retained = interactions[:retained_interactions]
    following = interactions[retained_interactions]
    if retained[-1]["done"] or following["step_number"] != retained_interactions + 1:
        raise ValueError("Boundary must precede a numbered, nonterminal rollout decision")
    next_query = following["query"]
    next_identity = next_query["instance_id"], next_query["instance_index"]
    outcomes = trace["instance_outcomes"]
    positions = {(row["instance_id"], row["instance_index"]): idx for idx, row in enumerate(outcomes)}
    if next_identity not in positions:
        raise ValueError("First excluded query has no matching completed instance identity")
    # Poker may auto-resolve extra hands between two model decisions.
    prior_outcomes = outcomes[:positions[next_identity]]
    prefix = {
        **trace, "status": "running", "interactions": retained, "instance_outcomes": prior_outcomes,
        "result": None, "execution": {**trace["execution"], "total_interactions": retained_interactions},
    }
    return prefix, next_query


def verify_boundary(source, retained_interactions, *, native=False):
    """Replay the prefix and hash the matching first excluded benchmark query."""
    prefix, next_query = build_prefix(source, retained_interactions)
    task, query = replay_prefix(source.job, prefix)
    actual = _query_payload(query, next_query, task, None)
    if actual != next_query:
        raise ValueError("Restored task query differs from the first excluded interaction")
    query_sha = canonical_hash(actual)
    receipt = {
        **source.receipt, "retained_interactions": retained_interactions,
        "retained_outcomes": len(prefix["instance_outcomes"]), "next_query_sha256": query_sha,
        "boundary_sha256": canonical_hash([source.receipt["trace_sha256"], retained_interactions, query_sha]),
        "scope": "official_environment_and_next_query",
    }
    if native:
        from .poker_native_resume import prepare_native_branch

        branch = prepare_native_branch(source.batch, source.attempt, prefix)
        receipt["native"] = {
            key: value for key, value in branch.lineage.items()
            if key.endswith("_sha256") or key in {"session_terminal_seq", "retained_usage_rows"}
        }
        receipt["native"]["memory_file_count"] = len(branch.memory)
        receipt["native"]["memory_files_sha256"] = canonical_hash({
            name: hashlib.sha256(content).hexdigest() for name, content in branch.memory.items()
        })
        receipt["environment_boundary_sha256"] = receipt["boundary_sha256"]
        receipt["boundary_sha256"] = canonical_hash([receipt["environment_boundary_sha256"], receipt["native"]])
        receipt["scope"] = "official_environment_next_query_and_native_state"
    return receipt


def _load_archive(batch):
    batch = Path(batch).resolve(strict=True)
    manifest, batch_sha = _read_manifest(batch / "manifest.json")
    if manifest.get("harness") != "dsh":
        raise ValueError("Archive must identify the DSH harness")
    return batch, manifest, batch_sha, _verify_checkout(manifest)


def verify_archive(batch, tasks=TASKS, runs=None, cuts=None, *, native=False):
    batch, manifest, batch_sha, commit = _load_archive(batch)
    entries = []
    for task in tasks:
        if task not in TASKS:
            raise ValueError("Archived boundary replay supports Poker and Spectrum")
        resource_sha = _verify_resources(manifest, task)
        indices = runs if runs is not None else range(manifest["tasks"][task]["rollout_count"])
        for index in indices:
            source = _load_source(batch, manifest, batch_sha, commit, resource_sha, task, index)
            length = len(source.trace["interactions"])
            boundaries = cuts if cuts is not None else (1, length // 2, length - 1)
            for cut in sorted(set(boundaries)):
                try:
                    entries.append(verify_boundary(source, cut, native=native))
                except ValueError as exc:
                    detail = str(exc) if type(exc) is ValueError else type(exc).__name__
                    raise ValueError(f"{task} run {index} at boundary {cut}: {detail}") from exc
    return {"schema_version": 1, "verified_boundaries": len(entries), "entries": entries}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--task", action="append", choices=TASKS)
    parser.add_argument("--run", action="append", type=int)
    parser.add_argument("--cut", action="append", type=int, help="number of saved interactions to replay")
    parser.add_argument("--native", action="store_true", help="also verify DSH session and observed memory state")
    args = parser.parse_args()
    try:
        result = verify_archive(args.batch, args.task or TASKS, args.run, args.cut, native=args.native)
    except (OSError, EOFError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
