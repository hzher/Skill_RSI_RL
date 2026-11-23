#!/usr/bin/env python3
"""Run the official CL Bench with a local DeepSeek Harness process."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import fcntl
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
from . import activate_benchmark

BENCH, BENCH_COMMIT = activate_benchmark()
SOURCE_PATHS = (
    "run.py", "dsh_system.py", "plugins/clbench-submit/index.mjs", "plugins/clbench-submit/context-find.mjs",
    "plugins/clbench-submit/context-edit.mjs", "plugins/clbench-submit/context-view.mjs",
    "plugins/clbench-submit/context-probe.mjs", "export_results.py",
    "../contracts.py", "configs/reference_baselines.json",
    "configs/leaderboard_reference.json", "poker_resume.py", "poker_native_resume.py", "codebase_resume.py",
    "sales_resume.py", "sales_capture.py",
)
RECOVERY_POLICY = "dsh-verified-stateful-prefix-resume-v4"
RECOVERY_SOURCES = {
    "dsh_system.py", "run.py", "poker_resume.py", "poker_native_resume.py", "codebase_resume.py", "sales_resume.py",
    "sales_capture.py",
}

TASKS = (
    "exploitable_poker", "blind_spectrum_monitoring", "cohort_studies",
    "database_exploration", "codebase_adaptation", "sales_prediction",
)
DOCKER_TASKS = {"codebase_adaptation", "sales_prediction"}


def load_json(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def save_json(path: Path, value) -> None:
    from src.trace_storage import _atomic_write_json

    _atomic_write_json(path, value)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is writing {path.parent}") from exc
        yield


def load_config(path: Path) -> dict:
    config = load_json(path)
    system = config["system"]
    supported = {"model", "provider", "base_url", "reasoning_effort", "max_tokens",
                 "dsh_bin", "patches", "max_steps", "timeout", "pricing", "tools_mode", "context_find", "context_edit", "context_read"}
    if system.keys() - supported:
        raise ValueError(f"Unsupported system config fields: {sorted(system.keys() - supported)}")
    system["patches"] = [str((ROOT / item).resolve()) for item in system.get("patches", [])]
    base_url = system.get("base_url") or os.environ.get("DEEPSEEK_BASE_URL")
    if base_url:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or (
            parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment
        ):
            raise ValueError("DSH base_url must be an HTTP(S) URL without credentials, query or fragment")
        system["base_url"] = base_url
    if config.get("pricing_reference"):
        config["pricing_provenance"] = load_json(ROOT / config["pricing_reference"])
    if not os.environ.get("DEEPSEEK_API_KEY"):
        credential_path = Path.home() / ".dsh" / ".credentials.yaml"
        if credential_path.exists():
            import yaml

            key = (yaml.safe_load(credential_path.read_text()) or {}).get("refs", {}).get("DEEPSEEK_API_KEY")
            if isinstance(key, str) and key:
                os.environ["DEEPSEEK_API_KEY"] = key
    return config


def benchmark_resources() -> tuple[dict, dict]:
    from skill_rsi_rl.sources import verify_source

    root, commit = verify_source("clbench")
    checkout = {"bench_commit": commit, "bench_dirty": False, "bench_changed_sha256": {},
                "bench_status_sha256": hashlib.sha256(b"").hexdigest()}
    resources = {p.relative_to(root).as_posix() for p in (root / "data").rglob("*")
                 if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)
                 and not p.name.endswith(("-wal", "-shm"))}
    resources.update({"data/database_exploration/products.db", "data/database_exploration/products_drifted.db"})
    files = {}
    for name in sorted(resources):
        path = BENCH / name
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)} if path.is_file() else {"status": "missing"}
    return checkout, files


def use_benchmark_data() -> None:
    # Some upstream task defaults look up data relative to the current directory.
    os.chdir(BENCH)


def result_from_trace(trace: dict):
    from src.interface import EvalMetrics, InstanceOutcome, TaskResult

    raw = trace["result"]
    fields = {f.name for f in dataclasses.fields(InstanceOutcome)}
    return TaskResult(
        metrics=raw["metrics"], summary=raw["summary"],
        eval_metrics=EvalMetrics(**raw["eval_metrics"]),
        instance_outcomes=[InstanceOutcome(**{k: v for k, v in item.items() if k in fields})
                           for item in trace["instance_outcomes"]],
    )


def attempt_root(job: dict) -> Path:
    path = Path(job["path"])
    return path.parent / "attempts" / path.name.removesuffix(".json.gz")


def valid_trace(job: dict, trace: dict) -> bool:
    outcomes = trace.get("instance_outcomes", [])
    expected = 1 if job["phase"] == "baseline" else job["expected_instances"]
    return (
        trace.get("status") == "completed"
        and trace.get("phase") == job["phase"]
        and trace.get("task", {}).get("name") == job["task"]
        and trace.get("task", {}).get("params", {}).get("schedule") == job["task_params"]["schedule"]
        and trace.get("execution", {}).get("run_group_id") == job["group"]
        and trace.get("execution", {}).get("run_index") == job["index"]
        and len(outcomes) == expected
        and len({row["instance_id"] for row in outcomes if row["instance_id"]}) == expected
        and trace.get("result", {}).get("instance_outcomes") == outcomes
    )


def attempt_records(job: dict) -> list[tuple[Path, dict]]:
    records = []
    for path in attempt_root(job).glob("*/manifest.json"):
        try:
            row = load_json(path)
            if row["attempt_id"] == path.parent.name and isinstance(row["created_at"], str) and "status" in row:
                records.append((path, row))
        except (OSError, KeyError, TypeError, ValueError):
            continue
    return sorted(records, key=lambda item: (item[1]["created_at"], item[1]["attempt_id"]))


def completed_checkpoint(job: dict) -> tuple[dict, str] | None:
    batch = Path(job["batch"])
    for manifest, attempt in reversed(attempt_records(job)):
        if attempt.get("status") != "completed" or attempt.get("fingerprint") != job["fingerprint"]:
            continue
        if attempt.get("execution_fingerprint", job["fingerprint"]) not in job["accepted_executions"]:
            continue
        trace_path = manifest.parent / "trace.json.gz"
        if attempt.get("trace_path") != trace_path.relative_to(batch).as_posix():
            continue
        try:
            digest = sha256(trace_path)
            if digest != attempt.get("trace_sha256"):
                continue
            trace = load_json(trace_path)
            identity = trace.get("capture_identity", {})
            if (not valid_trace(job, trace) or identity.get("attempt_id") != attempt["attempt_id"]
                    or identity.get("execution_fingerprint") != attempt.get("execution_fingerprint")):
                continue
            return trace, digest
        except (OSError, EOFError, AttributeError, KeyError, TypeError, ValueError):
            continue
    return None


def publish_checkpoint(path: Path, checkpoint: tuple[dict, str]) -> None:
    trace, digest = checkpoint
    if not path.is_file() or sha256(path) != digest:
        save_json(path, trace)


class UnverifiedRolloutPrefix(ValueError):
    """Published prefix evidence is insufficient to safely restart a rollout."""


def verify_resume_ancestry(batch: Path, live: Path, attempt: dict, prefix: dict) -> tuple[Path, dict, dict] | None:
    parent = attempt.get("resume_from")
    if parent is None:
        return
    previous = live.parent.parent / parent["attempt_id"] / "live.json.gz"
    if (
        previous == live
        or previous.resolve() != previous
        or previous.parent.parent != live.parent.parent
        or previous.relative_to(batch).as_posix() != parent["trace_path"]
        or sha256(previous) != parent["trace_sha256"]
    ):
        raise ValueError("Declared parent prefix cannot be verified")
    previous_manifest = load_json(previous.with_name("manifest.json"))
    previous_trace = load_json(previous)
    interactions = previous_trace["interactions"][:parent["interactions"]]
    outcomes = previous_trace["instance_outcomes"]
    if (
        previous_manifest["attempt_id"] != parent["attempt_id"]
        or previous_manifest["run_id"] != parent["run_id"]
        or previous_manifest["trace_sha256"] != parent["trace_sha256"]
        or previous_manifest["trace_path"] != parent["trace_path"]
        or previous_manifest["job_id"] != attempt["job_id"]
        or previous_manifest["fingerprint"] != attempt["fingerprint"]
        or (len(outcomes), len(interactions)) != (parent["instances"], parent["interactions"])
        or prefix["interactions"][:len(interactions)] != interactions
        or prefix["instance_outcomes"][:len(outcomes)] != outcomes
    ):
        raise ValueError("Resumed attempt diverges from its declared parent prefix")
    return previous, previous_manifest, previous_trace


def database_resume_boundaries(batch: Path, live: Path, attempt: dict, prefix: dict, approved) -> dict:
    """Identify private database relocation points backed by sealed native ancestry."""
    boundaries = {}
    visited = {attempt["attempt_id"]}
    while parent := verify_resume_ancestry(batch, live, attempt, prefix):
        previous, previous_manifest, previous_trace = parent
        previous_id = previous_manifest["attempt_id"]
        points = attempt["resume_from"]
        retained = points["interactions"]
        rows = prefix["interactions"]
        if (
            previous_id in visited
            or previous_manifest.get("status") not in {"failed", "interrupted"}
            or previous_manifest.get("execution_fingerprint", previous_manifest["fingerprint"]) not in approved
            or not 0 < retained <= len(previous_trace["interactions"])
        ):
            raise ValueError("Database replay has untrusted or cyclic attempt ancestry")
        visited.add(previous_id)
        if retained < len(rows):
            prior = rows[retained - 1]["response"]["metadata"]
            current = rows[retained]["response"]["metadata"]
            origin, destination = (Path(row["runtime_directory"]) for row in (prior, current))
            native = attempt.get("native_resume") or {}
            runtime_root = batch / "native/database_exploration"
            if (
                origin == destination
                or origin.parent != runtime_root
                or destination.parent != runtime_root
                or origin.relative_to(batch).as_posix() not in previous_manifest["runtime_directories"]
                or destination.relative_to(batch).as_posix() not in attempt["runtime_directories"]
                or native.get("source_attempt_id") != previous_id
                or native.get("source_trace_sha256") != points["trace_sha256"]
                or native.get("source_runtime") != origin.relative_to(batch).as_posix()
                or native.get("restored_workspace") != str(destination / "workspace")
                or not prior.get("session_id")
                or prior["session_id"] != current.get("session_id")
                or prior["session_id"] != native.get("session_id")
            ):
                raise ValueError("Database replay runtime changed without a matching native resume receipt")
            boundaries[retained + 1] = str(origin), str(destination)
        live, attempt, prefix = previous, previous_manifest, previous_trace
    return boundaries


def rollout_resume_source(job: dict, attempts: list[tuple[Path, dict]]) -> tuple | None:
    from .poker_native_resume import prepare_native_resume
    from .poker_resume import PREFIX_TASKS, replay_prefix

    if job["task"] not in PREFIX_TASKS or job["phase"] != "run":
        return None

    batch = Path(job["batch"])
    candidates = []
    rejected = {}

    def reject(attempt, exc):
        rejected[attempt["attempt_id"]] = str(exc) if type(exc) is ValueError else type(exc).__name__

    invalid_prefix = (OSError, EOFError, ValueError, TypeError, KeyError, AttributeError, subprocess.SubprocessError)
    for path, attempt in attempts:
        live = path.with_name("live.json.gz")
        if attempt["status"] not in {"failed", "interrupted"} or not live.is_file():
            continue
        try:
            if attempt.get("fingerprint") != job["fingerprint"] or (
                attempt.get("execution_fingerprint", job["fingerprint"]) not in job["accepted_executions"]
            ):
                raise ValueError("Rollout prefix does not match batch or approved executor")
            digest = sha256(live)
            if attempt.get("trace_path") != live.relative_to(batch).as_posix() or attempt.get("trace_sha256") != digest:
                raise ValueError("Rollout prefix has no matching sealed trace hash")
            prefix = load_json(live)
            if prefix["interactions"]:
                progress = (len(prefix["instance_outcomes"]), len(prefix["interactions"]), attempt["created_at"])
                candidates.append((progress, live, attempt, prefix, digest))
        except invalid_prefix as exc:
            reject(attempt, exc)

    # Every candidate is a complete prefix of one attempt. Never join sibling attempts.
    ordered = sorted(candidates, key=lambda entry: entry[0], reverse=True)
    for progress, live, attempt, prefix, digest in ordered:
        if job["task"] == "sales_prediction" and progress[0] < ordered[0][0][0]:
            break
        try:
            db_boundaries = None
            if job["task"] == "database_exploration":
                db_boundaries = database_resume_boundaries(batch, live, attempt, prefix, job["accepted_executions"])
            else:
                verify_resume_ancestry(batch, live, attempt, prefix)
            source_interactions = len(prefix["interactions"])
            if job["task"] == "codebase_adaptation":
                from .codebase_resume import restore_task_prefix

                task, query, prefix = restore_task_prefix(job, attempt, prefix)
                replayed = task, query
                native = prepare_native_resume(batch, attempt, prefix)
            elif job["task"] == "sales_prediction":
                from .sales_resume import replay_sales_prefix, sales_snapshot_registration, verify_sales_image

                native = prepare_native_resume(batch, attempt, prefix)
                replayed = replay_sales_prefix(job, prefix, attempt)
                snapshot = sales_snapshot_registration(batch, attempt, prefix, replayed[0])
                verify_sales_image(replayed[0], snapshot["image_id"], parent_image_id=snapshot.get("parent_image_id"))
            else:
                native = prepare_native_resume(batch, attempt, prefix)
                replayed = replay_prefix(job, prefix, db_boundaries=db_boundaries)
        except invalid_prefix as exc:
            reject(attempt, exc)
            continue
        source = {
            "attempt_id": attempt["attempt_id"], "run_id": attempt["run_id"],
            "trace_path": live.relative_to(batch).as_posix(), "trace_sha256": digest,
            "instances": len(prefix["instance_outcomes"]), "interactions": len(prefix["interactions"]),
            "source_interactions": source_interactions,
        }
        if rejected:
            source["unverified_attempts"] = rejected
        if job["task"] == "sales_prediction":
            source["docker_snapshot_image_id"] = snapshot["image_id"]
            source["docker_snapshot_parent_id"] = snapshot.get("parent_image_id")
        return prefix, replayed, native, source, attempt["runtime_directories"]

    if rejected:
        details = ", ".join(f"{attempt}: {reason}" for attempt, reason in rejected.items())
        raise UnverifiedRolloutPrefix(f"No verified {job['task']}/run-{job['index']} prefix. {details}")
    return None


def execute_job(job: dict) -> str:
    """Run one official baseline instance or rollout under an independent attempt identity."""
    from .dsh_system import DSHSystem
    from src.registry import get_task_class
    from src.runs import _run_baseline_instance, run_single

    batch = Path(job["batch"])
    path = Path(job["path"])
    parent = attempt_root(job)
    with exclusive_lock(parent / ".lock"):
        if (checkpoint := completed_checkpoint(job)) is not None:
            publish_checkpoint(path, checkpoint)
            return str(path)
        attempts = attempt_records(job)
        for stale_path, stale in attempts:
            if stale["status"] == "running":
                stale["status"] = "interrupted"
                stale["ended_at"] = None
                save_json(stale_path, stale)
        fresh_session = job.get("stateful_only") or job["system_params"].get("context_edit", False)
        recovery = None if fresh_session else rollout_resume_source(job, attempts)
        previous_source = recovery[3] if recovery else None
        attempt_id = uuid.uuid4().hex
        identity = {
            "run_id": uuid.uuid4().hex, "attempt_id": attempt_id,
            "job_id": f"{job['task']}/{job['phase']}-{job['index']}",
            "task": job["task"], "phase": job["phase"], "index": job["index"],
            "expected_instance_count": 1 if job["phase"] == "baseline" else job["expected_instances"],
            "fingerprint": job["fingerprint"],
            "execution_fingerprint": job["execution_fingerprint"],
            "mode": "reset_each_instance" if job["phase"] == "baseline" else job["mode"],
            "schedule": job["task_params"]["schedule"],
            "seed": job["seed"], "stage_seeds": job["stage_seeds"],
        }
        if previous_source:
            identity["parent_run_id"] = previous_source["run_id"]
            identity["resume_from"] = previous_source
        directory = parent / attempt_id
        manifest_path = directory / "manifest.json"
        trace_path = directory / "trace.json.gz"
        live = directory / "live.json.gz"
        manifest = {
            "schema_version": 1, **identity, "status": "running", "created_at": now(), "ended_at": None,
            "previous_attempt_id": attempts[-1][1]["attempt_id"] if attempts else None,
            "trace_path": None, "trace_sha256": None,
            "runtime_directories": list(recovery[4]) if recovery else [], "error": None,
            "skill": job["skill_loading"],
        }
        if job.get("stateful_only"):
            manifest["memory_policy"] = job["system_params"].get("memory_policy")
        if fresh_session:
            manifest["partial_recovery"] = "fresh_session"
        save_json(manifest_path, manifest)
        instances = []
        sales_tasks = []

        class JobSystem(DSHSystem):
            def __init__(self, **params):
                super().__init__(**params)
                instances.append(self)

            def reset(self) -> None:
                super().reset()
                manifest["runtime_directories"].append(self.directory.relative_to(batch).as_posix())
                try:
                    save_json(manifest_path, manifest)
                except OSError as exc:
                    self._capture_error("runtime_link", exc)

        trace = None
        try:
            kwargs = dict(
                task_class=get_task_class(job["task"]), task_params=job["task_params"],
                system_class=JobSystem, system_params={**job["system_params"], "capture_identity": identity},
                run_group_id=job["group"], system_name="dsh", task_name=job["task"],
            )
            if job["phase"] == "baseline":
                _, _, trace = _run_baseline_instance(**kwargs, instance_index=job["index"])
            elif recovery:
                from .poker_native_resume import restore_native_resume
                from .poker_resume import resume_rollout

                prefix, replayed, native, source, _ = recovery
                system = JobSystem(**kwargs["system_params"])
                manifest["native_resume"] = restore_native_resume(system, native, prefix)
                if job["task"] == "codebase_adaptation":
                    replayed[0]._start_container()
                elif job["task"] == "sales_prediction":
                    from .sales_resume import attach_sales_snapshot

                    sales_tasks.append(replayed[0])
                    manifest["recovery_container_id"] = attach_sales_snapshot(
                        replayed[0], source["docker_snapshot_image_id"],
                        parent_image_id=source["docker_snapshot_parent_id"],
                    )
                save_json(manifest_path, manifest)
                print(f"  resuming {identity['job_id']} after {source['instances']} scored instances "
                      f"and {source['interactions']} accepted interactions", flush=True)
                trace = resume_rollout(job, system, prefix, live, replayed=replayed)
            else:
                if job["task"] == "sales_prediction":
                    from .sales_capture import sales_task_with_checkpoint_owner

                    kwargs["task_class"] = sales_task_with_checkpoint_owner(sales_tasks)
                _, _, trace, _ = run_single(
                    **kwargs, run_index=job["index"], live_trace_path=live,
                )
            trace["capture_identity"] = identity
            trace["system"]["continuity"] = {
                "mode": "native_session", "baseline_scope": "isolated_instance",
                "reset_behavior": "new_session_home_workspace",
            }
            if job.get("stateful_only"):
                trace["system"]["continuity"]["baseline_scope"] = "none"
                trace["system"]["memory_policy"] = manifest["memory_policy"]
            save_json(trace_path, trace)
            if not valid_trace(job, trace):
                raise RuntimeError(f"Incomplete {identity['job_id']}; inspect {directory}")
            manifest["status"] = "completed"
        except BaseException as exc:
            if recovery and job["task"] != "sales_prediction":
                from .poker_resume import discard_replayed_task

                discard_replayed_task(job["task"], recovery[1][0])
            manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            try:
                for system in instances:
                    system.close()
                retained = trace_path if trace_path.exists() else live if live.exists() else None
                if retained:
                    manifest["trace_path"] = retained.relative_to(batch).as_posix()
                    manifest["trace_sha256"] = sha256(retained)
                manifest["ended_at"] = now()
                save_json(manifest_path, manifest)
                if sales_tasks and manifest["error"] and retained == live and sales_tasks[-1]._env is not None:
                    from .sales_capture import capture_sales_failure

                    try:
                        snapshot = capture_sales_failure(
                            batch, manifest, load_json(live), sales_tasks[-1], save_json=save_json,
                        )
                    except Exception as exc:
                        print(f"Sales snapshot could not be sealed: {type(exc).__name__}", file=sys.stderr)
                    else:
                        manifest["docker_snapshot_image_id"] = snapshot["image_id"]
                        save_json(manifest_path, manifest)
                        sales_tasks[-1]._cleanup_container()
            except Exception as archive_error:
                if manifest["error"] is None:
                    raise
                print(f"Failed to finalize attempt evidence at {manifest_path}: {archive_error}", file=sys.stderr)
        save_json(path, trace)
        return str(path)


class IncompleteRecoveredJobs(RuntimeError):
    """Known transient failures remain after retrying their independent jobs."""


class RejectedDSHRequest(RuntimeError):
    """The upstream provider rejected this job's request before an action was submitted."""


def dsh_no_submission(exc: Exception) -> dict:
    prefix = "DSH did not submit an action: "
    while exc is not None:
        terminal, separator, _ = str(exc).rpartition(". Logs: ")
        if isinstance(exc, RuntimeError) and separator and terminal.startswith(prefix):
            try:
                reason = ast.literal_eval(terminal.removeprefix(prefix))
            except (SyntaxError, ValueError):
                return {}
            return reason if isinstance(reason, dict) else {}
        exc = exc.__cause__
    return {}


def execute_with_recovery(job: dict) -> str:
    from .dsh_system import RetryableDSHError

    for attempt in range(2):
        try:
            return execute_job(job)
        except Exception as exc:
            cause = exc
            while cause is not None and not isinstance(cause, RetryableDSHError):
                cause = cause.__cause__
            if cause is None:
                reason = dsh_no_submission(exc)
                error = reason.get("error") if reason.get("kind") == "error" else None
                if isinstance(error, dict) and error.get("code") == "INVALID_REQUEST" and error.get("status") == 400:
                    raise RejectedDSHRequest(str(exc)) from exc
                if reason.get("kind") not in {"completed", "max-tokens"}:
                    raise
            if attempt:
                if cause is exc:
                    raise
                raise RetryableDSHError(str(cause or exc)) from exc
            print(f"  retrying {job['task']}/{job['phase']}-{job['index']} after recoverable DSH response failure", flush=True)


def run_pending_jobs(pending: list[dict], workers: int) -> None:
    from .dsh_system import RetryableDSHError

    exhausted = []

    def report(job, result):
        try:
            result()
        except RetryableDSHError:
            exhausted.append(f"{job['phase']}-{job['index']}")
            print(f"  retry exhausted for {job['task']}/{exhausted[-1]}", flush=True)
        except UnverifiedRolloutPrefix as exc:
            exhausted.append(f"{job['phase']}-{job['index']}")
            print(f"  recovery blocked: {exc}", flush=True)
        except RejectedDSHRequest:
            exhausted.append(f"{job['phase']}-{job['index']}")
            print(f"  upstream rejected {job['task']}/{exhausted[-1]} with HTTP 400", flush=True)
        else:
            print(f"  completed {job['phase']} {job['index']}", flush=True)

    if workers == 1:
        for job in pending:
            report(job, lambda: execute_with_recovery(job))
    elif pending:
        import multiprocessing

        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(execute_with_recovery, job): job for job in pending}
            try:
                for future in as_completed(futures):
                    report(futures[future], future.result)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    if exhausted:
        raise IncompleteRecoveredJobs(f"Unfinished jobs: {', '.join(sorted(exhausted))}")


def run_task(
    task_name: str, run_dir: Path, profile: str, config: dict, workers: int, provenance: dict,
    *, single_rollout: bool = False,
) -> dict:
    from src.interface import TaskResults
    from src.registry import get_task_class
    from src.runs import RunMode, _merge_baseline_instance_results, derive_run_task_params
    from src.tasks.schedules import get_task_schedule
    from src.trace_metrics import attach_baseline_to_outcomes, build_benchmark_aggregate
    from src.trace_metrics import build_sequence_signature, summarize_instance_outcomes
    from src.trace_storage import BaselineSummaryEntry, RunSummary, RunSummaryRunEntry

    artifact = run_dir / "tasks" / f"{task_name}.json.gz"
    schedule_id = "quick_test" if profile == "quick" else "default"
    schedule = get_task_schedule(task_name, schedule_id)
    task_params = {"schedule": schedule_id}
    task_class = get_task_class(task_name)
    task = task_class(**task_params)
    expected = task.num_instances
    seed = getattr(task, "seed", None)
    stage_seeds = [stage.schedule.get("seed") for stage in schedule.stages]
    defaults = schedule.run_defaults
    runs = 1 if profile == "quick" or single_rollout else defaults.runs
    mode = RunMode(defaults.mode)
    stateful_only = config.get("stateful_only", False)
    system_params = {**config["system"], "runtime_root": str(run_dir / "native" / task_name)}
    if config.get("memory_policy") is not None:
        system_params["memory_policy"] = config["memory_policy"]
    if config.get("skill_loading", {}).get("sha256"):
        system_params.update(
            skill_path=str(run_dir / config["skill_loading"]["path"]),
            skill_sha256=config["skill_loading"]["sha256"],
        )
    group = f"{run_dir.name}-{task_name}"
    checkpoint = run_dir / "checkpoints" / task_name
    checkpoint.mkdir(parents=True, exist_ok=True)
    jobs = []
    common = {
        "task": task_name, "group": group, "system_params": system_params, "batch": str(run_dir),
        **provenance, "expected_instances": expected, "mode": mode.value,
        "skill_loading": config.get("skill_loading"), "seed": seed, "stage_seeds": stage_seeds,
        "stateful_only": stateful_only,
    }
    for index in range(runs):
        jobs.append(dict(
            **common, phase="run", index=index,
            task_params=derive_run_task_params(task_params, mode, index),
            path=str(checkpoint / f"run-{index}.json.gz"),
        ))
    baseline_count = 0 if stateful_only else expected
    for index in range(baseline_count):
        jobs.append(dict(
            **common, phase="baseline", index=index,
            task_params={**task_params, **(task_class.baseline_task_params or {})},
            path=str(checkpoint / f"baseline-{index}.json.gz"),
        ))
    pending = []
    for job in jobs:
        saved = completed_checkpoint(job)
        if saved is None:
            pending.append(job)
        else:
            publish_checkpoint(Path(job["path"]), saved)
    scope = "stateful only" if stateful_only else "with baseline"
    print(f"{task_name}: {expected} instances × {runs} rollout(s), {scope}; "
          f"{len(pending)} pending jobs, {workers} workers", flush=True)
    run_pending_jobs(pending, workers)

    traces = [load_json(checkpoint / f"run-{index}.json.gz") for index in range(runs)]
    baseline = baseline_entry = None
    baseline_outcomes = []
    if baseline_count:
        baseline_traces = [load_json(checkpoint / f"baseline-{index}.json.gz") for index in range(baseline_count)]
        _, baseline, baseline_execution = _merge_baseline_instance_results(
            [(index, result_from_trace(trace), trace) for index, trace in enumerate(baseline_traces)],
            run_group_id=group, system_name="dsh", task_name=task_name, task_params=task_params,
            system_params=system_params, expected_num_instances=expected,
            start_time=min(trace["execution"]["start_time"] for trace in baseline_traces),
            end_time=max(trace["execution"]["end_time"] for trace in baseline_traces),
        )
        baseline_outcomes = baseline["instance_outcomes"]
        # Instance identity survives both schedule permutation and slicing.
        expected_identities = {row["instance_id"] for row in baseline_outcomes}
        if len(expected_identities) != expected or any(
            {row["instance_id"] for row in trace["instance_outcomes"]} != expected_identities
            for trace in traces
        ):
            raise RuntimeError(f"Baseline and rollout instance identity sets differ for {task_name}")
        baseline_entry = BaselineSummaryEntry(
            trace_path="embedded", execution=baseline_execution, instance_outcomes=baseline_outcomes,
        )
    entries = []
    for index, trace in enumerate(traces):
        outcomes = trace["instance_outcomes"]
        if baseline_count:
            outcomes = attach_baseline_to_outcomes(baseline_outcomes, outcomes)
        entries.append(RunSummaryRunEntry(
            run_index=index, score=trace["result"]["score"],
            trace_path=f"../checkpoints/{task_name}/run-{index}.json.gz",
            execution=trace["execution"], instance_outcomes=outcomes,
            aggregate=summarize_instance_outcomes(outcomes),
        ))
    results = TaskResults(group, [result_from_trace(trace) for trace in traces],
                          [trace["execution"] for trace in traces])
    aggregate = results.aggregate()
    if baseline_count:
        aggregate.update(build_benchmark_aggregate(baseline_outcomes, [trace["instance_outcomes"] for trace in traces]))
    reference = baseline or traces[0]
    summary = RunSummary(
        run_group_id=group, system=traces[0]["system"],
        task={"name": task_name, "params": {**task_params, "expected_num_instances": expected}},
        schedule=schedule_id, run_count=runs, max_workers=workers, run_mode=mode.value,
        aggregate=aggregate, runs=entries, baseline=baseline_entry,
        baseline_mode="none" if stateful_only else "reset_each_instance",
        task_brief=reference.get("task_brief"), sequence_signature=build_sequence_signature(reference["instance_outcomes"]),
    )
    payload = {
        "kind": "viewer_artifact", "summary": summary.to_dict(), "baseline_trace": baseline,
        "run_traces": [{"run_index": index, "trace": trace} for index, trace in enumerate(traces)],
    }
    try:
        current = load_json(artifact)
    except (OSError, EOFError, ValueError):
        current = None
    if current != payload:
        save_json(artifact, payload)
        print(f"Saved {artifact}", flush=True)
    return {
        "schedule": schedule_id, "run_mode": mode.value, "rollout_count": runs,
        "instances_per_rollout": expected, "baseline_count": baseline_count,
        "max_workers": workers, "seed": seed, "stage_seeds": stage_seeds,
        "sequence_signature": payload["summary"]["sequence_signature"],
    }


def doctor(config: dict) -> None:
    from .dsh_system import DSHSystem

    system = DSHSystem(**config["system"])
    env = {**os.environ, "PATH": str(Path(system.dsh_bin).parent) + os.pathsep + os.environ.get("PATH", "")}
    version = subprocess.run([system.dsh_bin, "--version"], env=env, capture_output=True, text=True, timeout=30)
    policy = check_memory_policy(config)
    print(json.dumps({
        "python": sys.version.split()[0], "python_prefix": sys.prefix,
        "bench_root": str(BENCH), "dsh_bin": system.dsh_bin,
        "dsh_version": version.stdout.strip(), "provider": system.provider, "model": system.model,
        "base_url": system.base_url or "https://api.deepseek.com/anthropic",
        "api_key_available": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "docker": shutil.which("docker"), "pricing": system.pricing,
        "dependencies": {name: bool(importlib.util.find_spec(name)) for name in (
            "litellm", "pydantic", "texasholdem", "datasets", "minisweagent", "lifelines")},
        **({"memory_policy": policy} if policy is not None else {}),
    }, indent=2))
    system.close()


def runtime_provenance(config: dict) -> tuple[dict, dict, dict]:
    from .dsh_system import DSHSystem

    checkout, resources = benchmark_resources()
    local = DSHSystem(**config["system"])
    binary = Path(local.dsh_bin)
    env = {**os.environ, "PATH": str(binary.parent) + os.pathsep + os.environ.get("PATH", "")}
    package = binary.resolve().parent.parent
    dependencies = package / "node_modules" / "@deepseek-ai"
    dsh_sources = {}
    for candidate in (package, *(dependencies / name for name in (
        "dsh-llm", "dsh-llm-deepseek", "dsh-llm-deepseek-api-key", "dsh-agent-loop", "dsh-tools",
        "dsh-sdk-jsonrpc-server", "dsh-session-persistence-jsonl",
    ))):
        if (candidate / "package.json").is_file():
            for path in [candidate / "package.json", *sorted((candidate / "lib").glob("*.js"))]:
                dsh_sources[path.relative_to(package).as_posix()] = sha256(path)
    runtime = {
        **checkout, "dsh_bin": str(binary), "dsh_executable_sha256": sha256(binary),
        "dsh_version": subprocess.check_output([str(binary), "--version"], env=env, text=True, timeout=30).strip(),
        "dsh_source_sha256": dsh_sources, "python": sys.version, "bench_root": str(BENCH),
    }
    sources = {name: sha256(ROOT / name) for name in SOURCE_PATHS}
    sources.update({str(path): sha256(path) for path in map(Path, config["system"]["patches"])})
    if config.get("stateful_only"):
        from .memory_policy import CONTRACT_PATH

        sources["memory_policy.py"] = sha256(ROOT / "memory_policy.py")
        sources["skill_rsi_rl/contracts.py"] = sha256(CONTRACT_PATH)
    packages = {
        name: item.version for item in importlib.metadata.distributions() if (name := item.metadata.get("Name"))
    }
    return runtime, resources, {"files": sources, "package_versions": packages}


def check_memory_policy(config: dict) -> dict | None:
    if config.get("memory_policy") is None:
        return None
    from .memory_policy import MemoryPolicy

    try:
        return MemoryPolicy(config["memory_policy"]).check()
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Memory policy health check failed: {exc}") from exc


def authorize_execution_revision(manifest: dict, fingerprint: str, frozen: dict, authorize: bool) -> None:
    if fingerprint == manifest["fingerprint"]:
        return
    revisions = manifest.get("execution_revisions", {})
    if fingerprint in revisions:
        return
    if not authorize:
        raise ValueError("Configuration, source, benchmark assets, or treatment changed. Use a new --name.")
    immutable = set(frozen) - {"sources"}
    changed_inputs = sorted(key for key in immutable if manifest.get(key) != frozen[key])
    old_sources, new_sources = manifest["sources"], frozen["sources"]
    changes = {path: {"before": old_sources.get(path), "after": new_sources.get(path)}
               for path in old_sources.keys() | new_sources.keys()
               if old_sources.get(path) != new_sources.get(path)}
    if changed_inputs or not changes or changes.keys() - RECOVERY_SOURCES:
        changed = ", ".join([*changed_inputs, *sorted(changes)])
        raise ValueError(f"Repair resume only permits DSH runner and bridge source changes; changed: {changed}")
    manifest.setdefault("execution_revisions", {})[fingerprint] = {
        "source_changes": changes, "recorded_at": now(), "policy": RECOVERY_POLICY,
    }


def run_batch(args: argparse.Namespace, config: dict, run_dir: Path, tasks: list[str]) -> None:
    from .export_results import export_results

    check_memory_policy(config)
    runtime, resources, sources = runtime_provenance(config)
    use_benchmark_data()
    profile = args.command
    stateful_only = config.get("stateful_only", False)
    scope = {"evaluation_mode": "stateful_only"} if stateful_only else {}
    comparison = canonical_hash({
        "profile": profile, "system": config["system"], "pricing_provenance": config.get("pricing_provenance"),
        "sources": sources, "runtime": runtime, "resources": resources, "capture_schema_version": 1,
        **scope,
    })
    skill = None
    loading = {"sha256": None, "method": None, "path": None}
    if args.skill:
        skill = args.skill.read_bytes()
        if hashlib.sha256(skill).hexdigest() != args.skill_sha256:
            raise ValueError("Selected SKILL.md does not match --skill-sha256")
        if not skill.decode("utf-8").strip():
            raise ValueError("Selected SKILL.md must contain UTF-8 Markdown")
        loading = {"sha256": args.skill_sha256, "method": "system_prompt_section", "path": "skill/SKILL.md"}
    config["skill_loading"] = loading
    treatment = {"memory_policy": config.get("memory_policy")} if stateful_only else {}
    fingerprint = canonical_hash({
        "comparison_fingerprint": comparison, "skill_sha256": loading["sha256"], **treatment,
    })
    frozen = {
        "profile": profile, "system": config["system"], "pricing_provenance": config.get("pricing_provenance"),
        "skill_loading": loading, "runtime": runtime, "sources": sources["files"],
        "benchmark_resources": resources, "package_versions": sources["package_versions"],
        **scope, **treatment,
    }
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        authorize_execution_revision(manifest, fingerprint, frozen, args.resume_repair)
    else:
        if args.resume_repair:
            raise ValueError("--resume-repair requires an existing experiment")
        manifest = {
            **frozen,
            "schema_version": 1, "run_name": run_dir.name, "display_name": f"DSH · {config['system']['model']}",
            "harness": "dsh", "fingerprint": fingerprint, "comparison_fingerprint": comparison,
            "training_permission": "evaluation_only", "created_at": now(), "tasks": {},
            "capture": {"schema_version": 1, "request_layer": "native_options", "endpoint_verified": False},
        }
    provenance = {
        "fingerprint": manifest["fingerprint"], "execution_fingerprint": fingerprint,
        "accepted_executions": (manifest["fingerprint"], *manifest.get("execution_revisions", {})),
    }
    if skill is not None:
        frozen = run_dir / loading["path"]
        frozen.parent.mkdir(exist_ok=True)
        if frozen.exists():
            if sha256(frozen) != loading["sha256"]:
                raise ValueError("The frozen skill in the batch directory has changed")
        else:
            pending = frozen.with_suffix(".md.tmp")
            pending.write_bytes(skill)
            pending.replace(frozen)
    for task in tasks:
        manifest["tasks"].setdefault(task, {"status": "pending"})
    save_json(manifest_path, manifest)
    exhausted = []
    for task in tasks:
        manifest["tasks"][task] = {"status": "running"}
        save_json(manifest_path, manifest)
        try:
            workers = args.docker_workers if task in DOCKER_TASKS and args.docker_workers else args.workers
            details = run_task(task, run_dir, profile, config, workers, provenance)
        except IncompleteRecoveredJobs as exc:
            exhausted.append(task)
            manifest["tasks"][task] = {"status": "failed", "error": str(exc), "recovery_exhausted": True}
            save_json(manifest_path, manifest)
            continue
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            manifest["tasks"][task] = {"status": status, "error": str(exc)}
            save_json(manifest_path, manifest)
            raise
        manifest["tasks"][task] = {"status": "completed", **details}
        save_json(manifest_path, manifest)
    export_profile = "full" if profile == "full" and all(
        manifest["tasks"].get(task, {}).get("status") == "completed" for task in TASKS
    ) else "quick"
    if not stateful_only and any((run_dir / "tasks").glob("*.json.gz")):
        exported_profile = export_profile
        rank_unavailable = None
        try:
            export_results(BENCH, run_dir, profile=export_profile)
        except ValueError as exc:
            if export_profile != "full" or not str(exc).endswith("pricing is incomplete"):
                raise
            rank_unavailable = str(exc)
            print(f"Official ranked export unavailable: {rank_unavailable}. Writing raw summaries.", flush=True)
            exported_profile = "quick"
            export_results(BENCH, run_dir, profile=exported_profile)
        manifest["export"] = {"profile": exported_profile, "rank_unavailable": rank_unavailable}
        save_json(manifest_path, manifest)
    if exhausted:
        raise IncompleteRecoveredJobs(f"Unfinished tasks after bounded retries: {', '.join(exhausted)}")
    print(f"Results: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "quick", "full", "export"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "official.json")
    parser.add_argument("--name", help="Result directory name; reuse it to resume the same configuration")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--docker-workers", type=int, help="Worker limit for Codebase Adaptation and Sales Prediction")
    parser.add_argument("--tasks", nargs="+", choices=TASKS)
    parser.add_argument("--run-dir", type=Path, help="Existing result directory to export")
    parser.add_argument("--skill", type=Path, help="Frozen SKILL.md for a separate validation batch")
    parser.add_argument("--skill-sha256", help="SHA-256 of the approved SKILL.md")
    parser.add_argument("--context-edit", action="store_true",
                        help="Enable model-controlled context editing and public source lookup")
    parser.add_argument("--stateful-only", action="store_true", help="Run the stateful evaluation arm without isolated baselines")
    parser.add_argument("--memory-policy", type=Path, help="Frozen loopback policy config for a stateful evaluation arm")
    parser.add_argument("--resume-repair", action="store_true",
                        help="Record a runner/bridge source repair when resuming an existing batch")
    args = parser.parse_args()
    args.config = args.config.resolve()
    if args.run_dir is not None:
        args.run_dir = args.run_dir.resolve()
    if args.skill is not None:
        args.skill = args.skill.resolve()
    if args.memory_policy is not None:
        args.memory_policy = args.memory_policy.resolve()
    if args.stateful_only and (args.command not in {"quick", "full"} or args.resume_repair):
        parser.error("--stateful-only requires quick or full and starts failed attempts in fresh sessions")
    if args.memory_policy is not None and args.command != "doctor" and (
        not args.stateful_only or not args.name or args.skill is not None
    ):
        parser.error("--memory-policy requires --stateful-only, an explicit --name, and no --skill")
    os.chdir(BENCH)
    if args.command == "export":
        from .export_results import export_results

        if not args.run_dir:
            parser.error("export requires --run-dir")
        manifest = args.run_dir / "manifest.json"
        if manifest.is_file() and load_json(manifest).get("evaluation_mode") == "stateful_only":
            parser.error("Official export requires isolated baselines. Compare stateful arms using their checkpoints.")
        export_results(BENCH, args.run_dir.resolve())
        return
    if args.workers < 1 or (args.docker_workers is not None and args.docker_workers < 1):
        parser.error("Worker counts must be positive")
    if (args.skill is None) != (args.skill_sha256 is None):
        parser.error("--skill and --skill-sha256 are required together")
    if args.skill is not None and (not args.name or not re.fullmatch(r"[0-9a-f]{64}", args.skill_sha256)):
        parser.error("--skill requires an explicit --name and lowercase SHA-256")
    config = load_config(args.config.resolve())
    if args.context_edit:
        config["system"]["context_edit"] = True
    config["stateful_only"] = args.stateful_only
    config["memory_policy"] = None
    if args.memory_policy is not None:
        from .memory_policy import validate_config

        try:
            config["memory_policy"] = validate_config(load_json(args.memory_policy))
        except (OSError, ValueError) as exc:
            parser.error(f"Invalid --memory-policy: {exc}")
    if args.command == "doctor":
        doctor(config)
        return
    if not os.environ.get("DEEPSEEK_API_KEY"):
        parser.error("Set DEEPSEEK_API_KEY or configure the local DSH credential store")
    profile = args.command
    tasks = args.tasks or (["exploitable_poker"] if profile == "quick" else list(TASKS))
    if profile == "quick" and tasks != ["exploitable_poker"]:
        parser.error("quick uses the official Poker quick_test schedule")
    if profile == "full" and set(tasks) & DOCKER_TASKS:
        if not shutil.which("docker"):
            parser.error("Full codebase/sales evaluation requires a running Docker engine and docker on PATH")
    model = re.sub(r"[^a-zA-Z0-9._-]+", "-", config["system"]["model"])
    name = args.name or f"dsh-{model}-{profile}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    if Path(name).name != name or name in {".", ".."}:
        parser.error("--name must be a directory name")
    run_dir = Path(os.environ.get("SKILL_RL_RESULTS", ROOT.parents[1] / "runs" / "dsh")).resolve() / name
    if run_dir.is_symlink():
        parser.error("Result directory cannot be a symlink")
    os.umask(0o077)
    run_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(run_dir / ".lock"):
        run_batch(args, config, run_dir, tasks)


if __name__ == "__main__":
    main()
