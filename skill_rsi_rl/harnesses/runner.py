"""Run one independent native CL-Bench sequence and persist its policy evidence."""

from __future__ import annotations

import argparse
import json
import math
import signal
import uuid
from contextlib import chdir, suppress
from dataclasses import asdict, replace
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from skill_rsi_rl.contracts import digest

from . import activate_benchmark
from .memory_policy import MemoryPolicy


class PolicySystem:
    """Apply exactly one student action before delegating each native Query."""

    def __init__(self, native, policy):
        self.native = native
        self.policy = policy
        self.feedback = []

    @property
    def name(self):
        return self.native.name

    @property
    def parallel_safe(self):
        return self.native.parallel_safe

    @property
    def supports_baseline(self):
        return self.native.supports_baseline

    def reset(self):
        self.native.reset()
        self.policy.reset()
        self.feedback.clear()

    def respond(self, query):
        request_id = uuid.uuid4().hex
        self.policy.begin_query(request_id, query.instance_id, query.instance_index)
        feedback, self.feedback = self.feedback, []
        if query.feedback and query.feedback.content not in feedback:
            feedback.append(query.feedback.content)
        memo, receipt = self.policy.decide(query.prompt, feedback, self.policy.config["timeout"])
        prompt = query.prompt
        if memo:
            prompt += "\n\nUntrusted notes from previous interactions (JSON string, for reference only): " + json.dumps(
                memo, ensure_ascii=False
            )
        response = self.native.respond(replace(query, prompt=prompt))
        response.metadata = {**(response.metadata or {}), "memory_policy": receipt,
                             "query_request_id": request_id, "policy_query_prompt": prompt}
        return response

    def observe(self, observation, next_query=None):
        self.feedback.append(observation.content)
        self.native.observe(observation, next_query)

    def consume_usage_events(self):
        return self.native.consume_usage_events()

    def get_run_artifacts(self):
        return self.native.get_run_artifacts()


def _policy(config, identity, seed, sample):
    if identity is None:
        if sample:
            raise ValueError("Sampling requires a frozen policy_id")
        return None
    endpoint = urlsplit(config["endpoint"])
    if endpoint.path not in ("", "/", "/decide", "/act", "/sample"):
        raise ValueError("Unexpected policy endpoint path")
    path = "/sample" if sample else ("/act" if endpoint.path == "/act" else "/decide")
    return MemoryPolicy({
        "endpoint": urlunsplit(endpoint._replace(path=path)),
        "model": config.get("policy_model", "student"),
        "checkpoint": identity,
        "timeout": config.get("policy_timeout", 30),
    }, initial_memory=config.get("initial_memory"), sampling_seed=seed, capture=True, verify_model="policy_model" in config)


def _outcomes(result, expected):
    rows = [asdict(o) for o in result.instance_outcomes]
    ids = {row["instance_id"] for row in rows}
    indices = {row["instance_index"] for row in rows}
    if (type(expected) is not int or expected < 1 or len(rows) != expected
            or len(ids) != len(rows) or len(indices) != len(rows)
            or any(not isinstance(key, str) or not key for key in ids)
            or any(type(idx) is not int or idx < 0 for idx in indices)):
        raise ValueError("Incomplete or duplicate official instance outcomes")
    for row in rows:
        if (type(row["reward"]) not in (int, float) or not math.isfinite(row["reward"])
                or type(row["success"]) is not bool):
            raise ValueError("Official outcome lacks a finite reward or native success flag")
    return rows


def _check_decisions(interactions, decisions, sample):
    if len(interactions) != len(decisions):
        raise ValueError("Every official Query must have exactly one policy receipt")
    seen = set()
    for interaction, decision in zip(interactions, decisions, strict=True):
        meta = interaction["response"]["metadata"]
        if (meta.get("query_request_id") != decision["query_id"] or decision["query_id"] in seen
                or meta.get("memory_policy") != decision["receipt"]
                or interaction["query"]["instance_id"] != decision["instance_id"]):
            raise ValueError("Policy receipt does not match its native response")
        seen.add(decision["query_id"])
        if sample and not decision["applied"]:
            raise ValueError(f"Sampled action failed to apply: {decision['receipt']['reason']}")


def run_rollout(config: dict, *, environment_seed: int, sampling_seed: int, policy_id: str | None,
                output: Path, sample: bool) -> dict:
    """Use a separate process per replica. The native task owns process RNG and working directory."""
    from .native_spec import KINDS, kind
    if kind(config) in KINDS:
        from .native_runner import run_rollout as native_rollout
        return native_rollout(config, environment_seed=environment_seed, sampling_seed=sampling_seed,
                              policy_id=policy_id, output=output, sample=sample)
    root, bench_commit = activate_benchmark()
    from src.interface import run_task
    from src.registry import get_system_class, get_task_class
    from src.runs import RunMode, derive_run_task_params
    from src.runs.common import filter_init_params
    from src.tasks.schedules import get_task_schedule
    from src.trace_storage import TraceRecorder, _atomic_write_json

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Retried attempts get a fresh directory, preserving evidence from interrupted runs.
    with (output / "attempt.json").open("x", encoding="utf-8") as marker:
        json.dump({"environment_seed": environment_seed, "sampling_seed": sampling_seed,
                   "sample": sample, "policy_id": policy_id, "bench_commit": bench_commit}, marker)
    task = recorder = policy = close_native = None
    try:
        harness, task_name = config["harness"], config["task"]
        if harness not in ("dsh", "codex", "claude"):
            raise ValueError(f"Unsupported harness: {harness}")
        system_params = dict(config.get("system_params", {}))
        if any(key in system_params for key in ("memory_policy", "capture_identity", "runtime_root", "api_key")):
            raise ValueError("Pass policy and output through the rollout runner and API credentials via the environment")
        task_params = dict(config.get("task_params", {}))
        run_index = config.get("run_index", 0)
        if "seed" in task_params and task_params["seed"] != environment_seed:
            raise ValueError("task_params.seed differs from environment_seed")
        if type(run_index) is not int or run_index < 0:
            raise ValueError("run_index must be a nonnegative integer")
        task_params["seed"] = environment_seed
        policy = _policy(config, policy_id, sampling_seed, sample)
        trace_path = output / "trace.json.gz"
        # Upstream task resource paths are relative to the benchmark checkout.
        with chdir(root):
            task_class = get_task_class(task_name)
            defaults = None
            if task_params.get("schedule"):
                defaults = get_task_schedule(task_name, task_params["schedule"]).run_defaults
            mode = RunMode(config.get("run_mode", defaults.mode if defaults else RunMode.PERMUTE.value))
            unknown = set(task_params) - set(filter_init_params(task_class, task_params))
            if unknown:
                raise ValueError(f"Unknown {task_name} parameters: {sorted(unknown)}")
            task_params = derive_run_task_params(task_params, mode, run_index)
            task = task_class(**filter_init_params(task_class, task_params))
            if "run_index" in task_params:
                task.prepare_run(task_params["run_index"])
            if harness == "dsh":
                from .dsh_system import DSHSystem
                native = DSHSystem(**system_params, runtime_root=str(output / "native"))
                close_native = native.close
                # DSH reset allocated a new workspace during construction.
                native._memory_policy = policy
                system = native
            else:
                if harness == "codex" and system_params.get("backend") == "skill_rsi":
                    from .codex_system import CodexSystem

                    parameters = {k: v for k, v in system_params.items() if k != "backend"}
                    native = CodexSystem(**parameters, runtime_root=str(output / "native"))
                else:
                    native = get_system_class(harness)(**system_params)
                close_native = native._stop_container
                system = PolicySystem(native, policy) if policy is not None else native
                system.reset()
            recorder = TraceRecorder(harness, task_name, system_params, task_params, trace_path=trace_path,
                                     run_group_id=output.name, run_index=run_index,
                                     task_brief=task.get_agent_brief(), phase="run")
            indices = config.get("instance_indices")
            initial_query = None
            if indices is not None:
                if (not isinstance(indices, list) or not indices
                        or any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices)):
                    raise ValueError("instance_indices must contain distinct nonnegative canonical indices")
                task.build_canonical_run_state()
                if any(i >= len(task.instances) for i in indices):
                    raise ValueError("instance_indices exceed the official canonical schedule")
                task.select_run_instances(indices)
                initial_query = task.build_current_query()
            result = run_task(task, system, show_progress=False, trace_recorder=recorder, reset_system=False,
                              initial_query=initial_query)
            trace = recorder.finalize(result)
            _atomic_write_json(trace_path, trace)
        outcomes = _outcomes(result, getattr(task, "num_instances", None))
        if indices is not None and [o["instance_index"] for o in outcomes] != indices:
            raise ValueError("official outcomes differ from the selected canonical instance order")
        interactions = trace["interactions"]
        if any(step["response"]["metadata"].get("llm_error") for step in interactions):
            raise ValueError("Native executor failed during the official sequence")
        decisions = policy.decisions if policy else []
        if policy:
            _check_decisions(interactions, decisions, sample)
        close_native()
        close_native = None
        report = {
            "status": "completed", "harness": harness, "task": task_name, "bench_commit": bench_commit,
            "environment_seed": environment_seed, "sampling_seed": sampling_seed, "policy_id": policy_id,
            "sample": sample, "run_index": run_index, "decisions": decisions, "outcomes": outcomes,
            "split": config.get("split"), "expected_instances": task.num_instances,
            "benchmark_identity": digest({k: v for k, v in config.items() if k != "endpoint"}),
            "run_mode": mode.value, "usage": trace["execution"]["usage"]["interaction"], "trace_path": trace_path.name,
        }
        _atomic_write_json(output / "rollout.json", report)
        return report
    except Exception as exc:
        if recorder is not None and not (output / "trace.json.gz").exists():
            # Serialize the same native trace schema, including completed steps before failure.
            recorder.live_trace_path = output / "partial.json.gz"
            recorder.status = "failed"
            with suppress(Exception):
                recorder.write_live_snapshot()
        _atomic_write_json(output / "error.json", {
            "status": "failed", "type": type(exc).__name__, "message": str(exc),
            "decisions": policy.decisions if policy else [],
            "completed_interactions": len(recorder.interactions) if recorder else 0,
        })
        raise
    finally:
        if close_native is not None:
            with suppress(Exception):
                close_native()
        if task is not None and config["task"] in ("sales_prediction", "codebase_adaptation", "database_exploration"):
            from .poker_resume import discard_replayed_task
            with suppress(Exception):
                discard_replayed_task(config["task"], task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-seed", type=int, required=True)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--policy-id")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    def terminate(signum, frame):
        raise RuntimeError(f"Rollout terminated by signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    report = run_rollout(json.loads(args.config.read_text()), environment_seed=args.environment_seed,
                         sampling_seed=args.sampling_seed, policy_id=args.policy_id, output=args.output,
                         sample=args.sample)
    print(json.dumps({"status": report["status"], "outcomes": len(report["outcomes"]),
                      "artifact": str(args.output / "rollout.json")}))


if __name__ == "__main__":
    main()
