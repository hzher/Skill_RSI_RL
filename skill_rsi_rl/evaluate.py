"""Pair complete native CL-Bench sequences across frozen memory policies."""

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
from statistics import fmean
import subprocess
import sys

from .compare import clustered_delta
from .contracts import digest, validate_memory
from .harnesses import activate_benchmark
from .harnesses.memory_policy import MemoryPolicy, validate_config

PROJECT = Path(__file__).resolve().parents[1]
OFFICIAL_DSH_CONFIG = Path(__file__).parent / "harnesses/configs/official.json"
TASKS = ("exploitable_poker", "blind_spectrum_monitoring", "cohort_studies",
         "database_exploration", "codebase_adaptation", "sales_prediction")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def make_plan(args):
    root, commit = activate_benchmark()
    from src.registry import get_task_class
    from src.tasks.schedules import get_task_schedule

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.tag):
        raise ValueError("Use a tag containing letters, digits, dots, dashes or underscores")
    if args.task and args.all_tasks:
        raise ValueError("Choose --task or --all-tasks")
    tasks = TASKS if args.all_tasks else args.task or (TASKS[:1] if args.profile == "quick" else TASKS[:2])
    if len(set(tasks)) != len(tasks) or (args.profile == "quick" and list(tasks) != [TASKS[0]]):
        raise ValueError("Tasks must be distinct; the quick profile uses the official Poker quick_test schedule")
    if args.bootstrap_samples < 100:
        raise ValueError("At least 100 bootstrap samples are required")
    schedule_id = "quick_test" if args.profile == "quick" else "default"
    inputs = {"base": args.base_policy, "offline": args.offline_policy}
    if args.online_policy:
        inputs["online"] = args.online_policy
    policies = {name: validate_config(read(path)) for name, path in inputs.items()}
    if len({p["model"] for p in policies.values()}) != 1 or len({p["checkpoint"] for p in policies.values()}) != len(policies):
        raise ValueError("Use the same student base model and distinct checkpoint identities in the policy arms")
    if any(p["endpoint"].endswith("/sample") for p in policies.values()):
        raise ValueError("Evaluation policy endpoints must select greedy actions")

    config_path = args.config
    if config_path is None:
        if args.harness != "dsh":
            raise ValueError("Supply a native executor configuration with --config")
        config_path = OFFICIAL_DSH_CONFIG
    config = read(config_path)
    if "benchmark" in config:
        # A training recipe can supply the native executor for a held-out evaluation.
        # The evaluation profile owns its schedules, environment seeds and initial state.
        benchmark = config["benchmark"]
        config = {"harness": benchmark["harness"], "system_params": benchmark["system_params"]}
    if "system" in config and args.harness == "dsh":
        from .harnesses.run import load_config
        system = load_config(config_path)["system"]
        if config_path.resolve() == OFFICIAL_DSH_CONFIG.resolve():
            system["base_url"] = config["system"].get("base_url") or "https://api.deepseek.com/anthropic"
    else:
        system = config["system_params"]
    if config.get("harness", args.harness) != args.harness or not isinstance(system, dict) or not system.get("model"):
        raise ValueError("Executor configuration must match --harness and specify its model")
    if set(system) & {"memory_policy", "api_key", "capture_identity", "runtime_root"}:
        raise ValueError("Use the configured native executor, external credentials and the arm policy files")
    memory = config.get("initial_memory", {})
    validate_memory(memory)
    task_params = dict(config.get("task_params", {}))
    if task_params.get("schedule", schedule_id) != schedule_id or set(task_params) & {"run_index", "rollout_index"}:
        raise ValueError("The profile owns the official schedule and sequence indices")
    task_params["schedule"] = schedule_id
    cases = []
    for task in tasks:
        defaults = get_task_schedule(task, schedule_id).run_defaults
        if defaults is None:
            raise ValueError(f"Missing official run defaults for {task}/{schedule_id}")
        seed = task_params.get("seed", inspect.signature(get_task_class(task).__init__).parameters["seed"].default)
        if type(seed) is not int or seed < 0:
            raise ValueError(f"Missing a valid environment seed for {task}")
        count = 1 if args.profile == "quick" else defaults.runs
        for index in range(count):
            cases.append({"task": task, "pair": index, "environment_seed": seed, "sampling_seed": index,
                          "run_index": index, "mode": defaults.mode})
    pkg = Path(__file__).parent
    source_files = [Path(__file__), pkg / "compare.py", pkg / "contracts.py",
                    *sorted((pkg / "harnesses").rglob("*.py")), *sorted((pkg / "harnesses/plugins").rglob("*.mjs"))]
    provenance = {str(f.relative_to(pkg)): file_hash(f) for f in source_files}
    resources = {}
    # External datasets and executor patches are outside the locked benchmark snapshot.
    if "dataset_path" in task_params:
        path = root / task_params["dataset_path"]
        resources["dataset_path"] = file_hash(path)
    for patch in system.get("patches", []):
        resources[str(patch)] = file_hash(Path(patch))
    return {"harness": args.harness, "system_params": system, "initial_memory": memory,
            "task_params": task_params, "profile": args.profile, "bench_commit": commit,
            "sources_sha256": digest(provenance), "package_lock_sha256": file_hash(PROJECT / "sources.lock.json"),
            "resources_sha256": resources, "config_sha256": file_hash(config_path),
            "policies": {"native": None, **policies}, "cases": cases,
            "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": 17}


def summarize_rollout(result, case, plan, policy, attempt):
    policy_id = policy["checkpoint"] if policy else None
    identity = {"harness": plan["harness"], "task": case["task"], "status": "completed",
                "policy_id": policy_id, "sample": False, "bench_commit": plan["bench_commit"],
                **{k: case[k] for k in ("environment_seed", "sampling_seed", "run_index")}}
    if any(result.get(k) != v for k, v in identity.items()):
        raise ValueError("Completed sequence has a different execution or policy identity")
    outcomes = result["outcomes"]
    sequence = [(o["instance_index"], o["instance_id"]) for o in outcomes]
    if (not sequence or len({key for _, key in sequence}) != len(sequence)
            or len({i for i, _ in sequence}) != len(sequence)
            or any(type(i) is not int or i < 0 or not isinstance(key, str) or not key for i, key in sequence)
            or any(type(o["success"]) is not bool or type(o["reward"]) not in (int, float)
                   or not math.isfinite(o["reward"]) for o in outcomes)):
        raise ValueError("Incomplete official outcomes or invalid instance identities")
    decisions = result["decisions"]
    if bool(policy) != bool(decisions):
        raise ValueError("Policy decision coverage disagrees with the evaluation arm")
    applied = fallbacks = injected = 0
    for decision in decisions:
        receipt = decision["receipt"]
        if (decision["policy_id"] != policy_id or receipt["checkpoint"] != policy_id
                or receipt["status"] not in ("applied", "fallback")
                or decision["applied"] != (receipt["status"] == "applied")):
            raise ValueError("Policy receipt differs from the frozen arm")
        applied += int(decision["applied"])
        fallbacks += int(receipt["status"] == "fallback")
        injected += int(bool(receipt["injected"]))
    trace = attempt / "trace.json.gz"
    if Path(result["trace_path"]).name != trace.name or not trace.is_file():
        raise ValueError("Official trace is missing")
    return {"sequence_sha256": digest(sequence), "instances": len(sequence),
            "reward": fmean(o["reward"] for o in outcomes), "success": fmean(o["success"] for o in outcomes),
            "calls": len(decisions), "applied": applied, "fallback": fallbacks, "injected": injected,
            "usage": result["usage"]}


def run_arm(directory, case, plan, name):
    from .harnesses.run import save_json

    policy = plan["policies"][name]
    directory.mkdir(parents=True, exist_ok=True)
    seal = directory / "accepted.json"
    if seal.is_file():
        accepted = read(seal)
        attempt = directory / accepted["attempt"]
        if (file_hash(attempt / "rollout.json") != accepted["rollout_sha256"]
                or file_hash(attempt / "trace.json.gz") != accepted["trace_sha256"]):
            raise ValueError(f"Sealed evaluation evidence changed at {attempt}")
    else:
        previous = sorted(directory.glob("attempt-*"))
        if previous and (previous[-1] / "rollout.json").is_file() and not (previous[-1] / "error.json").exists():
            attempt = previous[-1]
        else:
            attempt = directory / f"attempt-{len(previous) + 1:06d}"
            attempt.mkdir()
            config = {"harness": plan["harness"], "system_params": plan["system_params"],
                      "task": case["task"], "task_params": plan["task_params"], "run_index": case["run_index"],
                      "run_mode": case["mode"], "initial_memory": plan["initial_memory"]}
            command = [sys.executable, "-B", "-m", "skill_rsi_rl.harnesses.runner",
                       "--config", str(attempt / "config.json"), "--output", str(attempt),
                       "--environment-seed", str(case["environment_seed"]), "--sampling-seed", str(case["sampling_seed"])]
            if policy:
                config.update(endpoint=policy["endpoint"], policy_model=policy["model"], policy_timeout=policy["timeout"])
                command.extend(["--policy-id", policy["checkpoint"]])
            save_json(attempt / "config.json", config)
            with (attempt / "runner.log").open("w") as log:
                status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            if status:
                err = attempt / "error.json"
                reason = read(err).get("message") if err.exists() else f"exit {status}"
                raise ValueError(f"{attempt.name} failed: {reason} (see runner.log)")
    summary = summarize_rollout(read(attempt / "rollout.json"), case, plan, policy, attempt)
    if not seal.exists():
        save_json(seal, {"attempt": attempt.name, "rollout_sha256": file_hash(attempt / "rollout.json"),
                         "trace_sha256": file_hash(attempt / "trace.json.gz")})
    return {**summary, "artifact": str(attempt.relative_to(directory.parents[2])),
            "attempt_count": len(list(directory.glob("attempt-*")))}


def task_report(pairs, names, samples, seed):
    matched = []
    for pair in pairs:
        arms = pair["arms"]
        if pair["errors"]:
            continue
        if len({row["sequence_sha256"] for row in arms.values()}) != 1:
            pair["errors"]["pair"] = "official instance sequence mismatch"
            continue
        if any(arms[name]["applied"] == 0 for name in names if name != "native"):
            pair["errors"]["pair"] = "no verified policy delivery in a treatment arm"
            continue
        matched.append(arms)
    contrasts = {}
    for i, arm in enumerate(names):
        for other in names[:i]:
            deltas = {}
            for metric, scale in (("reward", 1), ("success", 100)):
                observations = [[scale * (p[arm][metric] - p[other][metric])] for p in matched]
                deltas["success_pp" if metric == "success" else metric] = clustered_delta(observations, samples, seed)
            contrasts[f"{arm}_vs_{other}"] = deltas
    arms = {}
    for name in names:
        completed = [p["arms"][name] for p in pairs if name in p["arms"]]
        usage = [row["usage"] for row in completed]
        totals = {k: sum(u[k] for u in usage) for k in
                  ("call_count", "input_tokens", "output_tokens", "total_tokens", "cost_usd",
                   "missing_total_token_count", "unpriced_call_count")}
        arms[name] = {"completed_sequences": len(completed), "matched_sequences": len(matched),
                      "recorded_usage": totals,
                      "reward": fmean(p[name]["reward"] for p in matched) if matched else None,
                      "success": fmean(p[name]["success"] for p in matched) if matched else None,
                      **{k: sum(row[k] for row in completed) for k in ("instances", "calls", "applied", "fallback", "injected")}}
    return {"planned_pairs": len(pairs), "matched_pairs": len(matched),
            "missing_pairs": len(pairs) - len(matched), "arms": arms, "contrasts": contrasts, "pairs": pairs}


def evaluate(plan, directory):
    from .harnesses.run import exclusive_lock, save_json

    directory = directory.resolve()
    with exclusive_lock(directory / ".eval.lock"):
        manifest = directory / "manifest.json"
        if manifest.exists():
            if read(manifest) != plan:
                raise ValueError(f"Configuration or source differs from {manifest}; use a fresh tag")
        else:
            if any(p.name != ".eval.lock" for p in directory.iterdir()):
                raise ValueError(f"Existing evaluation directory has no manifest: {directory}")
            save_json(manifest, plan)
        grouped = {}
        for case in plan["cases"]:
            row = {**case, "arms": {}, "errors": {}}
            grouped.setdefault(case["task"], []).append(row)
            for arm in plan["policies"]:
                path = directory / case["task"] / f"pair-{case['pair']:04d}" / arm
                try:
                    row["arms"][arm] = run_arm(path, case, plan, arm)
                except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
                    row["errors"][arm] = {"message": str(exc), "artifacts": str(path.relative_to(directory))}
        tasks = {task: task_report(pairs, list(plan["policies"]), plan["bootstrap_samples"], plan["bootstrap_seed"])
                 for task, pairs in grouped.items()}
        report = {"status": "completed" if all(t["missing_pairs"] == 0 for t in tasks.values()) else "incomplete",
                  "manifest_sha256": digest(plan), "harness": plan["harness"], "profile": plan["profile"],
                  "confidence_unit": "paired_complete_rollout", "tasks": tasks}
        save_json(directory / "paired-report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    for name in ("plan", "check", "run"):
        mode.add_argument(f"--{name}", dest="mode", action="store_const", const=name)
    parser.set_defaults(mode="plan")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--harness", choices=["dsh", "codex", "claude"], default="dsh")
    parser.add_argument("--config", type=Path,
                        help="native executor settings, or a training recipe used only for its executor configuration")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--task", choices=TASKS, action="append")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument("--offline-policy", "--trained-policy", dest="offline_policy", type=Path, required=True)
    parser.add_argument("--online-policy", type=Path)
    parser.add_argument("--output", type=Path, help="evaluation directory, reused only for identical inputs")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    plan = make_plan(args)
    output = args.output or PROJECT / "runs/evaluations" / f"{args.tag}-{args.profile}"
    if args.mode == "plan":
        print(json.dumps(plan, indent=2))
        return 0
    if args.mode == "check":
        for name, policy in plan["policies"].items():
            if policy:
                print(name, json.dumps(MemoryPolicy(policy).check()))
        return 0
    report = evaluate(plan, output)
    print(json.dumps({"status": report["status"], "report": str(output / "paired-report.json"),
                      "matched_pairs": {task: data["matched_pairs"] for task, data in report["tasks"].items()}}))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
