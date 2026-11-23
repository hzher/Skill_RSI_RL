"""Score all supported actions from the same verified DSH decision state."""

from argparse import ArgumentParser
import json
import math
import os
from pathlib import Path
import re

from .branch_data import build_state, initial_memory
from .contracts import digest
from .replay_spec import write_spec


PROTOCOL = "official_dsh_next_instances_first_action_fixed_recall_v1"
TASKS = ("exploitable_poker", "blind_spectrum_monitoring")
HARNESS = Path(__file__).resolve().parent / "harnesses"


def _option_meta(option):
    action = option["action"]
    return {"label": option["label"], "op": action["op"], "action_sha256": digest(action)}


def _plan_group(start, state):
    return {"state_id": state["state_id"], "scenario_id": state["scenario_id"],
            "receipt": start.receipt, "targets": [list(target) for target in start.target_outcomes],
            "options": [_option_meta(option) for option in state["options"]]}


def _validate_result(result, group, option, executor=None):
    if (result["boundary_sha256"] != group["receipt"]["boundary_sha256"] or
            result["state_sha256"] != group["receipt"]["state_sha256"] or
            result["score_window_sha256"] != group["receipt"]["score_window_sha256"] or
            result["action_sha256"] != option["action_sha256"] or
            result["target_outcomes"] != group["targets"]):
        raise ValueError("Stored candidate differs from the frozen boundary, action or targets")
    if result["status"] in {"failed", "unscorable"}:
        return False
    if result["status"] != "scored":
        raise ValueError("Unknown status in saved candidate receipt")
    if executor is not None and result["executor"]["model"] != executor["model"]:
        raise ValueError("Stored candidate was scored by a different executor model")
    outcomes = result["outcomes"]
    if ([[row["instance_id"], row["instance_index"]] for row in outcomes] != group["targets"] or
            any(type(row["native_success"]) is not bool or type(row["reward"]) not in (float, int)
                or not math.isfinite(row["reward"]) for row in outcomes) or
            type(result["native_success"]) is not bool or
            type(result["reward"]) not in (float, int) or not math.isfinite(result["reward"])):
        raise ValueError("Stored candidate does not have the frozen official outcomes")
    if result["native_success"] != all(row["native_success"] for row in outcomes) or not math.isclose(
            result["reward"], math.fsum(row["reward"] for row in outcomes) / len(outcomes)):
        raise ValueError("Stored group score differs from its official outcomes")
    return True


def _load_authorization(path, source):
    approved = json.loads(path.read_text())
    if not isinstance(approved, dict):
        raise ValueError("Authorization must be a JSON object")
    reference = approved.get("authorization_ref")
    if (approved.get("training_permission") != "weight_training" or not isinstance(reference, str) or
            not reference.strip() or approved.get("batch_manifest_sha256") != source["batch_manifest_sha256"]):
        raise ValueError("Authorization must identify this source manifest and explicitly allow weight training")
    tasks = approved.get("tasks")
    if tasks is not None and (not isinstance(tasks, list) or any(task not in TASKS for task in tasks)
                              or source["task"] not in tasks):
        raise ValueError("Authorization does not cover this source task")
    return reference


def _saved_scoring_plan(directory, expected):
    plan = json.loads((directory / "branch-plan.json").read_text())
    executor = plan.get("executor") if isinstance(plan, dict) else None
    if (not isinstance(executor, dict) or not isinstance(executor.get("model"), str)
            or not executor["model"].strip() or not isinstance(executor.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", executor["sha256"]) is None):
        raise ValueError("Saved plan is missing its frozen executor identity")
    if plan != {**expected, "executor": executor}:
        raise ValueError("Saved scoring plan differs from the verified archive, boundaries, or parameters")
    return plan


def _executor_config(config_path, harness_dir):
    from .harnesses.run import load_config, runtime_provenance

    config = load_config(config_path or harness_dir / "configs/official.json")
    runtime, _, code = runtime_provenance(config)
    sources = (Path(__file__), Path(__file__).with_name("branch_data.py"),
               Path(__file__).with_name("contracts.py"), harness_dir / "replay_branch.py",
               harness_dir / "historical_replay.py")
    system = config["system"]
    return system, {"model": system["model"], "sha256": digest([
        system, runtime, code, *(file.read_text() for file in sources),
    ])}


def _score_groups(groups, directory, plan, system=None, execute=None, save=None, *,
                  retry_failed=False, complete_only=False):
    manifest = directory / "branch-plan.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != plan:
            raise ValueError("Output belongs to a different replay or executor configuration")
    else:
        if execute is None:
            raise FileNotFoundError(manifest)
        save(manifest, plan)
    executor = plan.get("executor")
    scored_states = []
    report_groups = []
    for (start, state), group in zip(groups, plan["groups"], strict=True):
        root = directory / "branches" / group["state_id"]
        if execute is not None:
            root.mkdir(parents=True, exist_ok=True)
        scores = []
        summary = []
        for action, option in zip(state["options"], group["options"], strict=True):
            label = option["label"]
            receipt = root / f"{label}.json"
            if receipt.is_file():
                prior = json.loads(receipt.read_text())
                successful = _validate_result(prior, group, option, executor)
            else:
                prior, successful = None, False
            if execute is None and not successful:
                if not complete_only:
                    raise ValueError(f"Missing or incomplete candidate receipt: {receipt}")
                if prior is None:
                    summary.append({**option, "status": "missing", "reason": "no_receipt",
                                    "reward": None, "native_success": None, "queries": 0})
                    continue
            if prior is None or (retry_failed and not successful):
                if prior is not None:
                    archive = root / f"{label}.previous-{digest(prior)[:16]}.json"
                    if not archive.exists():
                        save(archive, prior)
                prior = execute(start, action=action["action"], system_params=system, runtime_root=root / label)
                successful = _validate_result(prior, group, option, executor)
                save(receipt, prior)
            if successful:
                scores.append({"reward": prior["reward"], "native_success": prior["native_success"]})
            summary.append({**option, "status": prior["status"], "reason": prior.get("reason"),
                            "reward": prior["reward"], "native_success": prior["native_success"],
                            "queries": prior["queries"]})
        complete = len(scores) == len(group["options"])
        if complete:
            scored_states.append({**state, "scores": scores})
        report_groups.append({"state_id": state["state_id"], "complete": complete,
                              "varying_reward": complete and len({score["reward"] for score in scores}) > 1,
                              "actions": summary})
    report = {"score_protocol": PROTOCOL, "plan_sha256": digest(plan), "complete_groups": len(scored_states),
              "source_training_permission": plan["source"]["source_training_permission"],
              "requested_groups": len(groups), "groups": report_groups}
    if save is not None:
        save(directory / "branch-report.json", report)
    return report, scored_states


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--cut", type=int, action="append", required=True)
    parser.add_argument("--score-instances", type=int, default=2)
    parser.add_argument("--max-queries", type=int, default=24)
    parser.add_argument("--seed-memory", action="store_true", help="seed from deterministic past public feedback writes")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="make real executor requests and save receipts")
    mode.add_argument("--export-scored", action="store_true", help="export from completed receipts without an executor")
    mode.add_argument("--check-executor", action="store_true", help="print executor identity without provider requests")
    parser.add_argument("--config", type=Path, help="DSH executor config for scoring (default: bundled official.json)")
    parser.add_argument("--expected-executor-sha256")
    parser.add_argument("--complete-only", action="store_true",
                        help="offline export of verified complete groups from a partially scored rollout")
    parser.add_argument("--output", type=Path, help="branch scoring directory, already populated for --export-scored")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--replay-spec", type=Path)
    parser.add_argument("--dataset-id")
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    args = parser.parse_args()
    if args.execute and not args.output:
        parser.error("--execute requires --output")
    if args.expected_executor_sha256 is not None and not args.execute:
        parser.error("--expected-executor-sha256 requires --execute")
    if args.complete_only and not args.export_scored:
        parser.error("--complete-only requires --export-scored")
    if args.export_scored and (not args.output or not args.replay_spec or args.retry_failed):
        parser.error("--export-scored requires --output and --replay-spec, with no --retry-failed")
    if args.replay_spec and not ((args.execute or args.export_scored) and args.authorization and args.dataset_id):
        parser.error("--replay-spec requires a scoring/export mode, --authorization and --dataset-id")
    try:
        from .harnesses.historical_replay import load_completed_source
        from .harnesses.replay_branch import execute_branch, prepare_branch
        from .harnesses.run import exclusive_lock, save_json

        source = load_completed_source(args.archive, args.task, args.run)
        authorization = _load_authorization(args.authorization, source.receipt) if args.authorization else None
        if any(path and path.resolve().is_relative_to(source.batch) for path in (args.output, args.replay_spec)):
            raise ValueError("Write branch artifacts outside the historical archive")
        if args.replay_spec and args.replay_spec.exists():
            raise FileExistsError(args.replay_spec)
        groups = []
        for cut in sorted(set(args.cut)):
            memory = initial_memory(source, cut) if args.seed_memory else {}
            start = prepare_branch(source, cut, memory=memory, score_instances=args.score_instances,
                                   max_queries=args.max_queries)
            state = build_state(source, start)
            if len(state["options"]) < 2:
                raise ValueError(f"Cut {cut} has only one available action")
            groups.append((start, state))
        plan = {"schema_version": 1, "source": source.receipt, "score_protocol": PROTOCOL,
                "max_queries": args.max_queries, "seed_memory": args.seed_memory,
                "groups": [_plan_group(start, state) for start, state in groups]}
        if args.execute or args.check_executor:
            system, plan["executor"] = _executor_config(args.config, HARNESS)
            if args.check_executor:
                print(json.dumps(plan["executor"]))
                return
            if (args.expected_executor_sha256 is not None
                    and plan["executor"]["sha256"] != args.expected_executor_sha256):
                raise ValueError("Executor differs from the batch preflight. No candidates were requested.")
            if not os.environ.get("DEEPSEEK_API_KEY"):
                raise ValueError("Set DEEPSEEK_API_KEY for scored branches")
        if not (args.execute or args.export_scored):
            branches = sum(len(group["options"]) for group in plan["groups"])
            print(json.dumps({**plan, "candidate_branches": branches,
                              "max_logical_queries": branches * args.max_queries}, ensure_ascii=False, indent=2))
            return
        os.umask(0o077)
        output = args.output.resolve(strict=args.export_scored)
        with exclusive_lock(output / ".lock"):
            if args.export_scored:
                plan = _saved_scoring_plan(output, plan)
                report, states = _score_groups(groups, output, plan, complete_only=args.complete_only)
            else:
                report, states = _score_groups(groups, output, plan, system, execute_branch, save_json,
                                               retry_failed=args.retry_failed)
            if args.replay_spec and (len(states) == len(groups) or (args.complete_only and states)):
                skipped = [group["state_id"] for group in report["groups"] if not group["complete"]]
                write_spec(args.replay_spec, {"training_permission": "weight_training", "authorization_ref": authorization,
                           "dataset_id": args.dataset_id, "score_protocol": PROTOCOL, "split": args.split,
                           "plan_sha256": report["plan_sha256"], "states": states,
                           **({"skipped_state_ids": skipped} if skipped else {})})
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if len(states) != len(groups) and not (args.complete_only and states):
            raise SystemExit(1)
    except (OSError, KeyError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
