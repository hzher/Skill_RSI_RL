"""Evaluate greedy base and adapter actions against a frozen, fully scored development replay."""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext
from itertools import combinations
from math import isclose
from pathlib import Path
from statistics import fmean
from tempfile import NamedTemporaryFile

from .compare import clustered_delta
from .contracts import MODEL_PATH, digest
from .replay_io import open_hashed_replay
from .serve import QwenPolicy, _file_digest, checkpoint_policy_ids
from .verify_replay import _inspect_stream, inspect_replay, require_disjoint


def verify_dev_groups(manifest, groups):
    if manifest.get("fixture_only"):
        raise ValueError("Uniform fixture replay has no real tokenizer evidence")
    _inspect_stream(manifest, groups, "dev")


def _stratum(manifest, group):
    if isinstance(group.get("family"), str) and group["family"]:
        return group["family"]
    families = manifest.get("scenario_families")
    index = group.get("state_index")
    if manifest.get("source_kind") == "synthetic" and families and type(index) is int:
        return families[index % len(families)]
    return None


def _one_arm(rows, arm):
    scores = [row[arm] for row in rows]
    successes = sum(score["native_success"] for score in scores)
    return {"states": len(rows), "successes": successes, "success_rate": successes / len(rows),
            "reward_mean": fmean(score["reward"] for score in scores),
            "oracle_reward_regret": fmean(row["oracle_reward"] - row[arm]["reward"] for row in rows),
            "actions": dict(Counter(row[f"{arm}_op"] for row in rows))}


def _summarize(rows, samples, seed):
    by_scenario = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario_id"]].append(row)
    clusters = list(by_scenario.values())
    reward = [[row["trained"]["reward"] - row["base"]["reward"] for row in cluster] for cluster in clusters]
    success = [[100 * (int(row["trained"]["native_success"]) - int(row["base"]["native_success"]))
                for row in cluster] for cluster in clusters]
    return {"base": _one_arm(rows, "base"), "trained": _one_arm(rows, "trained"),
            "choice_change_rate": fmean(row["changed"] for row in rows), "paired_scenarios": len(clusters),
            "paired_deltas": {"reward": clustered_delta(reward, samples, seed),
                              "success_pp": clustered_delta(success, samples, seed)}}


def _counterfactual(groups, rows, samples, seed):
    """Use identical queries that require changing operation across sibling states."""
    arms = ("base", "trained", "noop")
    siblings = defaultdict(list)
    for group, row in zip(groups, rows, strict=True):
        query = json.loads(group["messages"][1]["content"])["query"]
        best = row["oracle_reward"]
        best_options = [option for option, score in zip(group["options"], group["option_scores"], strict=True)
                        if isclose(score["reward"], best, rel_tol=1e-9, abs_tol=1e-12)]
        operations = frozenset(option["action"]["op"] for option in best_options)
        labels = {option["label"] for option in best_options}
        optimal = {arm: isclose(row[arm]["reward"], best, rel_tol=1e-9, abs_tol=1e-12) for arm in arms}
        siblings[row["scenario_id"], query].append((digest(group["messages"]), operations, labels, optimal))

    same_query_sets = duplicate_sets = constant_label_sets = eligible_sets = eligible_states = 0
    passes = Counter()
    deltas = defaultdict(list)
    for (scenario, _), states in siblings.items():
        if len(states) < 2:
            continue
        same_query_sets += 1
        if len({observation for observation, _, _, _ in states}) != len(states):
            duplicate_sets += 1
            continue
        oracle_operations = {operations for _, operations, _, _ in states}
        if not any(left.isdisjoint(right) for left, right in combinations(oracle_operations, 2)):
            continue
        if set.intersection(*(labels for _, _, labels, _ in states)):
            constant_label_sets += 1
            continue
        eligible_sets += 1
        eligible_states += len(states)
        outcomes = {arm: all(optimal[arm] for _, _, _, optimal in states) for arm in arms}
        passes.update(outcomes)
        deltas[scenario].append(100 * (int(outcomes["trained"]) - int(outcomes["base"])))
    return {
        "criterion": "reward-optimal on every sibling, with a disjoint oracle operation pair and no shared optimal label",
        "same_query_sets": same_query_sets,
        "excluded_duplicate_observation_sets": duplicate_sets,
        "excluded_constant_label_sets": constant_label_sets,
        "eligible_sets": eligible_sets,
        "eligible_states": eligible_states,
        "eligible_scenarios": len(deltas),
        "confidence_unit": "scenario",
        "paired_delta_pp": clustered_delta(list(deltas.values()), samples, seed),
        **{arm: {"sets_passed": passes[arm], "pass_rate": passes[arm] / eligible_sets if eligible_sets else None}
           for arm in arms},
    }


def summarize_choices(manifest, groups, base_choices, trained_choices, *, samples=5000, seed=20261118):
    """Summarize native outcomes by matched state, bootstrapping whole scenarios."""
    if samples < 100:
        raise ValueError("At least 100 bootstrap samples are required")
    groups = tuple(groups)
    rows = []
    strata = defaultdict(list)
    for group, base, trained in zip(groups, base_choices, trained_choices, strict=True):
        scores = group["option_scores"]
        if any(type(choice) is not int or not 0 <= choice < len(scores) for choice in (base, trained)):
            raise ValueError("Policy choice is outside the scored option support")
        row = {"scenario_id": group["scenario_id"], "base": scores[base], "trained": scores[trained],
               "noop": scores[0], "noop_op": "noop",
               "base_op": group["options"][base]["action"]["op"], "trained_op": group["options"][trained]["action"]["op"],
               "oracle_reward": max(score["reward"] for score in scores), "changed": base != trained}
        rows.append(row)
        if (stratum := _stratum(manifest, group)) is not None:
            strata[stratum].append(row)
    result = _summarize(rows, samples, seed)
    result.update(reward_delta=result["paired_deltas"]["reward"]["mean"],
                  success_delta_pp=result["paired_deltas"]["success_pp"]["mean"],
                  noop_baseline=_one_arm(rows, "noop"),
                  counterfactual=_counterfactual(groups, rows, samples, seed),
                  strata={name: _summarize(values, samples, seed) for name, values in sorted(strata.items())},
                  confidence_unit="scenario", bootstrap_samples=samples, bootstrap_seed=seed)
    return result


def _training_receipt(path, groups, run_settings, dev_sha):
    trained_sha = run_settings.get("replay_sha256")
    if trained_sha == dev_sha:
        raise ValueError("Development and recorded training replay are the same file")
    receipt = {"training_replay_sha256": trained_sha, "scenario_disjoint_verified": False}
    if path is None:
        return receipt
    report, *train_keys = inspect_replay(path, "train")
    supplied_sha = report["file_sha256"]
    if trained_sha and supplied_sha != trained_sha:
        raise ValueError("Supplied training replay differs from the adapter's recorded training replay")
    dev_keys = ({group["state_id"] for group in groups}, {group["scenario_id"] for group in groups},
                {digest(group["messages"]) for group in groups})
    overlap = require_disjoint(train_keys, dev_keys)
    return {"training_replay_sha256": supplied_sha, "scenario_disjoint_verified": True,
            "split_overlap": overlap,
            "train_replay_matches_adapter_manifest": bool(trained_sha)}


def _adapter_choices(policy, adapter, inputs):
    if adapter is None:
        return [policy.choose_tokens(*tokens) for tokens in inputs]
    policy.model.load_adapter(adapter, "evaluation", is_trainable=False,
                              torch_device=str(policy.device), local_files_only=True)
    try:
        policy.model.set_adapter("evaluation")
        return [policy.choose_tokens(*tokens) for tokens in inputs]
    finally:
        policy.model.set_adapter("default")
        policy.model.delete_adapter("evaluation")


def _parent_comparison(manifest, groups, parent_choices, candidate_choices, parent_path, parent_id, *, samples, seed):
    result = summarize_choices(manifest, groups, parent_choices, candidate_choices, samples=samples, seed=seed)
    arm_names = {"base": "parent", "trained": "candidate"}

    def name_arms(metrics):
        return {arm_names.get(key, key): value for key, value in metrics.items()}

    result = name_arms(result)
    result.update(checkpoint=str(parent_path), policy_id=parent_id,
                  strata={name: name_arms(metrics) for name, metrics in result["strata"].items()},
                  counterfactual=name_arms(result["counterfactual"]))
    return result


def _evaluate_adapters(replay, model, adapters, device, *, train_replay=None, samples=5000, seed=20261118,
                       max_prompt_tokens=4096, on_scored=None, compare_adapter=None):
    if samples < 100 or max_prompt_tokens < 1:
        raise ValueError("Invalid bootstrap sample count or prompt limit")
    with open_hashed_replay(replay) as (manifest, replay_groups, replay_digest):
        groups = list(replay_groups)
        dev_sha = replay_digest()
    verify_dev_groups(manifest, groups)
    run_file = adapters[0].parent / "run.json"
    run = json.loads(run_file.read_text(encoding="utf-8")) if run_file.is_file() else {"settings": {}}
    run_settings = run.get("settings") if isinstance(run, dict) else None
    if not isinstance(run_settings, dict):
        raise ValueError("Adapter training run has invalid settings")
    initial = run.get("initial_adapter")
    if compare_adapter is not None and initial is not None:
        if not isinstance(initial, dict) or initial.get("sha256") != _file_digest(compare_adapter / "adapter_model.safetensors"):
            raise ValueError("Comparison adapter differs from the recorded initial adapter weights")
    training = _training_receipt(train_replay, groups, run_settings, dev_sha)
    all_adapters = adapters if compare_adapter is None else [*adapters, compare_adapter]
    base_id, policy_ids = checkpoint_policy_ids(model, all_adapters)
    parent_id = policy_ids.pop() if compare_adapter is not None else None
    if run_settings.get("base_checkpoint_id") not in (None, base_id):
        raise ValueError("Adapter's recorded base checkpoint differs from the evaluation model")
    policy = QwenPolicy(model, adapters[0], device, max_prompt_tokens)
    inputs = []
    for group in groups:
        tokens = (group["prompt_token_ids"], group["option_token_ids"])
        if policy.action_tokens(group["messages"], group["options"]) != tokens:
            raise ValueError("Development replay tokenization differs from the evaluation tokenizer")
        inputs.append(tokens)
    with policy.model.disable_adapter():
        base = [policy.choose_tokens(*tokens) for tokens in inputs]
    parent_choices = _adapter_choices(policy, compare_adapter, inputs) if compare_adapter is not None else None

    metadata = {"dataset_id": manifest["dataset_id"], "source_kind": manifest["source_kind"],
                "score_protocol": manifest.get("score_protocol"), "development_replay_sha256": dev_sha,
                "base_policy_id": base_id, "holdout": training}
    reports = []
    for index, (adapter, policy_id) in enumerate(zip(adapters, policy_ids, strict=True)):
        trained = _adapter_choices(policy, adapter if index else None, inputs)
        report = {**metadata, "policy_id": policy_id,
                  **summarize_choices(manifest, groups, base, trained, samples=samples, seed=seed)}
        if parent_choices is not None:
            report["parent_comparison"] = _parent_comparison(
                manifest, groups, parent_choices, trained, compare_adapter, parent_id, samples=samples, seed=seed,
            )
        reports.append(report)
        if on_scored is not None:
            on_scored(index, report)
    return reports


def evaluate(replay, model, adapter, device, *, train_replay=None, samples=5000, seed=20261118, max_prompt_tokens=4096):
    adapter = Path(adapter).resolve(strict=True)
    return _evaluate_adapters(replay, model, [adapter], device, train_replay=train_replay, samples=samples,
                              seed=seed, max_prompt_tokens=max_prompt_tokens)[0]


def _epoch_checkpoints(directory, expected_epochs):
    directory = Path(directory).resolve(strict=True)
    checkpoints = []
    for path in directory.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-([0-9]+)", path.name)
        if match is None:
            continue
        epoch = int(match[1])
        if not epoch or not path.is_dir() or path.is_symlink():
            raise ValueError(f"Invalid training checkpoint: {path}")
        checkpoints.append((epoch, path))
    checkpoints.sort()
    epochs = [epoch for epoch, _ in checkpoints]
    if not epochs or len(epochs) != len(set(epochs)):
        raise ValueError(f"Expected training checkpoints with unique epoch numbers in {directory}")
    if expected_epochs is not None and (len(epochs) != expected_epochs or epochs[-1] != expected_epochs):
        raise ValueError(f"Expected checkpoints for epochs 1–{expected_epochs}, found {epochs}")
    return checkpoints


def evaluate_checkpoints(replay, model, directory, device, *, expected_epochs=None, eval_epochs=None,
                         progress=None, compare_adapter=None, **settings):
    checkpoints = _epoch_checkpoints(directory, expected_epochs)
    if compare_adapter is not None:
        compare_adapter = Path(compare_adapter).resolve(strict=True)
        if compare_adapter in (path for _, path in checkpoints):
            raise ValueError("Comparison adapter must be separate from the candidate checkpoints")
    sweep = {}
    if eval_epochs is not None:
        requested = set(eval_epochs)
        if not requested or any(type(epoch) is not int or epoch < 1 for epoch in requested):
            raise ValueError("Evaluation milestones must be positive epochs")
        committed = {epoch for epoch, _ in checkpoints}
        earliest, latest = checkpoints[0][0], checkpoints[-1][0]
        epochs = (requested & committed) | {earliest, latest}
        checkpoints = [(epoch, path) for epoch, path in checkpoints if epoch in epochs]
        sweep = {"requested_epochs": sorted(requested), "evaluated_epochs": sorted(epochs),
                 "unavailable_requested_epochs": sorted(requested - committed),
                 "earliest_committed_epoch": earliest, "latest_committed_epoch": latest}

    def write_progress(index, report):
        keys = ("base", "trained", "reward_delta", "success_delta_pp", "paired_deltas", "counterfactual",
                "choice_change_rate")
        row = {"epoch": checkpoints[index][0], **{key: report[key] for key in keys}}
        if "parent_comparison" in report:
            row["parent_comparison"] = report["parent_comparison"]
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), file=progress, flush=True)

    reports = _evaluate_adapters(replay, model, [path for _, path in checkpoints], device,
                                 on_scored=write_progress if progress is not None else None,
                                 compare_adapter=compare_adapter, **settings)
    summaries = []
    for (epoch, path), report in zip(checkpoints, reports, strict=True):
        summary = {"epoch": epoch, "checkpoint": str(path),
                   "meets_base_success_rate": report["trained"]["success_rate"] >= report["base"]["success_rate"],
                   **{key: report[key] for key in ("policy_id", "trained", "reward_delta",
                                                  "success_delta_pp", "paired_deltas")}}
        if "parent_comparison" in report:
            parent = report["parent_comparison"]
            summary.update(parent_comparison=parent,
                           meets_parent_reward_and_success=parent["reward_delta"] >= 0 and parent["success_delta_pp"] >= 0)
        summaries.append(summary)
    eligibility = "meets_parent_reward_and_success" if compare_adapter is not None else "meets_base_success_rate"
    eligible = [index for index, summary in enumerate(summaries) if summary[eligibility]]
    selected_index = max(eligible or range(len(summaries)), key=lambda index: (
        summaries[index]["trained"]["reward_mean"], summaries[index]["trained"]["success_rate"],
        -summaries[index]["epoch"],
    ))
    selected = summaries[selected_index]
    selection = {"basis": "development_replay",
                 "rule": "maximize reward among checkpoints at or above base success rate",
                 "tie_breakers": ["higher success rate", "earlier epoch"],
                 "eligible_epochs": [summaries[index]["epoch"] for index in eligible],
                 "fallback_used": not bool(eligible), **sweep}
    if compare_adapter is not None:
        selection.update(comparison_basis="parent_adapter", accepted=bool(eligible),
                         rule="maximize reward among checkpoints at or above parent reward and success rate")
    return {**reports[selected_index], "selected_epoch": selected["epoch"],
            "selected_checkpoint": selected["checkpoint"], "checkpoints": summaries,
            "checkpoint_selection": selection}


def _open_progress(path, *, automatic):
    if path is None:
        return nullcontext()
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError:
        if not automatic:
            raise
        stream = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                    prefix=f"{path.stem}-retry-", suffix=path.suffix, delete=False)
    print(f"Development progress: {stream.name}", file=sys.stderr, flush=True)
    return stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--train-replay", type=Path, help="Check disjoint scenarios against the adapter's training replay")
    parser.add_argument("--model", default=MODEL_PATH)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--adapter", type=Path)
    source.add_argument("--checkpoints-dir", type=Path, help="Score committed epoch checkpoints in a training directory")
    parser.add_argument("--compare-adapter", type=Path, help="Score a parent adapter on the same checkpoint-sweep states")
    parser.add_argument("--expected-epochs", type=int, help="Require checkpoints for all epochs through this number")
    parser.add_argument("--eval-epoch", action="append", type=int,
                        help="Score this epoch plus the first and latest committed checkpoints (repeatable)")
    parser.add_argument("--progress-jsonl", type=Path,
                        help="Journal scored checkpoints (default dev-progress.jsonl alongside dev-score.json)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261118)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.bootstrap_samples < 100 or args.max_prompt_tokens < 1:
        parser.error("Invalid bootstrap sample count or prompt limit")
    if args.expected_epochs is not None and (args.checkpoints_dir is None or args.expected_epochs < 1):
        parser.error("--expected-epochs must be positive and requires --checkpoints-dir")
    if args.eval_epoch and (args.checkpoints_dir is None or any(epoch < 1 for epoch in args.eval_epoch)):
        parser.error("--eval-epoch must be positive and requires --checkpoints-dir")
    if args.progress_jsonl is not None and args.checkpoints_dir is None:
        parser.error("--progress-jsonl requires --checkpoints-dir")
    if args.compare_adapter is not None and args.checkpoints_dir is None:
        parser.error("--compare-adapter requires --checkpoints-dir")
    try:
        if args.output and (args.output.exists() or args.output.is_symlink()):
            raise FileExistsError(args.output)
        settings = {"train_replay": args.train_replay, "samples": args.bootstrap_samples,
                    "seed": args.bootstrap_seed, "max_prompt_tokens": args.max_prompt_tokens}
        if args.checkpoints_dir is not None:
            progress_path = args.progress_jsonl
            if progress_path is None and args.output and args.output.name == "dev-score.json":
                progress_path = args.output.with_name("dev-progress.jsonl")
            if progress_path is not None and args.output and progress_path.resolve() == args.output.resolve():
                raise ValueError("Progress journal and final score need separate files")
            with _open_progress(progress_path, automatic=args.progress_jsonl is None) as progress:
                result = evaluate_checkpoints(args.replay, args.model, args.checkpoints_dir, args.device,
                                              expected_epochs=args.expected_epochs, eval_epochs=args.eval_epoch,
                                              progress=progress, compare_adapter=args.compare_adapter, **settings)
        else:
            result = evaluate(args.replay, args.model, args.adapter, args.device, **settings)
        raw = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(raw)
        else:
            print(raw, end="")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
