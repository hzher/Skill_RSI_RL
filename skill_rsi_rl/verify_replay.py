"""Audit frozen replay quality, provenance, and split separation before training."""

from argparse import ArgumentParser
from collections import Counter
import json
import math
from pathlib import Path

from .contracts import build_messages, digest, propose_options
from .replay_io import open_hashed_replay


def _validate_support(group):
    messages = group["messages"]
    observation = json.loads(messages[1]["content"])
    if not isinstance(observation, dict) or set(observation) != {"query", "feedback", "memory", "options"}:
        raise ValueError("Student observation must carry the shared decision state")
    query, feedback, memory = (observation[key] for key in ("query", "feedback", "memory"))
    options = group["options"]
    if options != propose_options(query, feedback, memory) or messages != build_messages(query, feedback, memory, options):
        raise ValueError("Student messages or candidates differ from the shared policy support")
    observed_sha = digest(messages)
    if group.get("observation_sha256", observed_sha) != observed_sha:
        raise ValueError("Recorded observation SHA-256 differs from the student messages")
    return observed_sha


def _score_pair(row):
    reward, success = row.get("reward"), row.get("native_success")
    if type(reward) not in (float, int) or not math.isfinite(reward) or type(success) is not bool:
        raise ValueError("Candidate evaluation needs a finite reward and boolean native_success")
    return reward, success


def _score_coverage(group, split):
    scores = group.get("option_scores")
    if scores is None and split == "dev":
        raise ValueError("Developers need downstream scores for all options in dev replay")
    if scores is not None:
        if not isinstance(scores, list) or len(scores) != len(group["options"]) or any(
                not isinstance(row, dict) for row in scores):
            raise ValueError("Candidate score table must match the full option support")
        scores = [_score_pair(row) for row in scores]
    sampled = {}
    for row in group["samples"]:
        index = row["option_index"]
        pair = _score_pair(row)
        if scores is not None and scores[index] != pair:
            raise ValueError("Sample reward differs from its frozen candidate evaluation")
        if sampled.setdefault(index, pair) != pair:
            raise ValueError("Repeated draws of one option disagree on its frozen score")
    return sampled


def _inspect_stream(manifest, groups, split):
    if manifest["split"] != split:
        raise ValueError(f"Expected {split} replay, found {manifest['split']}")
    if manifest.get("shard", {"rank": 0, "world_size": 1}) != {"rank": 0, "world_size": 1}:
        raise ValueError("Replay shards must be merged before split verification")
    counts = Counter({field: 0 for field in (
        "states", "samples", "eligible_grpo_groups", "zero_reward_variance", "fewer_than_two_samples",
        "multiple_sampled_options", "full_support_sampled", "successful_samples",
    )})
    candidate_ops = Counter()
    sampled_ops = Counter()
    state_ids = set()
    scenario_ids = set()
    observation_hashes = set()
    prompt_lengths = []
    effective_support = []
    for index, group in enumerate(groups):
        try:
            state_index = group.get("state_index", index)
            if type(state_index) is not int or state_index != index:
                raise ValueError("Merged state indices must be contiguous")
            if len(group["samples"]) != manifest.get("sampled_outcomes_per_state", len(group["samples"])):
                raise ValueError("Samples per state differ from the collection manifest")
            observation_hashes.add(_validate_support(group))
            scores = _score_coverage(group, split)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"{split} state index {index}: {exc}") from exc
        state_ids.add(group["state_id"])
        scenario_ids.add(group["scenario_id"])
        prompt_lengths.append(len(group["prompt_token_ids"]))
        # Entropy describes concentration in the recorded categorical behavior policy.
        effective_support.append(math.exp(-sum(lp * math.exp(lp) for lp in group["behavior_logprobs"])))
        options = group["options"]
        samples = group["samples"]
        counts["states"] += 1
        counts["samples"] += len(samples)
        counts["successful_samples"] += sum(row["native_success"] for row in samples)
        counts["multiple_sampled_options"] += len(scores) > 1
        counts["full_support_sampled"] += len(scores) == len(options)
        if len(samples) < 2:
            counts["fewer_than_two_samples"] += 1
        elif len({reward for reward, _ in scores.values()}) < 2:
            counts["zero_reward_variance"] += 1
        else:
            counts["eligible_grpo_groups"] += 1
        candidate_ops.update(row["action"]["op"] for row in options)
        sampled_ops.update(options[row["option_index"]]["action"]["op"] for row in samples)

    if manifest.get("state_count", counts["states"]) != counts["states"]:
        raise ValueError("Replay group count differs from the collection manifest")
    if split == "train" and not manifest.get("fixture_only") and not counts["eligible_grpo_groups"]:
        raise ValueError("Training replay has no group with at least two varying rewards")
    report = {
        "split": split, "dataset_id": manifest["dataset_id"],
        "source_kind": manifest["source_kind"], "training_permission": manifest["training_permission"],
        "score_protocol": manifest.get("score_protocol"), "source_sha256": manifest.get("source_sha256"),
        "behavior_checkpoint_id": manifest.get("behavior_checkpoint_id"),
        "behavior_tokenizer_sha256": manifest.get("behavior_tokenizer_sha256"),
        "behavior_sampler": manifest.get("behavior_sampler"),
        "behavior_uniform_mix": manifest.get("behavior_uniform_mix", 0.0),
        "inference_batch_size": manifest.get("inference_batch_size", 1),
        "fixture_only": manifest.get("fixture_only", False), "scenarios": len(scenario_ids),
        "unique_observations": len(observation_hashes), "counts": dict(counts),
        "candidates_by_operation": dict(candidate_ops), "samples_by_operation": dict(sampled_ops),
        "prompt_tokens": {"min": min(prompt_lengths), "max": max(prompt_lengths), "total": sum(prompt_lengths)},
        "mean_effective_behavior_support": math.fsum(effective_support) / len(effective_support),
    }
    return report, state_ids, scenario_ids, observation_hashes


def inspect_replay(path, split):
    with open_hashed_replay(path) as (manifest, groups, replay_digest):
        report, states, scenarios, observations = _inspect_stream(manifest, groups, split)
        report["file_sha256"] = replay_digest()
    return report, states, scenarios, observations


def require_disjoint(train_keys, dev_keys):
    """Require disjoint state IDs, scenario IDs, and canonical observation hashes."""
    names = ("state_ids", "scenario_ids", "identical_observations")
    overlap = {name: len(train & dev) for name, train, dev in zip(names, train_keys, dev_keys, strict=True)}
    if any(overlap.values()):
        raise ValueError(f"Train and dev overlap on {overlap['state_ids']} states and {overlap['scenario_ids']} "
                         f"scenarios, with {overlap['identical_observations']} identical observations")
    return overlap


def verify_splits(train_path, dev_path):
    train, *train_keys = inspect_replay(train_path, "train")
    dev, *dev_keys = inspect_replay(dev_path, "dev")
    overlap = require_disjoint(train_keys, dev_keys)
    return {"schema_version": 1, "train": train, "dev": dev, "split_overlap": overlap}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = verify_splits(args.train, args.dev)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
