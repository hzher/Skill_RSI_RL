"""Compare independently executed, matched DSH stateful benchmark arms."""

import argparse
import gzip
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path


ARM_NAMES = ("control", "base", "trained")
POLICY_OPS = ("noop", "recall", "compact", "write")
PART2_ENDPOINTS = ("reward", "success_pp")


def _read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _policy_name(manifest):
    config = manifest.get("memory_policy")
    return config.get("checkpoint") if isinstance(config, dict) else None


def _recorded_intervention(receipt):
    if receipt["op"] in ("recall", "compact"):
        return receipt.get("injected") is True
    if receipt["op"] == "write":
        before, after = receipt.get("memory_before_sha256"), receipt.get("memory_after_sha256")
        return before != after and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                                       for value in (before, after))
    return False


def _outcome_stratum(task, row):
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if type(metadata.get("stage_index")) is int:
        return f"stage {metadata['stage_index']}"
    if task == "exploitable_poker":
        match = re.search(r":stage(\d+)_seed\d+_hand\d+$", row["instance_id"])
        if match:
            return f"stage {match.group(1)}"
    variant = metadata.get("variant_id")
    if isinstance(variant, str) and variant:
        return variant
    if task == "blind_spectrum_monitoring":
        fields = row["instance_id"].split(":")
        if len(fields) > 3 and fields[0] == task and fields[2] in {"corpus", "seed"}:
            return fields[1]
    return None


def _accepted_rollout(directory, task, index, expected, manifest):
    path = directory / "checkpoints" / task / f"run-{index}.json.gz"
    if not path.is_file():
        raise ValueError("rollout checkpoint missing")
    payload = path.read_bytes()
    trace = json.loads(gzip.decompress(payload))
    identity = trace["capture_identity"]
    if (identity["task"], identity["index"], identity["phase"]) != (task, index, "run"):
        raise ValueError("rollout identity mismatch")
    if identity.get("fingerprint") != manifest.get("fingerprint"):
        raise ValueError("rollout fingerprint mismatch")
    if identity.get("execution_fingerprint", identity["fingerprint"]) not in {
        manifest["fingerprint"], *manifest.get("execution_revisions", {}),
    }:
        raise ValueError("rollout executor revision is outside the arm manifest")
    plan = manifest["tasks"][task]
    if any(key in plan and identity.get(field) != plan[key] for key, field in (
        ("schedule", "schedule"), ("run_mode", "mode"), ("seed", "seed"), ("stage_seeds", "stage_seeds"),
    )):
        raise ValueError("rollout state differs from the frozen task protocol")
    if trace.get("status") != "completed":
        raise ValueError("rollout incomplete")
    run_id, attempt_id = identity["run_id"], identity["attempt_id"]
    if (not isinstance(run_id, str) or not run_id or not isinstance(attempt_id, str) or not attempt_id
            or Path(attempt_id).name != attempt_id or attempt_id in {".", ".."}):
        raise ValueError("rollout has an invalid run or attempt identity")
    trace_sha = hashlib.sha256(payload).hexdigest()
    attempt_ref = path.parent / "attempts" / f"run-{index}" / attempt_id / "manifest.json"
    sealed = attempt_ref.is_file()
    if sealed:
        attempt = _read(attempt_ref)
        if (attempt.get("status") != "completed" or attempt.get("trace_sha256") != trace_sha
                or attempt.get("run_id") != run_id or attempt.get("attempt_id") != attempt_id):
            raise ValueError("published rollout differs from its sealed attempt")
    outcomes = trace["instance_outcomes"]
    if len(outcomes) != expected or identity["expected_instance_count"] != expected:
        raise ValueError("official instance count mismatch")
    sequence = []
    reward = []
    success = []
    strata = []
    official_latency = []
    for row in outcomes:
        if type(row.get("success")) is not bool or not _finite(row.get("reward")):
            raise ValueError("non-scorable official outcome")
        if (type(row.get("instance_index")) is not int or row["instance_index"] < 0
                or not isinstance(row.get("instance_id"), str) or not row["instance_id"]):
            raise ValueError("invalid official instance identity")
        sequence.append((row["instance_index"], row["instance_id"]))
        reward.append(float(row["reward"]))
        success.append(row["success"])
        strata.append(_outcome_stratum(task, row))
        if _finite(row.get("latency_seconds")):
            official_latency.append(float(row["latency_seconds"]))
    if len({name for _, name in sequence}) != expected:
        raise ValueError("duplicate official instance identity")
    policy_receipts = Counter()
    policy_latency = []
    interactions = trace.get("interactions", [])
    expected_policy = _policy_name(manifest)
    for query_index, interaction in enumerate(interactions):
        receipt = ((interaction.get("response") or {}).get("metadata") or {}).get("memory_policy")
        if receipt:
            if not isinstance(receipt, dict) or not expected_policy or (receipt.get("checkpoint") not in (None, expected_policy)
                                       or (manifest.get("evaluation_mode") == "stateful_only"
                                           and receipt.get("checkpoint") != expected_policy)):
                raise ValueError("policy receipt differs from the frozen arm")
            status, op = receipt.get("status"), receipt.get("op")
            policy_receipts["total"] += 1
            policy_receipts[status if status in ("applied", "fallback") else "unknown"] += 1
            if receipt.get("injected"):
                policy_receipts["injected"] += 1
            if op:
                policy_receipts[f"op_{op}"] += 1
            if receipt.get("checkpoint") == expected_policy and status == "applied" and op in POLICY_OPS:
                policy_receipts["verified_applied"] += 1
                policy_receipts["verified_applied_after_first_query"] += query_index > 0
                policy_receipts["verified_applied_non_noop"] += op != "noop"
                policy_receipts["recorded_interventions"] += _recorded_intervention(receipt)
            if _finite(receipt.get("elapsed_seconds")):
                policy_latency.append(float(receipt["elapsed_seconds"]))
    return {"sequence": sequence, "state": (identity["mode"], identity["schedule"],
                                            identity["seed"], identity.get("stage_seeds")),
            "run_id": run_id, "attempt_id": attempt_id, "trace_sha256": trace_sha, "sealed_attempt": sealed,
            "reward": reward, "success": success, "strata": strata, "receipts": policy_receipts,
            "query_count": len(interactions),
            "official_latency": official_latency, "policy_latency": policy_latency}


def _finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def _score_summary(rows):
    rewards = [reward for row in rows for reward in row["reward"]]
    successes = sum(sum(row["success"]) for row in rows)
    return {"scorable_instances": len(rewards), "successes": successes,
            "success_rate": successes / len(rewards) if rewards else None,
            "reward_mean": statistics.fmean(rewards) if rewards else None}


def _summary(rows, *, treated=False):
    receipts = Counter()
    for row in rows:
        receipts.update(row["receipts"])
    official_latency = [seconds for row in rows for seconds in row["official_latency"]]
    policy_latency = [seconds for row in rows for seconds in row["policy_latency"]]
    result = {**_score_summary(rows), "policy_receipts": dict(receipts),
              "recorded_queries": sum(row["query_count"] for row in rows),
              "official_instance_latency_seconds": statistics.fmean(official_latency) if official_latency else None,
              "policy_query_latency_seconds": statistics.fmean(policy_latency) if policy_latency else None}
    if treated:
        for name in ("verified_applied", "verified_applied_after_first_query", "verified_applied_non_noop",
                     "recorded_interventions"):
            result["policy_receipts"].setdefault(name, 0)
        delivered = sum(bool(row["receipts"]["verified_applied_after_first_query"]
                             or row["receipts"]["recorded_interventions"]) for row in rows)
        result["rollouts_with_verified_policy_delivery"] = delivered
        caveat = None
        if delivered == 0:
            caveat = "no_verified_policy_delivery"
        elif delivered < len(rows):
            caveat = "partial_verified_policy_delivery"
        elif not receipts["recorded_interventions"]:
            caveat = "no_recorded_intervention"
        result["policy_caveat"] = caveat
    return result


def clustered_delta(clusters, samples, seed, *, adjusted=False):
    """Estimate a paired mean and percentile interval, resampling complete clusters."""
    bounds = {"ci95": (0.025, 0.975)}
    if adjusted:
        bounds["ci975"] = (0.0125, 0.9875)
    result = {"mean": None, **dict.fromkeys(bounds)}
    if not clusters:
        return result
    sums_and_counts = [(math.fsum(group), len(group)) for group in clusters]
    result["mean"] = math.fsum(total for total, _ in sums_and_counts) / sum(count for _, count in sums_and_counts)
    if len(clusters) < 2:
        return result
    rng = random.Random(seed)
    reps = []
    for _ in range(samples):
        draw = rng.choices(sums_and_counts, k=len(clusters))
        reps.append(math.fsum(total for total, _ in draw) / sum(count for _, count in draw))
    reps.sort()
    result.update({name: [reps[int(q * (samples - 1))] for q in quantiles] for name, quantiles in bounds.items()})
    return result


def _contrast(rows, baseline, candidate, samples, seed, *, adjusted=False):
    reward = [statistics.fmean(pair[candidate]["reward"]) - statistics.fmean(pair[baseline]["reward"]) for pair in rows]
    success = [100 * (statistics.fmean(pair[candidate]["success"]) - statistics.fmean(pair[baseline]["success"]))
               for pair in rows]
    return {"reward": clustered_delta([[value] for value in reward], samples, seed, adjusted=adjusted),
            "success_pp": clustered_delta([[value] for value in success], samples, seed, adjusted=adjusted)}


def _contrasts(rows, samples, seed, *, adjusted=False):
    return {f"{candidate}_vs_{baseline}": _contrast(rows, baseline, candidate, samples, seed,
                                                   adjusted=adjusted and candidate == "trained")
            for baseline, candidate in (("control", "base"), ("control", "trained"), ("base", "trained"))}


def _strata(rows, samples, seed):
    names = sorted({name for pair in rows for name in pair["control"]["strata"] if name is not None})
    result = {}
    for name in names:
        stratum = []
        for pair in rows:
            keep = [index for index, label in enumerate(pair["control"]["strata"]) if label == name]
            if keep:
                stratum.append({arm: {key: [row[key][index] for index in keep] for key in ("reward", "success")}
                                for arm, row in pair.items()})
        result[name] = {"matched_pairs": len(stratum),
                        "matched_instances_per_arm": sum(len(pair["control"]["reward"]) for pair in stratum),
                        "arms": {arm: _score_summary([pair[arm] for pair in stratum]) for arm in ARM_NAMES},
                        "contrasts": _contrasts(stratum, samples, seed)}
    return result


def _positive_interval(interval):
    return interval is not None and interval[0] > 0


def _positive(contrast):
    return all(_positive_interval(row["ci95"]) for row in contrast.values())


def compare(control, base, trained, *, samples=5000, seed=20261118, minimum_pairs=5):
    directories = {name: Path(path).resolve() for name, path in zip(ARM_NAMES, (control, base, trained))}
    if len(set(directories.values())) != len(ARM_NAMES):
        raise ValueError("arm directories must be independent")
    if samples < 100 or minimum_pairs < 2:
        raise ValueError("insufficient bootstrap samples or minimum pairs")
    manifests = {name: _read(path / "manifest.json") for name, path in directories.items()}
    reference = manifests["control"]
    fingerprint = reference.get("comparison_fingerprint")
    if not fingerprint or any(row.get("comparison_fingerprint") != fingerprint for row in manifests.values()):
        raise ValueError("manifests differ in their frozen comparison configuration")
    for key in ("system", "skill_loading", "evaluation_mode"):
        if any(row.get(key) != reference.get(key) for row in manifests.values()):
            raise ValueError(f"manifests have different {key} settings")
    if any(_policy_name(manifests[arm]) is None for arm in ("base", "trained")) or _policy_name(reference) is not None:
        raise ValueError("expected native control plus base and trained policy arms")
    if _policy_name(manifests["base"]) == _policy_name(manifests["trained"]):
        raise ValueError("base and trained policy must be distinct checkpoints")
    if any(set(row["tasks"]) != set(reference["tasks"]) for row in manifests.values()):
        raise ValueError("manifests have different task sets")
    report = {"comparison_fingerprint": fingerprint, "executor": reference.get("system", {}).get("model"),
              "policy_checkpoints": {name: _policy_name(row) for name, row in manifests.items()},
              "minimum_pairs": minimum_pairs, "bootstrap_samples": samples, "bootstrap_seed": seed,
              "confidence_unit": "paired_stateful_rollout",
              "part2_acceptance": {"scope": "per_task", "multiplicity": "bonferroni",
                                   "family_alpha": 0.05, "endpoint_confidence_level": 0.975,
                                   "endpoints": list(PART2_ENDPOINTS), "comparators": ["control", "base"]},
              "tasks": {}}
    used_run_ids = set()
    used_attempt_ids = set()
    for task, meta in reference["tasks"].items():
        rollout_count = meta["rollout_count"]
        expected = meta["instances_per_rollout"]
        if any((row["tasks"].get(task, {}).get("rollout_count"), row["tasks"].get(task, {}).get("instances_per_rollout"))
               != (rollout_count, expected) for row in manifests.values()):
            raise ValueError(f"planned rollout counts differ for {task}")
        for key in ("schedule", "run_mode", "seed", "stage_seeds", "max_workers"):
            if any(row["tasks"][task].get(key) != meta.get(key) for row in manifests.values()):
                raise ValueError(f"planned {key} differs for {task}")
        matched = []
        paired_rows = []
        rejected = []
        observed = Counter()
        for index in range(rollout_count):
            pair = {}
            reasons = {}
            for arm, path in directories.items():
                try:
                    row = _accepted_rollout(path, task, index, expected, manifests[arm])
                    if row["run_id"] in used_run_ids or row["attempt_id"] in used_attempt_ids:
                        raise ValueError("reused rollout or attempt identity")
                    used_run_ids.add(row["run_id"])
                    used_attempt_ids.add(row["attempt_id"])
                    observed[arm] += len(row["reward"])
                    pair[arm] = row
                except (ValueError, IOError, EOFError, KeyError, TypeError) as exc:
                    reasons[arm] = str(exc)
            if not reasons:
                native = pair["control"]
                if any(row["sequence"] != native["sequence"] or row["state"] != native["state"] for row in pair.values()):
                    reasons["pair"] = "official instance sequence or seed mismatch"
                elif any(row["strata"] != native["strata"] for row in pair.values()):
                    reasons["pair"] = "official stage or variant mismatch"
            if reasons:
                rejected.append({"index": index, "reasons": reasons})
            else:
                matched.append(pair)
                paired_rows.append({"index": index, "arms": {
                    arm: {"run_id": row["run_id"], "attempt_id": row["attempt_id"],
                          "trace_sha256": row["trace_sha256"], "sealed_attempt": row["sealed_attempt"],
                          "reward_mean": statistics.fmean(row["reward"]),
                          "success_rate": statistics.fmean(row["success"])} for arm, row in pair.items()
                }})
        contrasts = _contrasts(matched, samples, seed, adjusted=True)
        complete = len(matched) == rollout_count and len(matched) >= minimum_pairs
        official_improvement = (complete and _positive(contrasts["trained_vs_control"])
                                and _positive(contrasts["trained_vs_base"]))
        arms = {arm: _summary([pair[arm] for pair in matched], treated=arm != "control") for arm in ARM_NAMES}
        intervention_supported = (all(arms[arm]["rollouts_with_verified_policy_delivery"] > 0 for arm in ("base", "trained"))
                                  and arms["trained"]["policy_receipts"]["recorded_interventions"] > 0)
        # Either endpoint can pass, but it must beat both comparators at its adjusted level.
        part2_metrics = [metric for metric in PART2_ENDPOINTS
                         if complete and intervention_supported and all(
                             _positive_interval(contrasts[f"trained_vs_{arm}"][metric]["ci975"])
                             for arm in ("control", "base"))]
        report["tasks"][task] = {"planned_pairs": rollout_count, "matched_pairs": len(matched),
                                  "planned_instances_per_arm": expected * rollout_count,
                                  "observed_scorable_instances": {arm: observed[arm] for arm in ARM_NAMES},
                                  "arms": arms,
                                  "contrasts": contrasts, "matched_rollouts": paired_rows, "rejected": rejected,
                                  "strata": _strata(matched, samples, seed),
                                  "official_metric_improvement_supported": official_improvement,
                                  "improvement_supported": official_improvement and intervention_supported,
                                  "part2_gain_supported": bool(part2_metrics),
                                  "part2_improved_metrics": part2_metrics}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ARM_NAMES:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-pairs", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261118)
    args = parser.parse_args()
    try:
        report = compare(args.control, args.base, args.trained, samples=args.bootstrap_samples,
                         seed=args.bootstrap_seed, minimum_pairs=args.minimum_pairs)
        raw = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(raw)
        else:
            print(raw, end="")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
