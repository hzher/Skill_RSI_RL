import gzip
import json
import tempfile
import unittest
from pathlib import Path

from skill_rsi_rl.compare import clustered_delta, compare


def _arm(root, arm, reward, success, mismatched=False, wrong_policy=False, decisions=None):
    directory = root / arm
    checkpoints = directory / "checkpoints" / "poker"
    checkpoints.mkdir(parents=True)
    policy = {"checkpoint": arm} if arm != "control" else None
    (directory / "manifest.json").write_text(json.dumps({
        "comparison_fingerprint": "shared", "fingerprint": arm,
        "system": {"model": "deepseek-v4-pro"}, "memory_policy": policy, "evaluation_mode": "stateful_only",
        "tasks": {"poker": {"rollout_count": 2, "instances_per_rollout": 1}},
    }))
    for index in range(2):
        instance = "another" if (mismatched and index == 1) else "same"
        choices = decisions if decisions is not None else [{"status": "applied", "op": "recall", "injected": True}]
        checkpoint = "unrelated" if wrong_policy and index else arm
        receipts = [{"response": {"metadata": {"memory_policy": {**choice, "checkpoint": checkpoint,
                     "elapsed_seconds": 0.25}}}} for choice in choices] if policy else []
        with gzip.open(checkpoints / f"run-{index}.json.gz", "wt") as stream:
            json.dump({
                "status": "completed", "capture_identity": {
                    "task": "poker", "index": index, "phase": "run", "fingerprint": arm,
                    "expected_instance_count": 1, "mode": "permute", "schedule": "default", "seed": index,
                    "run_id": f"{arm}-{index}", "attempt_id": f"{arm}-{index}-attempt",
                },
                "instance_outcomes": [{"instance_index": 0, "instance_id": instance, "reward": reward,
                                       "success": success, "latency_seconds": 2.0, "metadata": {"stage_index": 0}}],
                "interactions": receipts,
            }, stream)
    return directory


class CompareTests(unittest.TestCase):
    def test_aggregates_official_success_independently_of_positive_reward(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arms = [_arm(root, "control", 0.2, False), _arm(root, "base", 0.4, False),
                    _arm(root, "trained", 0.9, True)]
            result = compare(*arms, minimum_pairs=2, samples=100)["tasks"]["poker"]
            self.assertEqual(result["arms"]["base"]["successes"], 0)
            self.assertEqual(result["arms"]["trained"]["policy_receipts"]["injected"], 2)
            self.assertEqual(result["contrasts"]["trained_vs_base"]["success_pp"]["mean"], 100)
            self.assertTrue(result["official_metric_improvement_supported"])
            self.assertTrue(result["improvement_supported"])
            self.assertTrue(result["part2_gain_supported"])
            self.assertEqual(result["part2_improved_metrics"], ["reward", "success_pp"])
            self.assertEqual(result["strata"]["stage 0"]["matched_instances_per_arm"], 2)
            self.assertEqual(result["strata"]["stage 0"]["contrasts"]["trained_vs_base"]["success_pp"]["mean"], 100)

    def test_native_improvement_preserved_without_treated_delivery(self):
        noop = {"status": "applied", "op": "noop"}
        fallback = {"status": "fallback", "op": "noop"}
        for base, trained in ((None, [noop, fallback]), ([], None), ([fallback], None)):
            with self.subTest(base=base, trained=trained), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                arms = [_arm(root, "control", 0.2, False), _arm(root, "base", 0.4, False, decisions=base),
                        _arm(root, "trained", 0.9, True, decisions=trained)]
                result = compare(*arms, minimum_pairs=2, samples=100)["tasks"]["poker"]
                undelivered = "trained" if trained is not None else "base"
                self.assertEqual(result["arms"][undelivered]["policy_caveat"], "no_verified_policy_delivery")
                self.assertEqual(result["matched_pairs"], 2)
                self.assertEqual(result["rejected"], [])
                self.assertTrue(result["official_metric_improvement_supported"])
                self.assertFalse(result["improvement_supported"])
                self.assertFalse(result["part2_gain_supported"])
                self.assertEqual(result["part2_improved_metrics"], [])

    def test_part2_accepts_reward_gain_without_success_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arms = [_arm(root, "control", 0.2, False), _arm(root, "base", 0.4, False),
                    _arm(root, "trained", 0.9, False)]
            report = compare(*arms, minimum_pairs=2, samples=100)
            result = report["tasks"]["poker"]
            self.assertFalse(result["improvement_supported"])
            self.assertFalse(result["official_metric_improvement_supported"])
            self.assertTrue(result["part2_gain_supported"])
            self.assertEqual(result["part2_improved_metrics"], ["reward"])
            for arm in ("control", "base"):
                contrast = result["contrasts"][f"trained_vs_{arm}"]
                self.assertGreater(contrast["reward"]["ci975"][0], 0)
                self.assertEqual(contrast["success_pp"]["ci975"], [0, 0])
            self.assertEqual(report["part2_acceptance"]["scope"], "per_task")
            self.assertEqual(report["part2_acceptance"]["endpoint_confidence_level"], 0.975)

    def test_part2_requires_the_same_endpoint_to_clear_both_comparators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arms = [_arm(root, "control", 0.8, False), _arm(root, "base", 0.2, True),
                    _arm(root, "trained", 0.6, True)]
            result = compare(*arms, minimum_pairs=2, samples=100)["tasks"]["poker"]
            self.assertGreater(result["contrasts"]["trained_vs_control"]["success_pp"]["ci975"][0], 0)
            self.assertGreater(result["contrasts"]["trained_vs_base"]["reward"]["ci975"][0], 0)
            self.assertFalse(result["part2_gain_supported"])
            self.assertEqual(result["part2_improved_metrics"], [])

    def test_adjusted_interval_uses_same_draws_as_default(self):
        clusters = [[-1], [0], [2], [8], [16]]
        default = clustered_delta(clusters, 1000, 7)
        adjusted = clustered_delta(clusters, 1000, 7, adjusted=True)
        self.assertEqual(default, {key: adjusted[key] for key in ("mean", "ci95")})
        self.assertLess(adjusted["ci975"][0], adjusted["ci95"][0])
        self.assertGreater(adjusted["ci975"][1], adjusted["ci95"][1])

    def test_trained_write_requires_an_actual_memory_change(self):
        noop = {"status": "applied", "op": "noop"}
        for changed in (False, True):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                write = {"status": "applied", "op": "write", "memory_before_sha256": "a" * 64,
                         "memory_after_sha256": ("b" if changed else "a") * 64}
                arms = [_arm(root, "control", 0.2, False), _arm(root, "base", 0.4, False, decisions=[noop, noop]),
                        _arm(root, "trained", 0.9, True, decisions=[noop, write])]
                result = compare(*arms, minimum_pairs=2, samples=100)["tasks"]["poker"]
                self.assertEqual(result["arms"]["base"]["rollouts_with_verified_policy_delivery"], 2)
                self.assertEqual(result["arms"]["trained"]["policy_receipts"]["verified_applied_non_noop"], 2)
                self.assertEqual(result["arms"]["trained"]["policy_receipts"]["recorded_interventions"], 2 * changed)
                self.assertTrue(result["official_metric_improvement_supported"])
                self.assertEqual(result["improvement_supported"], changed)
                self.assertEqual(result["part2_gain_supported"], changed)

    def test_different_instance_identity_rejects_entire_rollout_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arms = [_arm(root, "control", 0.2, False), _arm(root, "base", 0.4, False),
                    _arm(root, "trained", 0.9, True, mismatched=True)]
            result = compare(*arms, minimum_pairs=2, samples=100)["tasks"]["poker"]
            self.assertEqual(result["matched_pairs"], 1)
            self.assertIn("sequence", result["rejected"][0]["reasons"]["pair"])
            self.assertIsNone(result["contrasts"]["trained_vs_base"]["reward"]["ci95"])
            self.assertIsNone(result["contrasts"]["trained_vs_base"]["reward"]["ci975"])
            self.assertFalse(result["improvement_supported"])
            self.assertFalse(result["part2_gain_supported"])

    def test_wrong_policy_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arms = [_arm(root, "control", 0.2, False), _arm(root, "base", 0.4, False),
                    _arm(root, "trained", 0.9, True, wrong_policy=True)]
            result = compare(*arms, minimum_pairs=2, samples=100)["tasks"]["poker"]
            self.assertEqual(result["matched_pairs"], 1)
            self.assertIn("frozen arm", result["rejected"][0]["reasons"]["trained"])
