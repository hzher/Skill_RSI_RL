from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from skill_rsi_rl import verify_replay
from skill_rsi_rl.collect import collect
from skill_rsi_rl.contracts import load_replay, write_replay
from skill_rsi_rl.verify_replay import inspect_replay, verify_splits


def fixture(path, split, seed=17):
    collect(SimpleNamespace(
        fixture=True, replay_spec=None, split=split, seed=seed, states=24,
        samples_per_state=8, max_prompt_tokens=4096, temperature=1.0, checkpoint_id=None,
        inference_batch_size=5, rank=0, world_size=1, output=path,
    ))


class VerifyReplayTests(unittest.TestCase):
    def test_fingerprint_covers_the_same_file_read_despite_rename(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.jsonl"
            replacement = root / "replacement.jsonl"
            fixture(source, "train", seed=17)
            fixture(replacement, "train", seed=23)
            original_bytes = source.read_bytes()
            replacement_bytes = replacement.read_bytes()
            self.assertNotEqual(original_bytes, replacement_bytes)
            original_report, original_states, *_ = inspect_replay(source, "train")
            inspect_stream = verify_replay._inspect_stream

            def inspect_after_rename(manifest, groups, split):
                replacement.replace(source)
                return inspect_stream(manifest, groups, split)

            with patch.object(verify_replay, "_inspect_stream", side_effect=inspect_after_rename):
                report, states, *_ = inspect_replay(source, "train")
            self.assertEqual(source.read_bytes(), replacement_bytes)
            self.assertEqual(report["file_sha256"], sha256(original_bytes).hexdigest())
            self.assertEqual(report, original_report)
            self.assertEqual(states, original_states)

    def test_split_summary_and_scenario_overlap(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train, dev = root / "train.jsonl", root / "dev.jsonl"
            fixture(train, "train")
            fixture(dev, "dev")
            report = verify_splits(train, dev)
            self.assertEqual(report["split_overlap"], {"state_ids": 0, "scenario_ids": 0, "identical_observations": 0})
            self.assertEqual(report["train"]["counts"]["samples"], 24 * 8)
            self.assertGreater(report["train"]["counts"]["eligible_grpo_groups"], 0)
            self.assertEqual(sum(report["train"]["samples_by_operation"].values()), 24 * 8)
            self.assertEqual(len(report["dev"]["file_sha256"]), 64)

            _, train_rows = load_replay(train)
            manifest, dev_rows = load_replay(dev)
            original_dev = dev_rows[0].copy()
            dev_rows[0]["scenario_id"] = train_rows[0]["scenario_id"]
            leak = root / "leak.jsonl"
            write_replay(leak, manifest, dev_rows)
            with self.assertRaisesRegex(ValueError, "overlap on 0 states and 1 scenarios"):
                verify_splits(train, leak)

            duplicate = root / "duplicate-observation.jsonl"
            dev_rows[0] = {
                **train_rows[0], "state_id": original_dev["state_id"], "scenario_id": original_dev["scenario_id"],
                "option_scores": [{"reward": 0, "native_success": False} for _ in train_rows[0]["options"]],
            }
            for sample in dev_rows[0]["samples"]:
                dev_rows[0]["option_scores"][sample["option_index"]].update(
                    reward=sample["reward"], native_success=sample["native_success"],
                )
            write_replay(duplicate, manifest, dev_rows)
            with self.assertRaisesRegex(ValueError, "1 identical observations"):
                verify_splits(train, duplicate)

    def test_modified_observation_and_frozen_scores_fail(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "dev.jsonl"
            fixture(original, "dev")
            manifest, rows = load_replay(original)
            altered = root / "observation.jsonl"
            expected_hash = rows[0]["observation_sha256"]
            rows[0]["observation_sha256"] = "0" * 64
            write_replay(altered, manifest, rows)
            with self.assertRaisesRegex(ValueError, "observation SHA-256"):
                inspect_replay(altered, "dev")

            altered = root / "score.jsonl"
            rows[0]["observation_sha256"] = expected_hash
            index = rows[0]["samples"][0]["option_index"]
            rows[0]["option_scores"][index]["reward"] += 0.25
            write_replay(altered, manifest, rows)
            with self.assertRaisesRegex(ValueError, "frozen candidate evaluation"):
                inspect_replay(altered, "dev")

    def test_zero_variance_allowed_for_fixture_diagnostics_only(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.jsonl"
            fixture(source, "train")
            manifest, rows = load_replay(source)
            for row in rows:
                row["samples"] = [row["samples"][0]] * len(row["samples"])
            collapsed = root / "collapsed-fixture.jsonl"
            write_replay(collapsed, manifest, rows)
            self.assertEqual(inspect_replay(collapsed, "train")[0]["counts"]["eligible_grpo_groups"], 0)

            actual = root / "collapsed-behavior.jsonl"
            manifest.pop("fixture_only")
            write_replay(actual, manifest, rows)
            with self.assertRaisesRegex(ValueError, "no group with at least two varying rewards"):
                inspect_replay(actual, "train")


if __name__ == "__main__":
    unittest.main()
