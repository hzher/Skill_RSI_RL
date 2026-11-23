from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from skill_rsi_rl.algorithms.api import read_training_replay, validate_replay_tokens
from skill_rsi_rl.collect import collect
from skill_rsi_rl.contracts import load_replay


class AlgorithmApiTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "replay.jsonl"
        collect(SimpleNamespace(
            fixture=True, replay_spec=None, split="train", seed=17, states=1, samples_per_state=2,
            max_prompt_tokens=4096, temperature=1.0, checkpoint_id=None, rank=0, world_size=1, output=self.path,
        ))
        self.manifest, (self.row,) = load_replay(self.path)
        self.manifest.update(fixture_only=False, behavior_checkpoint_id="sha256:" + "1" * 64)
        self.row["samples"] = [self.row["samples"][0]] * 2

    def write_snapshot(self, manifest, row):
        # Also permits writing malformed input for the public reader to reject.
        self.path.write_text(json.dumps(manifest) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    def test_flat_reward_snapshot_keeps_identity_and_hash_with_or_without_shard_metadata(self):
        for manifest in (self.manifest, {k: v for k, v in self.manifest.items() if k != "shard"}):
            with self.subTest(shard=manifest.get("shard")):
                self.write_snapshot(manifest, self.row)
                saved, rows, fingerprint = read_training_replay(self.path)
                self.assertEqual(saved, manifest)
                self.assertEqual(rows, [self.row])
                self.assertEqual(fingerprint, sha256(self.path.read_bytes()).hexdigest())

    def test_ineligible_or_incomplete_training_manifest_is_rejected(self):
        for overrides, message in (
            ({"training_permission": "evaluation_only"}, "weight_training"),
            ({"split": "dev"}, "split=train"),
            ({"fixture_only": True}, "fixture"),
            ({"shard": {"rank": 1, "world_size": 2}}, "shard"),
            ({"behavior_checkpoint_id": "  "}, "behavior_checkpoint_id"),
            ({"behavior_checkpoint_id": None}, "behavior_checkpoint_id"),
            ({"state_count": 2}, "group count"),
        ):
            with self.subTest(overrides=overrides):
                self.write_snapshot({**self.manifest, **overrides}, self.row)
                with self.assertRaisesRegex(ValueError, message):
                    read_training_replay(self.path)

    def test_changed_observation_identity_is_rejected(self):
        self.write_snapshot(self.manifest, {**self.row, "observation_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "observation SHA-256"):
            read_training_replay(self.path)

    def test_tokenizer_must_reproduce_allowed_prompt_and_action_ids(self):
        tokenizer = Mock(all_special_ids=[])
        tokenizer.apply_chat_template.return_value = self.row["prompt_token_ids"]
        tokenizer.encode.side_effect = lambda label, **kwargs: [ord(label)]
        limits = {"vocab_size": 256, "max_prompt_tokens": len(self.row["prompt_token_ids"])}
        validate_replay_tokens(self.row, tokenizer, **limits)

        for change, limits_change, message in (
            ({}, {"max_prompt_tokens": limits["max_prompt_tokens"] - 1}, "prompt length"),
            ({"option_token_ids": [256]}, {}, "vocabulary"),
            ({"prompt_token_ids": [0]}, {}, "tokenization differs"),
            ({"option_token_ids": self.row["option_token_ids"][::-1]}, {}, "tokenization differs"),
        ):
            with self.subTest(change=change, limits_change=limits_change):
                with self.assertRaisesRegex(ValueError, message):
                    validate_replay_tokens({**self.row, **change}, tokenizer, **{**limits, **limits_change})


if __name__ == "__main__":
    unittest.main()
