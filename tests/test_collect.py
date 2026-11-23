import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from skill_rsi_rl.collect import _prepare_group, _load_replay_spec, behavior_log_probs, collect, merge
from skill_rsi_rl.contracts import load_replay, write_replay
from skill_rsi_rl.scenarios import synthetic_state


class CollectTests(unittest.TestCase):
    def test_logged_exploration_is_the_actual_categorical_mixture(self):
        import torch

        logits = torch.tensor([1.1, -2.0, -1000.0], dtype=torch.float64)
        tempered = (logits / 1.3).softmax(-1)
        for mix in (0.0, 0.35, 1.0):
            logged = behavior_log_probs(logits, 1.3, mix)
            torch.testing.assert_close(logged.exp(), (1 - mix) * tempered + mix / 3)
            if mix:
                self.assertTrue(logged.isfinite().all())

    def test_shared_candidates_reward_each_intervention(self):
        scenario_sets = []
        for split in ("train", "dev"):
            scenarios = [synthetic_state(split, 17, index) for index in range(28)]
            scenario_sets.append({state["scenario_id"] for state in scenarios})
            for state in scenarios:
                options, scores = state["options"], state["scores"]
                self.assertEqual({option["action"]["op"] for option, score in zip(options, scores)
                                  if score["native_success"]}, {state["family"]})
                self.assertEqual(min(score["reward"] for score in scores), 0.0)
        self.assertTrue(scenario_sets[0].isdisjoint(scenario_sets[1]))

    def test_v2_sharding_keeps_heldout_counterfactual_sets(self):
        from collections import defaultdict

        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                fixture=True, replay_spec=None, scenario_suite="v2", split="dev", seed=17, states=16,
                samples_per_state=8, max_prompt_tokens=4096, temperature=1.0, checkpoint_id=None,
                inference_batch_size=3, rank=0, world_size=2, output=root / "shard-0.jsonl",
            )
            collect(args)
            first = args.output
            args.rank, args.output = 1, root / "shard-1.jsonl"
            collect(args)
            output = root / "merged.jsonl"
            merge(output, [first, args.output])
            manifest, groups = load_replay(output)
            by_scenario = defaultdict(list)
            for group in groups:
                by_scenario[group["scenario_id"]].append(group)
            self.assertEqual(manifest["states_per_scenario"], 4)
            self.assertEqual(len(by_scenario), 4)
            for siblings in by_scenario.values():
                self.assertEqual(len({group["contrast_id"] for group in siblings}), 1)
                self.assertEqual({group["family"] for group in siblings}, {"noop", "recall", "compact", "write"})
                self.assertEqual(len({json.loads(group["messages"][1]["content"])["query"] for group in siblings}), 1)

    def test_frozen_shards_match_single_worker_byte_for_byte(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                fixture=True, replay_spec=None, split="dev", seed=17, states=12,
                samples_per_state=8, max_prompt_tokens=4096, temperature=1.0, checkpoint_id=None,
                inference_batch_size=3, rank=0, world_size=1, output=root / "one.jsonl",
            )
            collect(args)
            expected = args.output
            parts = []
            for rank in range(2):
                args.rank, args.world_size, args.output = rank, 2, root / f"shard-{rank}.jsonl"
                collect(args)
                parts.append(args.output)
            merged = root / "merged.jsonl"
            with self.assertRaisesRegex(ValueError, "Every replay shard"):
                merge(merged, parts[:1])
            merge(merged, reversed(parts))
            self.assertEqual(expected.read_bytes(), merged.read_bytes())
            manifest, groups = load_replay(merged)
            self.assertEqual(manifest["shard"], {"rank": 0, "world_size": 1})
            self.assertTrue(manifest["fixture_only"])
            for group in groups:
                for sample in group["samples"]:
                    score = group["option_scores"][sample["option_index"]]
                    self.assertEqual((sample["reward"], sample["native_success"]),
                                     (score["reward"], score["native_success"]))

            incomplete, rejected = root / "incomplete.jsonl", root / "rejected.jsonl"
            manifest, groups = load_replay(parts[0])
            write_replay(incomplete, manifest, groups[:-1])
            with self.assertRaisesRegex(ValueError, "coverage differs|coverage or sample count differs"):
                merge(rejected, [incomplete, parts[1]])
            self.assertFalse(rejected.exists())

    def test_scored_source_needs_explicit_training_permission_and_matching_support(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            spec = {
                "training_permission": "evaluation_only", "split": "train", "dataset_id": "owned-fixture",
                "score_protocol": "fixture-v1", "authorization_ref": "local data owner approval record",
                "states": [synthetic_state("train", 17, 0)],
            }
            source.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "weight_training"):
                _load_replay_spec(source, "train")
            spec["training_permission"] = "weight_training"
            source.write_text(json.dumps(spec), encoding="utf-8")
            states, provenance = _load_replay_spec(source, "train")
            self.assertEqual(provenance["source_kind"], "authorized_replay")
            self.assertEqual(provenance["authorization_ref"], spec["authorization_ref"])
            self.assertEqual(len(provenance["source_sha256"]), 64)
            _prepare_group(states[0], 0, 17)
            states[0]["options"].reverse()
            with self.assertRaisesRegex(ValueError, "shared proposal policy"):
                _prepare_group(states[0], 0, 17)


if __name__ == "__main__":
    unittest.main()
