from collections import defaultdict
import unittest

from skill_rsi_rl.contracts import build_messages, propose_options
from skill_rsi_rl.scenarios_v2 import FAMILIES, STATES_PER_SCENARIO, synthetic_state


class CounterfactualScenarioTests(unittest.TestCase):
    def test_exact_query_and_support_shape_require_state_sensitive_choices(self):
        expected_ops = ["noop", "recall", "recall", "compact", "write", "write", "write"]
        winning_labels = defaultdict(set)
        for split, seed in (("train", 17), ("dev", 221)):
            for start in range(0, 256, STATES_PER_SCENARIO):
                rows = [synthetic_state(split, seed, index) for index in range(start, start + STATES_PER_SCENARIO)]
                self.assertEqual(len({row["query"] for row in rows}), 1)
                self.assertEqual(len({row["scenario_id"] for row in rows}), 1)
                self.assertEqual(len({row["contrast_id"] for row in rows}), 1)
                self.assertEqual(len({row["state_id"] for row in rows}), STATES_PER_SCENARIO)
                families = {row["family"]: row for row in rows}
                self.assertEqual(set(families), set(FAMILIES))
                self.assertEqual(families["recall"]["feedback"], families["compact"]["feedback"])
                self.assertEqual(families["noop"]["feedback"], families["write"]["feedback"])
                for row in rows:
                    self.assertEqual(row["options"], propose_options(row["query"], row["feedback"], row["memory"]))
                    self.assertEqual([option["action"]["op"] for option in row["options"]], expected_ops)
                    messages = build_messages(row["query"], row["feedback"], row["memory"], row["options"])
                    self.assertNotIn("contrast_id", messages[1]["content"])
                    best = max(score["reward"] for score in row["scores"])
                    winners = [(option, score) for option, score in zip(row["options"], row["scores"], strict=True)
                               if score["reward"] == best]
                    self.assertEqual({option["action"]["op"] for option, _ in winners}, {row["family"]})
                    self.assertTrue(all(score["native_success"] for _, score in winners))
                    winning_labels[row["family"]].update(option["label"] for option, _ in winners)
        self.assertEqual(winning_labels["recall"], {"B", "C"})
        self.assertEqual(winning_labels["write"], {"E", "F", "G"})

    def test_delayed_write_and_harmful_stale_read(self):
        states = (synthetic_state("dev", 221, index) for index in range(STATES_PER_SCENARIO))
        rows = {state["family"]: state for state in states}
        write = rows["write"]
        self.assertAlmostEqual(write["scores"][0]["reward"], 2 / 3)
        self.assertFalse(write["scores"][0]["native_success"])
        self.assertTrue(any(option["action"]["op"] == "write" and score["native_success"]
                            for option, score in zip(write["options"], write["scores"], strict=True)))
        noop = rows["noop"]
        self.assertEqual(noop["scores"][0], {"reward": 1.0, "native_success": True})
        self.assertFalse(noop["scores"][2]["native_success"])
        self.assertLess(noop["scores"][2]["reward"], 2 / 3)

    def test_heldout_uses_distinct_rendering_and_more_distractors(self):
        train = synthetic_state("train", 17, 0)
        dev = synthetic_state("dev", 17, 0)
        self.assertEqual(train, synthetic_state("train", 17, 0))
        self.assertNotEqual(train["scenario_id"], dev["scenario_id"])
        self.assertEqual(len(train["memory"]) + 1, len(dev["memory"]))
        self.assertIn("record ", " ".join(train["memory"].values()))
        self.assertIn("] -> ", " ".join(dev["memory"].values()))


if __name__ == "__main__":
    unittest.main()
