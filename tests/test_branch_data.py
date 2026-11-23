from types import SimpleNamespace
import unittest

from skill_rsi_rl.branch_data import build_state, initial_memory, observation_from_prefix
from skill_rsi_rl.contracts import MAX_FEEDBACK_ITEMS, digest


def interaction(index):
    return {
        "query": {"prompt": f"Question {index}", "feedback": None if index == 0 else {"content": f"Reply {index - 1}"}},
        "observation": {"content": f"Reply {index}", "metadata": {"evaluator": "hidden outcome"}},
        "response": {"action": {"reasoning": "private chain"}},
    }


class BranchDataTests(unittest.TestCase):
    def test_prefill_uses_only_feedback_before_each_retained_decision(self):
        rows = [interaction(index) for index in range(7)]
        source = SimpleNamespace(trace={"interactions": rows})
        self.assertEqual(initial_memory(source, 1), {})
        memory = initial_memory(source, 6)
        self.assertEqual(set(memory.values()), {f"Reply {index}" for index in range(5)})
        state = observation_from_prefix({"interactions": rows[:6]}, rows[6]["query"], memory=memory)
        self.assertEqual(state["query"], "Question 6")
        self.assertEqual(state["feedback"], [f"Reply {index}" for index in range(6 - MAX_FEEDBACK_ITEMS, 6)])
        self.assertNotIn("Reply 5", state["memory"].values())
        self.assertNotIn("private chain", str(state))
        self.assertNotIn("hidden outcome", str(state))

    def test_seed_family_stays_together_while_state_tracks_boundary_and_horizon(self):
        def state(trace, boundary, targets):
            source = SimpleNamespace(
                receipt={"task": "poker", "benchmark_resources_sha256": "board", "trace_sha256": trace},
                attempt={"seed": 42, "stage_seeds": [1, 2]},
            )
            observation = {"query": "Read the last note", "feedback": ["Observed clue"], "memory": {}}
            start = SimpleNamespace(
                receipt={"trace_sha256": trace, "boundary_sha256": boundary, "state_sha256": digest(observation),
                         "score_window_sha256": digest(targets)},
                target_outcomes=targets, **observation,
            )
            return build_state(source, start)

        first = state("trace-a", "early", (("one", 1),))
        second = state("trace-b", "mid", (("two", 2),))
        longer = state("trace-a", "early", (("one", 1), ("two", 2)))
        self.assertEqual(first["scenario_id"], second["scenario_id"])
        self.assertEqual(len({first["state_id"], second["state_id"], longer["state_id"]}), 3)
        self.assertEqual([option["action"]["op"] for option in first["options"]], ["noop", "write"])


if __name__ == "__main__":
    unittest.main()
