import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from skill_rsi_rl.branch_score import _load_authorization, _plan_group, _saved_scoring_plan, _score_groups
from skill_rsi_rl.contracts import digest


class BranchScoreTest(unittest.TestCase):
    def test_resume_reuses_successes_and_only_retries_failed_candidate(self):
        start = SimpleNamespace(receipt={"boundary_sha256": "boundary", "state_sha256": "state",
                                         "score_window_sha256": "window"}, target_outcomes=(("instance", 1),))
        state = {"state_id": "state-id", "scenario_id": "scenario", "options": [
            {"label": "A", "action": {"op": "noop"}},
            {"label": "B", "action": {"op": "write", "key": "k", "text": "visible"}},
        ]}
        expected = {"groups": [_plan_group(start, state)], "source": {"source_training_permission": "evaluation_only"}}
        plan = {**expected, "executor": {"model": "fixture-executor", "sha256": "a" * 64}}
        called = []

        def execute(start, *, action, **kwargs):
            called.append(action["op"])
            success = action["op"] == "write" and called.count("write") > 1
            failed = action["op"] == "write" and not success
            outcome = {"instance_id": "instance", "instance_index": 1, "reward": float(success), "native_success": success}
            return {**start.receipt, "target_outcomes": [["instance", 1]], "action_sha256": digest(action), "queries": 1,
                    "status": "failed" if failed else "scored", "reward": None if failed else float(success),
                    "native_success": None if failed else success, "outcomes": [] if failed else [outcome],
                    "executor": {"model": "fixture-executor"}}

        def save(path, value):
            path.write_text(json.dumps(value))

        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.assertFalse(_score_groups([(start, state)], directory, plan, {}, execute, save)[1])
            with self.assertRaisesRegex(ValueError, "Missing or incomplete candidate receipt"):
                _score_groups([(start, state)], directory, _saved_scoring_plan(directory, expected))
            self.assertFalse(_score_groups([(start, state)], directory, plan, {}, execute, save)[1])
            report, states = _score_groups([(start, state)], directory, plan, {}, execute, save, retry_failed=True)
            self.assertEqual(called, ["noop", "write", "write"])
            self.assertEqual(states[0]["scores"], [{"reward": 0, "native_success": False}, {"reward": 1, "native_success": True}])
            self.assertTrue(report["groups"][0]["varying_reward"])
            with self.assertRaisesRegex(ValueError, "differs from the verified archive"):
                _saved_scoring_plan(directory, {**expected, "max_queries": 100})
            previous_files = {file: file.read_bytes() for file in directory.rglob("*.json")}
            self.assertEqual(_score_groups([(start, state)], directory, _saved_scoring_plan(directory, expected)),
                             (report, states))
            self.assertEqual({file: file.read_bytes() for file in directory.rglob("*.json")}, previous_files)
            missing_state = {**state, "state_id": "missing-state"}
            partial_plan = {**plan, "groups": [plan["groups"][0], _plan_group(start, missing_state)]}
            save(directory / "branch-plan.json", partial_plan)
            partial_report, complete = _score_groups([(start, state), (start, missing_state)], directory,
                                                     partial_plan, complete_only=True)
            self.assertEqual(complete, states)
            self.assertEqual(partial_report["complete_groups"], 1)
            self.assertEqual([action["status"] for action in partial_report["groups"][1]["actions"]],
                             ["missing", "missing"])
            self.assertEqual(called, ["noop", "write", "write"])
            save(directory / "branch-plan.json", plan)
            receipt = directory / "branches" / state["state_id"] / "B.json"
            candidate = json.loads(receipt.read_text())
            candidate["executor"]["model"] = "different-model"
            save(receipt, candidate)
            with self.assertRaisesRegex(ValueError, "different executor model"):
                _score_groups([(start, state)], directory, plan)

    def test_training_authorization_requires_the_exact_manifest(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "permission.json"
            approval = {"training_permission": "weight_training", "authorization_ref": "signed-source",
                        "batch_manifest_sha256": "hash"}
            path.write_text(json.dumps(approval))
            self.assertEqual(_load_authorization(path, {"batch_manifest_sha256": "hash"}), "signed-source")
            with self.assertRaisesRegex(ValueError, "this source manifest"):
                _load_authorization(path, {"batch_manifest_sha256": "other"})
            approval["tasks"] = ["exploitable_poker"]
            path.write_text(json.dumps(approval))
            source = {"batch_manifest_sha256": "hash", "task": "exploitable_poker"}
            self.assertEqual(_load_authorization(path, source), "signed-source")
            with self.assertRaisesRegex(ValueError, "does not cover this source task"):
                _load_authorization(path, {**source, "task": "blind_spectrum_monitoring"})
