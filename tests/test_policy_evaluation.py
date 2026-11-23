from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import http.client
import json
import math
from pathlib import Path
import tempfile
from threading import Event, Thread
import unittest
from unittest.mock import patch

from skill_rsi_rl.contracts import build_messages, digest, propose_options, write_replay
from skill_rsi_rl.dev_eval import _open_progress, _training_receipt, evaluate_checkpoints, summarize_choices, verify_dev_groups
from skill_rsi_rl.scenarios import synthetic_state
from skill_rsi_rl.serve import checkpoint_identities, make_server


@contextmanager
def serve(decide, **limits):
    with make_server("127.0.0.1", 0, "test", "checkpoint", decide, **limits) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield server.server_port
        finally:
            server.shutdown()
            worker.join()


def request(port, path, value=None):
    client = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        method = "GET" if value is None else "POST"
        client.request(method, path, body=None if value is None else json.dumps(value))
        response = client.getresponse()
        return response.status, json.loads(response.read())
    finally:
        client.close()


def recorded_group(seed, index):
    state = synthetic_state("dev", seed, index)
    options = state["options"]
    scores = state["scores"]
    messages = build_messages(state["query"], state["feedback"], state["memory"], options)
    return {"state_id": state["state_id"], "scenario_id": state["scenario_id"], "state_index": index,
            "messages": messages, "observation_sha256": digest(messages), "options": options,
            "prompt_token_ids": [1, 2], "option_token_ids": list(range(3, 3 + len(options))),
            "behavior_logprobs": [-math.log(len(options))] * len(options), "option_scores": scores,
            "samples": [{"option_index": i, **scores[i]} for i in (0, 1)]}


def contrast_group(scenario, query, variation, rewards):
    memory = {"first": f"first {variation}", "second": f"second {variation}"}
    options = propose_options(query, [], memory)
    return {"scenario_id": scenario, "messages": build_messages(query, [], memory, options), "options": options,
            "option_scores": [{"reward": score, "native_success": score > 0.5} for score in rewards]}


class ServeTests(unittest.TestCase):
    def test_http_contract_validates_state_and_support(self):
        observed = []

        def decide(**state):
            observed.append(state)
            return {"op": "noop"} if state["query"] == "valid" else {"op": "recall", "keys": ["absent"]}

        state = {"query": "valid", "feedback": [f"old {index}" for index in range(6)], "memory": {}}
        with serve(decide) as port:
            self.assertEqual(request(port, "/health"), (200, {"model": "test", "policy_id": "checkpoint"}))
            self.assertEqual(request(port, "/decide", state),
                             (200, {"model": "test", "policy_id": "checkpoint", "action": {"op": "noop"}}))
            self.assertEqual(observed[0]["feedback"], state["feedback"][-4:])
            self.assertEqual(request(port, "/act", {**state, "memory": []})[0], 400)
            with self.assertLogs(level="ERROR"):
                self.assertEqual(request(port, "/act", {**state, "query": "invalid action"})[0], 500)

    def test_busy_policy_keeps_health_available(self):
        entered = Event()
        release = Event()

        def decide(**_):
            entered.set()
            release.wait(2)
            return {"op": "noop"}

        state = {"query": "state", "feedback": [], "memory": {}}
        with serve(decide, max_inflight=1) as port:
            result = []
            slow_request = Thread(target=lambda: result.append(request(port, "/decide", state)), daemon=True)
            slow_request.start()
            try:
                self.assertTrue(entered.wait(1))
                self.assertEqual(request(port, "/decide", state)[0], 503)
                self.assertEqual(request(port, "/health")[0], 200)
            finally:
                release.set()
                slow_request.join()
            self.assertEqual(result[0][0], 200)


class DevelopmentEvaluationTests(unittest.TestCase):
    def test_checkpoint_sweep_reuses_base_and_selects_reward_at_base_success(self):
        group = recorded_group(31, 0)
        group["option_scores"] = [
            {"reward": reward, "native_success": success}
            for reward, success in ((0.3, True), (0.6, True), (1.0, False), (0.6, True))
        ]
        group["samples"] = [{"option_index": i, **group["option_scores"][i]} for i in (0, 1)]
        states = []
        progress_path = None
        choices = {"checkpoint-0001": 1, "checkpoint-0002": 2, "checkpoint-0010": 3}

        class Policy:
            def __init__(self, _model, adapter, device, _max_prompt_tokens):
                states.append("loaded_base")
                self.device = device
                self.model = self
                self.active = "default"
                self.adapters = {"default": adapter.name}

            def action_tokens(self, *_):
                states.append("validated_tokens")
                return group["prompt_token_ids"], group["option_token_ids"]

            @contextmanager
            def disable_adapter(self):
                self.active = None
                try:
                    yield
                finally:
                    self.active = "default"

            def load_adapter(self, path, name, **_):
                assert name not in self.adapters
                self.adapters[name] = path.name
                if progress_path is not None:
                    sent = [json.loads(line)["epoch"] for line in progress_path.read_text().splitlines()]
                    states.append(("journal_epochs", sent))

            def set_adapter(self, name):
                self.active = name

            def delete_adapter(self, name):
                assert name != self.active
                del self.adapters[name]

            def choose_tokens(self, *_):
                name = self.adapters.get(self.active)
                states.append(name)
                return choices[name] if name else 0

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            model = directory / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "model.safetensors").write_bytes(b"model fixture")
            checkpoints = directory / "train"
            for name in reversed(choices):
                checkpoint = checkpoints / name
                checkpoint.mkdir(parents=True)
                (checkpoint / "adapter_config.json").write_text("{}")
                (checkpoint / "adapter_model.safetensors").write_text(name)
            (checkpoints / ".checkpoint-0012").mkdir()
            replay = directory / "dev.jsonl"
            write_replay(replay, {"schema_version": 1, "split": "dev", "dataset_id": "test",
                                  "training_permission": "weight_training", "source_kind": "synthetic",
                                  "behavior_model": "fixture"}, [group])
            with patch("skill_rsi_rl.dev_eval.QwenPolicy", Policy):
                with self.assertRaisesRegex(ValueError, "Expected checkpoints for epochs"):
                    evaluate_checkpoints(replay, model, checkpoints, "cpu", expected_epochs=3, samples=100)
                report = evaluate_checkpoints(replay, model, checkpoints, "cpu", samples=100)
                self.assertEqual(states, ["loaded_base", "validated_tokens", None, *choices])
                self.assertEqual(report["selected_epoch"], 1)
                self.assertEqual(report["selected_checkpoint"], str((checkpoints / "checkpoint-0001").resolve()))
                self.assertEqual(report["checkpoint_selection"]["eligible_epochs"], [1, 10])
                self.assertFalse(report["checkpoint_selection"]["fallback_used"])
                self.assertEqual(report["trained"]["reward_mean"], 0.6)
                self.assertEqual(report["policy_id"], checkpoint_identities(model, report["selected_checkpoint"])[1])
                self.assertEqual([row["epoch"] for row in report["checkpoints"]], [1, 2, 10])
                self.assertFalse(report["checkpoints"][1]["meets_base_success_rate"])

                states.clear()
                progress_path = directory / "dev-progress.jsonl"
                with _open_progress(progress_path, automatic=True) as progress:
                    report = evaluate_checkpoints(replay, model, checkpoints, "cpu", eval_epochs=[2, 12, 2],
                                                  samples=100, progress=progress)
                self.assertEqual(states, ["loaded_base", "validated_tokens", None, "checkpoint-0001",
                                          ("journal_epochs", [1]), "checkpoint-0002",
                                          ("journal_epochs", [1, 2]), "checkpoint-0010"])
                self.assertEqual(report["selected_epoch"], 1)
                self.assertEqual(report["checkpoint_selection"]["requested_epochs"], [2, 12])
                self.assertEqual(report["checkpoint_selection"]["evaluated_epochs"], [1, 2, 10])
                self.assertEqual(report["checkpoint_selection"]["unavailable_requested_epochs"], [12])
                self.assertEqual(report["checkpoint_selection"]["latest_committed_epoch"], 10)
                records = progress_path.read_text().splitlines(keepends=True)
                self.assertTrue(all(line.endswith("\n") for line in records))
                self.assertEqual([json.loads(line)["epoch"] for line in records], [1, 2, 10])
                self.assertEqual(json.loads(records[0])["paired_deltas"], report["paired_deltas"])
                with self.assertRaises(FileExistsError):
                    _open_progress(progress_path, automatic=False)
                with _open_progress(progress_path, automatic=True) as retry:
                    self.assertTrue(Path(retry.name).name.startswith("dev-progress-retry-"))
                    self.assertEqual(progress_path.read_text(), "".join(records))
                progress_path = None

                choices.update({name: 2 for name in choices})
                report = evaluate_checkpoints(replay, model, checkpoints, "cpu", samples=100)
                self.assertEqual(report["selected_epoch"], 1)
                self.assertEqual(report["trained"]["reward_mean"], 1.0)
                self.assertTrue(report["checkpoint_selection"]["fallback_used"])

                parent = directory / "parent"
                parent.mkdir()
                (parent / "adapter_config.json").write_text("{}")
                (parent / "adapter_model.safetensors").write_text("parent")
                (checkpoints / "run.json").write_text(json.dumps(
                    {"settings": {}, "initial_adapter": {"sha256": sha256(b"parent").hexdigest()}}
                ))
                choices.update({"parent": 1, "checkpoint-0001": 0, "checkpoint-0002": 2, "checkpoint-0010": 3})
                group["family"] = "recall"
                replay.unlink()
                write_replay(replay, {"schema_version": 1, "split": "dev", "dataset_id": "test",
                                      "training_permission": "weight_training", "source_kind": "synthetic",
                                      "behavior_model": "fixture"}, [group])
                states.clear()
                report = evaluate_checkpoints(replay, model, checkpoints, "cpu", compare_adapter=parent, samples=100)
                self.assertEqual(states, ["loaded_base", "validated_tokens", None, "parent",
                                          "checkpoint-0001", "checkpoint-0002", "checkpoint-0010"])
                self.assertTrue(report["checkpoint_selection"]["accepted"])
                self.assertEqual(report["checkpoint_selection"]["eligible_epochs"], [10])
                self.assertEqual(report["selected_epoch"], 10)
                comparison = report["parent_comparison"]
                self.assertEqual(comparison["checkpoint"], str(parent.resolve()))
                self.assertEqual(comparison["policy_id"], checkpoint_identities(model, parent)[1])
                self.assertEqual(comparison["parent"]["reward_mean"], 0.6)
                self.assertEqual(comparison["candidate"]["successes"], 1)
                self.assertEqual(comparison["paired_deltas"]["reward"]["mean"], 0.0)
                self.assertIn("parent", comparison["strata"]["recall"])
                self.assertIn("candidate", comparison["counterfactual"])
                self.assertEqual(report["base"]["reward_mean"], 0.3)

                choices["checkpoint-0010"] = 0
                report = evaluate_checkpoints(replay, model, checkpoints, "cpu", compare_adapter=parent, samples=100)
                self.assertFalse(report["checkpoint_selection"]["accepted"])
                self.assertTrue(report["checkpoint_selection"]["fallback_used"])
                self.assertEqual(report["checkpoint_selection"]["eligible_epochs"], [])
                self.assertEqual(report["selected_epoch"], 2)
                self.assertEqual(report["parent_comparison"]["success_delta_pp"], -100)
                self.assertEqual(Path(report["selected_checkpoint"]).parent, checkpoints.resolve())
                (parent / "adapter_model.safetensors").write_text("different")
                with self.assertRaisesRegex(ValueError, "recorded initial adapter"):
                    evaluate_checkpoints(replay, model, checkpoints, "cpu", compare_adapter=parent, samples=100)

    def test_canonical_prompt_and_contested_option_scores(self):
        group = recorded_group(31, 0)
        manifest = {"split": "dev", "state_count": 1, "dataset_id": "test",
                    "training_permission": "weight_training", "source_kind": "synthetic"}
        verify_dev_groups(manifest, [group])
        wrong = deepcopy(group)
        wrong["messages"][0]["content"] = "unrecorded prompt"
        with self.assertRaisesRegex(ValueError, "shared policy support"):
            verify_dev_groups(manifest, [wrong])
        wrong = deepcopy(group)
        wrong["option_scores"][group["samples"][0]["option_index"]]["reward"] = 100
        with self.assertRaisesRegex(ValueError, "frozen candidate evaluation"):
            verify_dev_groups(manifest, [wrong])

    def test_paired_scenario_bootstrap_and_native_success(self):
        groups = [recorded_group(51, index) for index in range(3)]
        groups[0]["scenario_id"] = groups[1]["scenario_id"]
        for group in groups:
            group["option_scores"][:2] = [{"reward": 0.9, "native_success": False},
                                           {"reward": 0.2, "native_success": True}]
        manifest = {"source_kind": "synthetic", "scenario_families": ["recall", "compact", "write", "noop"]}
        result = summarize_choices(manifest, groups, [0] * 3, [1] * 3, samples=100)
        self.assertEqual(result["paired_scenarios"], 2)
        self.assertEqual(result["base"]["successes"], 0)
        self.assertEqual(result["trained"]["successes"], 3)
        self.assertEqual(result["success_delta_pp"], 100)
        self.assertAlmostEqual(result["reward_delta"], -0.7)
        self.assertEqual(set(result["strata"]), {"recall", "compact", "write"})
        self.assertIsNone(result["strata"]["recall"]["paired_deltas"]["reward"]["ci95"])
        self.assertEqual(result["noop_baseline"]["reward_mean"], 0.9)
        self.assertEqual(result["counterfactual"]["eligible_sets"], 0)
        self.assertIsNone(result["counterfactual"]["trained"]["pass_rate"])

    def test_counterfactual_requires_correct_actions_on_all_siblings(self):
        groups = [
            contrast_group("a", "same query", "already known", [1, 0, 0, 0]),
            contrast_group("a", "same query", "needs recall", [0, 1, 1 - 1e-12, 0]),
            contrast_group("a", "same query", "needs compact", [0, 0, 0, 1]),
            contrast_group("b", "another query", "already known", [1, 0, 0, 0]),
            contrast_group("b", "another query", "needs recall", [0, 1, 0, 0]),
        ]
        result = summarize_choices({}, iter(groups), [0, 0, 0, 0, 0], [0, 2, 3, 0, 1], samples=100)
        contrast = result["counterfactual"]
        self.assertEqual((contrast["eligible_sets"], contrast["eligible_states"], contrast["eligible_scenarios"]),
                         (2, 5, 2))
        self.assertEqual(contrast["trained"], {"sets_passed": 2, "pass_rate": 1.0})
        self.assertEqual(contrast["base"]["sets_passed"], 0)
        self.assertEqual(contrast["noop"]["sets_passed"], 0)
        self.assertEqual(contrast["paired_delta_pp"], {"mean": 100.0, "ci95": [100.0, 100.0]})

    def test_counterfactual_excludes_common_oracle_ops_and_indistinguishable_states(self):
        groups = [
            contrast_group("a", "same query", "version one", [1, 1 - 1e-12, 0, 0]),
            contrast_group("a", "same query", "version two", [0, 0, 1, 0]),
            contrast_group("b", "same query", "same input", [1, 0, 0, 0]),
            contrast_group("b", "same query", "same input", [0, 1, 0, 0]),
            contrast_group("c", "first query", "version one", [1, 0, 0, 0]),
            contrast_group("c", "second query", "version two", [0, 1, 0, 0]),
        ]
        result = summarize_choices({}, groups, [0] * 6, [0, 2, 0, 1, 0, 1], samples=100)
        contrast = result["counterfactual"]
        self.assertEqual(contrast["same_query_sets"], 2)
        self.assertEqual(contrast["excluded_duplicate_observation_sets"], 1)
        self.assertEqual(contrast["eligible_sets"], 0)
        self.assertEqual(contrast["paired_delta_pp"], {"mean": None, "ci95": None})

    def test_counterfactual_rejects_same_optimal_label_for_different_operations(self):
        groups = []
        for memory, feedback in [({"item": "visible fact"}, []), ({}, ["visible fact"])]:
            options = propose_options("same query", feedback, memory)
            groups.append({"scenario_id": "a", "messages": build_messages("same query", feedback, memory, options),
                           "options": options, "option_scores": [
                               {"reward": int(index == 1), "native_success": index == 1}
                               for index in range(len(options))
                           ]})
        self.assertNotEqual(groups[0]["options"][1]["action"]["op"], groups[1]["options"][1]["action"]["op"])
        result = summarize_choices({}, groups, [0, 0], [1, 1], samples=100)
        self.assertEqual(result["counterfactual"]["excluded_constant_label_sets"], 1)
        self.assertEqual(result["counterfactual"]["eligible_sets"], 0)

    def test_train_overlap_rejects_even_when_state_ids_differ(self):
        group = recorded_group(31, 0)
        manifest = {"schema_version": 1, "split": "train", "training_permission": "weight_training",
                    "source_kind": "synthetic", "dataset_id": "check", "behavior_model": "fixture"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            write_replay(path, manifest, [{**group, "state_id": "separate state"}])
            with self.assertRaisesRegex(ValueError, "1 scenarios"):
                _training_receipt(path, [group], {}, "dev-digest")
            with self.assertRaisesRegex(ValueError, "1 identical observations"):
                _training_receipt(path, [{**group, "scenario_id": "separate scenario"}], {}, "dev-digest")
            self.assertTrue(_training_receipt(path, [recorded_group(51, 0)], {}, "dev-digest")["scenario_disjoint_verified"])
