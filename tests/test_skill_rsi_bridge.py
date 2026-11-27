import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from skill_rsi_rl.skill_rsi_bridge import framework_identity, pin_benchmark, receive


def bundle_fixture(root):
    from skill_rsi.evidence.io import digest, encode

    state = dict(query="Use public evidence", feedback=[], memory={}, host="codex", host_version="synthetic-v1",
                 environment_version="fixture-v1", task_id="public-task", instance_id="instance",
                 instance_index=0, instance_order=["instance"], seed=17,
                 checkpoint_ref={"path": "checkpoint.json", "sha256": "a" * 64, "locator": {"type": "file"}},
                 boundary_ref={"event_id": "boundary"})
    origin = dict(window_id="window", revision="b" * 64, lineage_id="lineage", run_id="run",
                  record_index=0, source="fixture", event_id="event", split="train",
                  source_ref={"path": "state.json", "sha256": "c" * 64, "locator": {"type": "file"}})
    state_id = digest(encode(origin))
    records = {"train/states.jsonl": [{**state, "state_id": state_id}],
               "sources.jsonl": [{**origin, "state_id": state_id, "projection_sha256": digest(encode(state))}]}
    manifest = dict(schema="skill-rsi.training-evidence.v1", dataset_id="synthetic-handoff",
                    score_protocol="fixture-unscored", training_permission="weight_training",
                    authorization_ref="synthetic-test-only", snapshot_sha256="d" * 64, partitions={"train": 1}, files={})
    for name, rows in records.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = b"".join(encode(row) + b"\n" for row in rows)
        path.write_bytes(raw)
        manifest["files"][name] = {"sha256": digest(raw), "bytes": len(raw)}
    manifest["content_sha256"] = digest(encode(manifest))
    (root / "manifest.json").write_bytes(encode(manifest))


class BridgeTests(unittest.TestCase):
    def test_gateway_is_pinned_only_when_selected(self):
        original = {"harness": "dsh", "system_params": {"model": "synthetic"}}
        self.assertIs(pin_benchmark(original), original)
        gateway = {"harness": "codex", "system_params": {"backend": "skill_rsi", "model": "synthetic"}}
        pinned = pin_benchmark(gateway)
        self.assertEqual(pinned["system_params"]["source_id"], framework_identity()[1])
        self.assertNotIn("source_id", gateway["system_params"])
        with self.assertRaisesRegex(ValueError, "changed"):
            pin_benchmark({**gateway, "system_params": {**gateway["system_params"], "source_id": "different"}})

    def test_receipt_accepts_evidence_without_manufacturing_scores_or_tokens(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / "source", root / "received"
            source.mkdir()
            bundle_fixture(source)
            receipt = receive(source, destination, "past-bench")
            self.assertEqual(receipt["accepted"], {"train": 1})
            from skill_rsi.exchange.training import verify_receipt
            verify_receipt(destination, destination / "receiver.json")
            state = json.loads((destination / "pending-states.jsonl").read_text())
            self.assertEqual(state["status"], "awaiting_counterfactual_native_scores")
            self.assertFalse({"reward", "scores", "behavior_logprobs", "prompt_token_ids"} & state.keys())
            with self.assertRaises(FileExistsError):
                receive(source, destination, "past-bench")
            with self.assertRaisesRegex(ValueError, "datasets"):
                receive(source, root / "wrong", "workbuddy")
            (source / "train/states.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "digest"):
                receive(source, root / "corrupt", "cl-bench")


if __name__ == "__main__":
    unittest.main()
