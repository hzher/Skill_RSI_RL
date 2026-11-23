import json
import tempfile
import unittest
from pathlib import Path

from skill_rsi_rl.preflight import archive, check, checkpoint


class PreflightTests(unittest.TestCase):
    def test_checks_every_weight_from_the_index(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            config = {"model_type": "qwen3", "vocab_size": 16, "max_position_embeddings": 128}
            (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (model / name).write_text("{}", encoding="utf-8")
            files = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
            (model / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict(zip(("first", "last"), files))}), encoding="utf-8",
            )
            (model / files[0]).write_bytes(b"checkpoint")
            missing = check(checkpoint, model)
            self.assertFalse(missing["ok"])
            self.assertIn(files[1], missing["error"])
            (model / files[1]).write_bytes(b"shard")
            complete = check(checkpoint, model)
            self.assertTrue(complete["ok"])
            self.assertEqual(complete["weight_shards"], 2)
            self.assertEqual(complete["weight_bytes"], len(b"checkpointshard"))

    def test_reports_evaluation_only_source_separately_from_training_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory)
            (batch / "manifest.json").write_text(
                json.dumps({"training_permission": "evaluation_only"}), encoding="utf-8",
            )
            index = batch / "derived" / "skill-rsi-v1"
            index.mkdir(parents=True)
            (index / "batch_index.json").write_text(
                json.dumps({"batch_id": "fixed", "runs": [{"run_id": "r1"}]}), encoding="utf-8",
            )
            result = check(archive, batch)
            self.assertTrue(result["ok"])
            self.assertEqual(result["training_permission"], "evaluation_only")
            self.assertEqual(result["indexed_runs"], 1)


if __name__ == "__main__":
    unittest.main()
