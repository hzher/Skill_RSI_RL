import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from skill_rsi_rl.train import Worker
    from skill_rsi_rl.training_artifacts import (
        TrainingProgress,
        initialize_run,
        load_training_state,
        lock_output,
        save_checkpoint,
    )


class ExampleAdapter:
    def save_pretrained(self, directory):
        (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
        (directory / "adapter_model.safetensors").write_bytes(b"example adapter")


class ExampleOptimizer:
    def state_dict(self):
        return {"state": {}, "param_groups": []}


@unittest.skipIf(torch is None, "requires PyTorch")
class TrainingArtifactsTests(unittest.TestCase):
    def test_checkpoint_is_atomic_and_rejects_changed_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "checkpoint-0001"
            settings = {"replay_sha256": "a"}
            model = ExampleAdapter()
            optimizer = ExampleOptimizer()
            with patch("skill_rsi_rl.training_artifacts.torch.save", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    save_checkpoint(destination, model, optimizer, settings, 1, 8)
            self.assertEqual(list(root.iterdir()), [])
            save_checkpoint(destination, model, optimizer, settings, 1, 8, loss_ema=0.125)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                save_checkpoint(destination, model, optimizer, settings, 1, 8)
            self.assertEqual(load_training_state(destination, settings, 2),
                             TrainingProgress(1, 8, optimizer.state_dict(), 0.125))
            (destination / "adapter_model.safetensors").write_bytes(b"unexpected adapter")
            with self.assertRaisesRegex(ValueError, "adapter files do not match"):
                load_training_state(destination, settings, 2)

    def test_existing_checkpoints_without_integrity_hashes_still_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            settings = {"replay_sha256": "a"}
            state = {"settings": settings, "next_epoch": 1, "step": 3,
                     "optimizer": ExampleOptimizer().state_dict()}
            ExampleAdapter().save_pretrained(path)
            torch.save(state, path / "training_state.pt")
            recovered = load_training_state(path, settings, 2)
            self.assertEqual(recovered.step, 3)
            self.assertIsNone(recovered.loss_ema)
            state["step"] = None
            torch.save(state, path / "training_state.pt")
            with self.assertRaisesRegex(ValueError, "invalid epoch, step, or optimizer"):
                load_training_state(path, settings, 2)

    def test_metadata_and_output_conflict_checks_are_locked_and_nonoverwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_data = {"settings": {"backend": "native"}}
            with lock_output(output):
                with self.assertRaisesRegex(RuntimeError, "already locked"):
                    with lock_output(output):
                        self.fail("a second process could open the same training output")
                metrics = initialize_run(output, run_data, TrainingProgress(), 2)
                self.assertEqual(metrics.name, "metrics.jsonl")
                self.assertEqual(json.loads((output / "run.json").read_text()), run_data)
                (output / "checkpoint-0002").mkdir()
                with self.assertRaisesRegex(ValueError, "epoch scheduled to train"):
                    initialize_run(output, run_data, TrainingProgress(1, 8), 2)


@unittest.skipIf(torch is None, "requires PyTorch")
class TrainingWorkerTests(unittest.TestCase):
    def test_primary_failure_is_reported_before_other_rank_can_continue(self):
        captured = []

        def broadcast(value, **kwargs):
            if value[0] is not None:
                captured.extend(value)
            else:
                value[:] = captured

        def fail():
            raise OSError("disk full")

        device = torch.device("cpu")
        with patch("skill_rsi_rl.train.dist.broadcast_object_list", side_effect=broadcast):
            with self.assertRaisesRegex(OSError, "disk full"):
                Worker(0, 2, device).run_on_primary("save checkpoint", fail)
            with self.assertRaisesRegex(RuntimeError, "save checkpoint failed on rank 0: OSError: disk full"):
                Worker(1, 2, device).run_on_primary("save checkpoint", self.fail)

    def test_mismatched_replay_settings_raise_before_ddp(self):
        def gather(output, value):
            output[:] = [value, (None, {"replay_sha256": "other"})]

        with patch("skill_rsi_rl.train.dist.all_gather_object", side_effect=gather):
            worker = Worker(0, 2, torch.device("cpu"))
            with self.assertRaisesRegex(RuntimeError, "settings differs across ranks"):
                worker.run_on_all_ranks("settings", lambda: {"replay_sha256": "local"},
                                        require_same_result=True)


if __name__ == "__main__":
    unittest.main()
