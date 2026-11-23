import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from . import offline


def decision(draws, scores=None):
    row = {"options": ["A", "B", "C"],
           "samples": [{"option_index": i, "reward": value, "native_success": False} for i, value in draws]}
    if scores is not None:
        row["option_scores"] = [{"reward": value, "native_success": False} for value in scores]
    return row


class BestActionTests(unittest.TestCase):
    def test_unique_winner_can_come_from_an_unsampled_frozen_score(self):
        self.assertEqual(offline.best_action(decision([(1, -3)], [-2, -3, -1])), 2)
        self.assertIsNone(offline.best_action(decision([(0, 1)], [1, 2, 2])))
        with self.assertRaisesRegex(ValueError, "differs"):
            offline.best_action(decision([(0, 3)], [1, 2, 3]))

    def test_sample_scores_require_complete_coverage_and_ignore_draw_frequency(self):
        self.assertIsNone(offline.best_action(decision([(1, 9), (0, 1)])))
        self.assertEqual(offline.best_action(decision([(0, 2), (0, 2), (2, 1), (1, 3)])), 1)
        self.assertIsNone(offline.best_action(decision([(2, 4), (0, 4), (1, 4)])))
        with self.assertRaisesRegex(ValueError, "disagree"):
            offline.best_action(decision([(0, 1), (1, 3), (0, 2)]))

    def test_skip_incomplete_states_and_fail_if_no_training_example_remains(self):
        tokenizer = SimpleNamespace(all_special_ids=[], apply_chat_template=lambda *args, **kwargs: [7],
                                    encode=lambda label, **kwargs: ["ABC".index(label)])
        tokens = dict(options=[{"label": label} for label in "ABC"], prompt_token_ids=[7],
                      option_token_ids=[0, 1, 2], messages=[], state_id="toy")
        complete = {**decision([(0, 0), (1, 3), (2, 2)]), **tokens}
        partial = {**decision([(0, 0)]), **tokens}
        examples, skipped = offline.training_examples([partial, complete], tokenizer, 16, 8)
        self.assertEqual((examples, skipped), ([([7], [0, 1, 2], 1)], 1))
        with self.assertRaisesRegex(ValueError, "no states"):
            offline.training_examples([partial], tokenizer, 16, 8)

    def test_cross_entropy_updates_only_the_offered_actions(self):
        class TinyPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scores = torch.nn.Parameter(torch.tensor([50.0, 0.0, 0.0, 20.0]))

            def forward(self, **kwargs):
                return SimpleNamespace(logits=self.scores.view(1, 1, -1))

        policy = TinyPolicy()
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        loss = offline.example_loss(policy, [0, 3], [1, 2], 1, torch.device("cpu"))
        self.assertAlmostEqual(loss.item(), torch.log(torch.tensor(2.0)).item())
        loss.backward()
        torch.testing.assert_close(policy.scores.grad, torch.tensor([0.0, 0.5, -0.5, 0.0]))
        optimizer.step()
        self.assertLess(offline.example_loss(policy, [0, 3], [1, 2], 1, torch.device("cpu")).item(), loss.item())

    def test_distributed_invocation_fails_before_opening_files(self):
        with TemporaryDirectory() as directory:
            out = Path(directory) / "out"
            with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0"}):
                with self.assertRaisesRegex(ValueError, "one process"):
                    offline.train(SimpleNamespace(replay=Path(directory) / "missing", output=out))
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
