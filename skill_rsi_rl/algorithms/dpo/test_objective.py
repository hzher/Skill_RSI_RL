import copy
import math
from types import SimpleNamespace
import unittest

import torch

from .objective import dpo_loss, online_preferences, preference_pairs, score


def replay(draws, scores=None):
    row = {"options": ["A", "B", "C"], "samples": [
        {"option_index": i, "reward": r, "native_success": False} for i, r in draws]}
    if scores is not None:
        row["option_scores"] = [{"reward": r, "native_success": False} for r in scores]
    return row


def step(index, memory=None):
    return dict(instance_id="hand-1", instance_index=0, state={"memory": memory or {}},
                messages=[{"content": "same input"}], options=["A", "B"],
                prompt_token_ids=[2], option_token_ids=[0, 1], option_index=index)


class PreferenceTests(unittest.TestCase):
    def test_only_sampled_distinct_actions_form_pairs(self):
        self.assertEqual(preference_pairs(replay([(0, 0), (1, 2), (1, 2)], [0, 2, 9])), [(1, 0)])
        self.assertEqual(preference_pairs(replay([(0, 1), (1, 1)])), [])
        self.assertEqual(preference_pairs(replay([(0, 0), (1, 2)]), 2), [])
        self.assertEqual(preference_pairs(replay([(0, 0), (1, 2)]), 1), [(1, 0)])

    def test_inconsistent_or_missing_scores_are_not_filled(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            preference_pairs(replay([(0, 1), (0, 2)]))
        with self.assertRaisesRegex(ValueError, "differs"):
            preference_pairs(replay([(0, 2)], [0, 1, 2]))
        for gap in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                preference_pairs(replay([(0, 1)]), gap)

    def test_hand_computed_loss_and_gradient(self):
        logits = torch.nn.Parameter(torch.zeros(2))
        reference_logits = torch.zeros(2, requires_grad=True)
        reference = reference_logits.log_softmax(-1)
        optimizer = torch.optim.SGD([logits], lr=0.5)
        loss = dpo_loss(logits.log_softmax(-1), reference, 1, 0, 0.2)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([0.1, -0.1]))
        self.assertIsNone(reference_logits.grad)
        optimizer.step()
        self.assertGreater(logits[1].item(), logits[0].item())
        self.assertLess(dpo_loss(logits.log_softmax(-1), reference, 1, 0, 0.2).item(), loss.item())

    def test_reference_margin_matters_and_extreme_logits_stay_finite(self):
        policy = torch.tensor([0.0, 1.0])
        self.assertAlmostEqual(dpo_loss(policy, policy, 1, 0, 1).item(), math.log(2), places=6)
        self.assertTrue(torch.isfinite(dpo_loss(torch.tensor([10000., -10000.]), torch.zeros(2), 1, 0, 1)))
        for beta in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                dpo_loss(policy, policy, 1, 0, beta)

    def test_offered_support_is_scored_and_updated(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.tensor([20., 0., 0., 50.]))

            def forward(self, **kwargs):
                return SimpleNamespace(logits=self.logits.view(1, 1, -1))

        model = Tiny()
        policy = score(model, [0, 3], [1, 2], "cpu")
        dpo_loss(policy, policy.detach(), 1, 0, 1).backward()
        torch.testing.assert_close(model.logits.grad, torch.tensor([0., 0.5, -0.5, 0.]))

    def test_online_pairs_use_identical_state_and_different_replicas(self):
        winner, loser = step(1), step(0)
        rollouts = [{"decisions": [winner]}, {"decisions": [loser]}]
        self.assertEqual(online_preferences(rollouts, [3, 1]), [(winner, loser)])
        self.assertEqual(online_preferences(rollouts, [1, 1]), [])
        changed = copy.deepcopy(rollouts)
        changed[1]["decisions"][0]["state"] = {"memory": {"changed": "value"}}
        self.assertEqual(online_preferences(changed, [3, 1]), [])
        changed = copy.deepcopy(rollouts)
        changed[1]["decisions"][0]["option_index"] = 1
        self.assertEqual(online_preferences(changed, [3, 1]), [])
        changed = copy.deepcopy(rollouts)
        changed[1]["decisions"][0]["messages"] = [{"content": "different feedback"}]
        self.assertEqual(online_preferences(changed, [3, 1]), [])


if __name__ == "__main__":
    unittest.main()
