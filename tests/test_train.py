import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from skill_rsi_rl.train import (
        configuration,
        group_losses,
        learning_rate_at_step,
        option_logprobs,
        prepare_groups,
        replay_policy_diagnostics,
        time_for_next_epoch,
    )


@unittest.skipIf(torch is None, "requires PyTorch")
class GRPOTests(unittest.TestCase):
    def test_warmup_cosine_and_floor_use_global_update_index(self):
        rates = [learning_rate_at_step(1.0, step, 2, 6, 0.2) for step in range(9)]
        self.assertEqual(rates[:3], [0.5, 1.0, 1.0])
        self.assertTrue(all(a > b for a, b in zip(rates[2:5], rates[3:6])))
        self.assertEqual(rates[5:], [0.2] * 4)
        self.assertEqual(learning_rate_at_step(1.0, 100, 0, 0, 0.2), 1.0)

    def test_next_epoch_requires_last_duration_and_margin(self):
        self.assertTrue(time_for_next_epoch(1000, 685, 300))
        self.assertFalse(time_for_next_epoch(1000, 686, 300))
        self.assertTrue(time_for_next_epoch(1000, 960, 30))
        self.assertFalse(time_for_next_epoch(1000, 961, 30))

    def test_deadline_is_optional_and_does_not_set_decay_horizon(self):
        args = ["train", "--replay", "train.jsonl", "--output", "run", "--lr-decay-epochs", "6"]
        with patch("sys.argv", args):
            self.assertIsNone(configuration().stop_at_unix)
        with patch("sys.argv", [*args, "--stop-at-unix", "1800000000"]):
            config = configuration()
        self.assertEqual(config.stop_at_unix, 1800000000)
        self.assertEqual(config.lr_decay_epochs, 6)

    def test_advantage_uses_every_draw_and_skips_uninformative_states(self):
        def row(state, rewards):
            return {
                "state_id": state,
                "prompt_token_ids": [2, 3],
                "option_token_ids": [4, 5],
                "behavior_logprobs": [-math.log(2)] * 2,
                "samples": [{"option_index": int(reward > 0), "reward": reward} for reward in rewards],
            }

        groups, skipped = prepare_groups(
            [row("varied", [0., 2., 2.]), row("constant", [1., 1.]), row("singleton", [1.])], 8, 8,
        )
        (group,) = groups
        self.assertEqual(group.actions, [0, 1, 1])
        self.assertAlmostEqual(group.advantages[0], -math.sqrt(2))
        self.assertAlmostEqual(group.advantages[1], 1 / math.sqrt(2))
        self.assertEqual(group.advantages[1], group.advantages[2])
        self.assertEqual(skipped, {"fewer_than_two_samples": 1, "zero_reward_variance": 1})
        with self.assertRaisesRegex(ValueError, "no groups contain a reward comparison"):
            prepare_groups([row("constant", [1., 1.])], 8, 8)
        small = row("low-variance", [0., 0.01])
        (scaled,), _ = prepare_groups([small], 8, 8, advantage_std_floor=0.05)
        self.assertAlmostEqual(scaled.advantages[0], -0.1)
        self.assertAlmostEqual(scaled.advantages[1], 0.1)

    def test_clipped_ratio_and_exact_kl_with_padding(self):
        logits = torch.tensor([[math.log(0.8), math.log(0.2), 0.]], requires_grad=True)
        options = torch.tensor([[True, True, False]])
        policy = logits.masked_fill(~options, -torch.inf).log_softmax(-1)
        behavior = torch.tensor([[-math.log(2), -math.log(2), 0.]])
        reference = torch.tensor([[-math.log(2), -math.log(2), -torch.inf]])
        losses, kl, clipping = group_losses(
            policy, behavior, reference, options, torch.tensor([[0, 1, 0]]),
            torch.tensor([[1., -1., 0.]]), torch.tensor([[True, True, False]]), 0.2, 0.2, 0.1,
        )
        expected_kl = 0.8 * math.log(1.6) + 0.2 * math.log(0.4)
        self.assertAlmostEqual(kl.item(), expected_kl)
        self.assertAlmostEqual(losses.item(), -0.2 + 0.1 * expected_kl)
        self.assertEqual(clipping.item(), 1.)
        losses.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[0, 2].item(), 0.)

    def test_recall_write_policy_weights_preserve_logged_draws(self):
        original = {
            "state_id": "example", "prompt_token_ids": [2], "option_token_ids": [4, 5],
            "behavior_logprobs": [-math.log(2)] * 2,
            "samples": [{"option_index": 0, "reward": 1.0}, {"option_index": 1, "reward": 0.0}],
        }
        rows = [dict(original, family=family) for family in ("noop", "recall", "compact", "write")]
        groups, skipped = prepare_groups(rows, 8, 8, recall_write_weight=1.5)
        self.assertEqual(skipped, {})
        self.assertEqual([group.policy_weight for group in groups], [1.0, 1.5, 1.0, 1.5])
        for group in groups:
            self.assertEqual(group.actions, [0, 1])
            self.assertEqual(group.behavior, original["behavior_logprobs"])
            self.assertEqual(group.advantages, groups[0].advantages)
        with self.assertRaisesRegex(ValueError, "identified scenario family"):
            prepare_groups([original], 8, 8, recall_write_weight=1.5)

    def test_rare_clipped_and_padded_samples_do_not_poison_gradients(self):
        logits = torch.zeros(1, 4, dtype=torch.float32, requires_grad=True)
        policy = logits.log_softmax(-1)
        behavior = torch.tensor([[-150., -130., -2., 0.]]).log_softmax(-1)
        active = torch.tensor([[True, True, False]])
        self.assertTrue(torch.isinf((policy[0, :2] - behavior[0, :2]).exp()).all())
        loss, _, clipping = group_losses(
            policy, behavior, policy.detach(), torch.ones_like(policy, dtype=torch.bool),
            torch.tensor([[0, 2, 1]]), torch.tensor([[1., -1., 0.]]), active, 0.2, 0.2, 0.,
        )
        # Only the rare positive advantage is capped. The negative advantage still uses its raw ratio.
        expected = ((policy[0, 2] - behavior[0, 2]).exp() - 1.2) / 2
        expected_gradient = torch.autograd.grad(expected, logits, retain_graph=True)[0]
        loss.sum().backward()
        torch.testing.assert_close(loss.squeeze(), expected)
        torch.testing.assert_close(logits.grad, expected_gradient)
        self.assertEqual(clipping.item(), 1.)

    def test_replay_policy_diagnostics_respect_padding_and_advantage_signs(self):
        policy = torch.tensor([[math.log(0.8), math.log(0.2), -math.inf]])
        behavior = torch.tensor([[-math.log(2), -math.log(2), 0.]])
        stats = replay_policy_diagnostics(
            policy, behavior, torch.tensor([[True, True, False]]), torch.tensor([[0, 1, 0]]),
            torch.tensor([[1., 1., 100.]]), torch.tensor([[True, True, False]]), 0.2, 0.2,
        )
        entropy = -0.8 * math.log(0.8) - 0.2 * math.log(0.2)
        torch.testing.assert_close(stats, torch.tensor([
            entropy, math.log(2) - entropy, 1 / (2 * (0.8 ** 2 + 0.2 ** 2)), 0.5, 1.,
        ]))

    def test_left_padding_uses_relative_positions_and_final_logits(self):
        def model(**kwargs):
            self.assertEqual(kwargs["logits_to_keep"], 1)
            self.assertEqual(kwargs["position_ids"].tolist(), [[0, 0, 1], [0, 1, 2]])
            return SimpleNamespace(logits=torch.tensor([[[5., 4., 3.]], [[8., 7., 6.]]]))

        prediction = option_logprobs(
            model, torch.tensor([[0, 1, 2], [1, 2, 1]]), torch.tensor([[0, 1, 1], [1, 1, 1]]),
            torch.tensor([[2, 1], [0, 0]]), torch.tensor([[True, True], [True, False]]),
        )
        self.assertTrue(torch.allclose(prediction[0], torch.tensor([3., 4.]).log_softmax(-1)))
        self.assertEqual(prediction[1].tolist(), [0., -math.inf])


if __name__ == "__main__":
    unittest.main()
