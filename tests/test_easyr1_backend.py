from functools import partial
from hashlib import sha256
from importlib.util import find_spec
import math
from pathlib import Path
from unittest import TestCase, skipUnless
from unittest.mock import patch

import torch

from skill_rsi_rl import easyr1_backend
from skill_rsi_rl.train import Group, group_losses as native_group_losses, make_batch


@skipUnless(find_spec("verl") is not None, "pinned EasyR1 checkout unavailable")
class EasyR1BackendTests(TestCase):
    def test_data_proto_keeps_state_groups_and_behavior_distribution(self):
        groups = [Group([1, 2], [3, 4], [-0.4, -1.1096], [1, 0, 1], [-1.0, 0.0, 1.0]),
                  Group([2], [3, 4, 5], [-1.0, -2.0, -0.6996], [0, 2], [1.0, -1.0])]
        batch = make_batch(groups, 0, torch.device("cpu"))
        data = easyr1_backend.pack_batch(batch)
        self.assertEqual(data.meta_info["batch_unit"], "replay_state")
        self.assertEqual(list(data.batch.batch_size), [2])
        for actual, expected in zip(easyr1_backend.roundtrip_batch(batch), batch, strict=True):
            torch.testing.assert_close(actual, expected)
        provenance = easyr1_backend.backend_provenance()
        self.assertEqual(provenance["commit"], easyr1_backend.EASYR1_COMMIT)
        self.assertEqual(provenance["adapter_sha256"], sha256(Path(easyr1_backend.__file__).read_bytes()).hexdigest())

    def test_upstream_objective_and_gradient_match_native_group_weighting(self):
        from verl.trainer import core_algos

        support = torch.tensor([[True, True, True], [True, True, False]])
        sampled = torch.tensor([[True, True, True], [True, True, False]])
        actions = torch.tensor([[0, 0, 1], [1, 0, 0]])
        advantages = torch.tensor([[0.2, -0.5, 0.3], [1.0, -1.0, 0.0]])
        behavior = torch.tensor([[0.3, -0.4, 0.1], [0.1, 0.5, 0.0]]).masked_fill(~support, -torch.inf).log_softmax(-1)
        reference = torch.tensor([[-0.4, 0.5, 0.2], [0.6, 0.2, 0.0]]).masked_fill(~support, -torch.inf).log_softmax(-1)
        inputs = torch.tensor([[0.7, -0.4, 0.8], [0.2, 0.4, 0.0]])

        def evaluate(loss_function, behavior):
            raw = inputs.clone().requires_grad_()
            policy = raw.masked_fill(~support, -torch.inf).log_softmax(-1)
            outputs = loss_function(policy, behavior, reference, support, actions, advantages, sampled, 0.2, 0.15, 0.05)
            outputs[0].sum().backward()
            return (*outputs, raw.grad)

        with patch.object(core_algos, "compute_policy_loss", wraps=core_algos.compute_policy_loss) as upstream:
            easy = evaluate(easyr1_backend.group_losses, behavior)
            self.assertEqual(upstream.call_count, 2)
        native = evaluate(native_group_losses, behavior)
        for actual, expected in zip(easy, native, strict=True):
            torch.testing.assert_close(actual, expected, atol=3e-7, rtol=1e-6)

        extreme = behavior.clone()
        extreme[0] = torch.tensor([-26.0, 0.0, 1.0]).log_softmax(0)
        with patch.object(core_algos, "compute_policy_loss", wraps=core_algos.compute_policy_loss) as upstream:
            easy_tail = evaluate(easyr1_backend.group_losses, extreme)
            self.assertEqual(upstream.call_count, 1)
        native_tail = evaluate(native_group_losses, extreme)
        for actual, expected in zip(easy_tail, native_tail, strict=True):
            torch.testing.assert_close(actual, expected, atol=3e-7, rtol=1e-6)

        summed_losses = partial(easyr1_backend.group_losses, sum_groups=True)
        for supplied, expected, num_regular in ((behavior, native, 2), (extreme, native_tail, 1)):
            with patch.object(core_algos, "compute_policy_loss", wraps=core_algos.compute_policy_loss) as upstream:
                summed = evaluate(summed_losses, supplied)
                self.assertEqual(upstream.call_count, 1)
                self.assertEqual(upstream.call_args.kwargs["loss_avg_mode"], "seq")
                regular = upstream.call_args.kwargs["response_mask"]
                self.assertEqual(regular.any(-1).sum().item(), num_regular)
            torch.testing.assert_close(summed[0], expected[0].sum(), atol=3e-7, rtol=1e-6)
            for actual, wanted in zip(summed[1:], expected[1:], strict=True):
                torch.testing.assert_close(actual, wanted, atol=3e-7, rtol=1e-6)

    def test_float32_policy_gradient_preserved_at_upper_log_ratio_boundary(self):
        from verl.trainer import core_algos

        reference = torch.tensor([[-1.0, math.log1p(-math.exp(-1.0))]], dtype=torch.float32)
        behavior = torch.tensor([[-21.0, math.log1p(-math.exp(-21.0))]], dtype=torch.float32)
        actions = torch.tensor([[0, 1]])
        advantages = torch.tensor([[-1.0, 1.0]])
        valid = torch.tensor([[True, True]])

        def gradient(loss_function):
            policy = reference.clone().requires_grad_()
            self.assertEqual((policy[0, 0] - behavior[0, 0]).item(), 20.0)
            losses, _, _ = loss_function(policy, behavior, reference, valid, actions, advantages, valid, 0.2, 0.2, 0.0)
            losses.sum().backward()
            return policy.grad

        with patch.object(core_algos, "compute_policy_loss", wraps=core_algos.compute_policy_loss) as upstream:
            result = gradient(easyr1_backend.group_losses)
            upstream.assert_not_called()
        torch.testing.assert_close(result, gradient(native_group_losses), atol=3e-7, rtol=1e-6)

    def test_unrepresentable_clipped_ratio_with_padding_uses_exact_tail(self):
        from verl.trainer import core_algos

        logits = torch.zeros((1, 4), dtype=torch.float32, requires_grad=True)
        policy = logits.log_softmax(-1)
        behavior = torch.tensor([[-150., -130., -2., 0.]]).log_softmax(-1)
        support = torch.ones_like(policy, dtype=torch.bool)
        actions = torch.tensor([[0, 2, 1]])
        advantages = torch.tensor([[1., -1., 0.]])
        sampled = torch.tensor([[True, True, False]])
        with patch.object(core_algos, "compute_policy_loss", wraps=core_algos.compute_policy_loss) as upstream:
            loss, _, clipped_fraction = easyr1_backend.group_losses(
                policy, behavior, policy.detach(), support, actions, advantages, sampled, 0.2, 0.2, 0.,
            )
            upstream.assert_not_called()
        expected = ((policy[0, 2] - behavior[0, 2]).exp() - 1.2) / 2
        expected_gradient = torch.autograd.grad(expected, logits, retain_graph=True)[0]
        loss.sum().backward()
        torch.testing.assert_close(loss.squeeze(), expected)
        torch.testing.assert_close(logits.grad, expected_gradient)
        self.assertEqual(clipped_fraction.item(), 1.)

        inputs = logits.detach().clone().requires_grad_()
        summed_policy = inputs.log_softmax(-1)
        summed_loss, summed_kl, summed_fraction = easyr1_backend.group_losses(
            summed_policy, behavior, summed_policy.detach(), support, actions, advantages, sampled,
            0.2, 0.2, 0., sum_groups=True,
        )
        summed_loss.backward()
        torch.testing.assert_close(summed_loss, expected)
        torch.testing.assert_close(inputs.grad, expected_gradient)
        torch.testing.assert_close(summed_kl, torch.zeros_like(summed_kl))
        torch.testing.assert_close(summed_fraction, clipped_fraction)
