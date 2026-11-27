from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch

from skill_rsi_rl.contracts import apply_action, build_messages, digest, propose_options
from . import online


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(4))
        self.config = SimpleNamespace(vocab_size=4, max_position_embeddings=16)
        self.disabled = False

    @contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False

    def forward(self, **kwargs):
        return SimpleNamespace(logits=(self.logits * (0 if self.disabled else 1)).view(1, 1, -1))


def native_group(policy_id="frozen"):
    state = {"query": "latency unit?", "feedback": [], "memory": {"unit": "seconds"}}
    options = propose_options(**state)
    result = []
    for replica, index in enumerate((0, 1)):
        seed = 10 + replica
        action = options[index]["action"]
        updated, memo = apply_action(action, state["memory"])
        receipt = dict(checkpoint=policy_id, status="applied", op=action["op"],
                       input_sha256=digest(state), memory_before_sha256=digest(state["memory"]),
                       memory_after_sha256=digest(updated), injected=bool(memo),
                       memo_sha256=hashlib.sha256(memo.encode()).hexdigest() if memo else None)
        decision = dict(query_id=f"query-{replica}", policy_id=policy_id, applied=True,
                        instance_id="instance", instance_index=0, seed=random.Random(seed).randrange(2**63),
                        state=state, options=options, messages=build_messages(**state, options=options),
                        option_index=index, action=action, receipt=receipt, prompt_token_ids=[3],
                        option_token_ids=list(range(len(options))),
                        behavior_logprobs=[-math.log(len(options))] * len(options))
        result.append(dict(task="synthetic", harness="codex", environment_seed=17, sampling_seed=seed,
                           bench_commit="synthetic", run_index=0, benchmark_identity="synthetic",
                           status="completed", split="train", policy_id=policy_id, sample=True,
                           expected_instances=1, decisions=[decision],
                           outcomes=[dict(instance_id="instance", instance_index=0, reward=replica, success=bool(replica))]))
    return result


class OnlineTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyPolicy()
        self.config = dict(device="cpu", seed=17, lora_rank=1, group_size=2, min_reward_gap=0,
                           max_prompt_tokens=16, logp_tolerance=0.002, learning_rate=0.1, beta=0.2)
        self.tokenizer = SimpleNamespace(all_special_ids=[], apply_chat_template=lambda *args, **kwargs: [3],
                                         encode=lambda label, **kwargs: ["ABC".index(label)])

    def test_real_receipts_backward_update_and_optimizer_continuation(self):
        group = native_group()
        with TemporaryDirectory() as tmp, patch.object(online, "load_policy", return_value=(self.model, self.tokenizer, 3)), \
                patch("skill_rsi_rl.training_artifacts.save_checkpoint") as save:
            first = online.update(self.config, None, None, None, [group], "frozen", 1, Path(tmp), {})
            self.assertEqual(first["preference_pairs"], 1)
            self.assertEqual(first["max_logp_error"], 0)
            self.assertGreater(first["max_parameter_change"], 0)
            self.assertGreater(self.model.logits[1].item(), self.model.logits[0].item())
            optimizer = save.call_args.args[2].state_dict()
            refreshed = copy.deepcopy(group)
            for row in refreshed:
                row["decisions"][0]["behavior_logprobs"] = self.model.logits[:3].log_softmax(-1).detach().tolist()
            online.update(self.config, None, None, optimizer, [refreshed], "frozen", 2, Path(tmp), {})
            self.assertEqual(save.call_args.args[-1], 2)
            self.assertEqual(next(iter(save.call_args.args[2].state.values()))["step"].item(), 2)

    def test_stale_logprobs_fail_before_optimizer_or_checkpoint(self):
        group = native_group()
        self.model.logits.data[0] = 1
        before = self.model.logits.detach().clone()
        with patch.object(online, "load_policy", return_value=(self.model, self.tokenizer, 3)), \
                patch("skill_rsi_rl.training_artifacts.save_checkpoint") as save:
            with self.assertRaisesRegex(ValueError, "logp mismatch"):
                online.update(self.config, None, None, None, [group], "frozen", 1, None, {})
            save.assert_not_called()
        torch.testing.assert_close(before, self.model.logits)

    def test_unapplied_or_old_policy_receipt_rejected(self):
        for field, value in (("applied", False), ("policy_id", "old")):
            group = native_group()
            group[0]["decisions"][0][field] = value
            with self.assertRaisesRegex(ValueError, "unapplied|stale"):
                online.update(self.config, None, None, None, [group], "frozen", 1, None, {})

    def test_round_publication_loads_new_parent_and_restart_never_updates_twice(self):
        from skill_rsi_rl.training_artifacts import save_checkpoint

        parents, updates = [], []

        @contextmanager
        def service(config, model, adapter, policy_id):
            parents.append((adapter, policy_id))
            yield "http://synthetic"

        def collect(config, directory, number, policy_id, *args):
            (directory / "batch.json").write_text(json.dumps({"parent": policy_id, "round": number}))

        def update(config, model, parent, optimizer, groups, policy_id, number, destination, settings):
            if number == 2:
                self.assertEqual(optimizer, {"synthetic_step": 1})
            updates.append(number)

            class Artifact:
                def save_pretrained(self, path):
                    (path / "adapter_config.json").write_text("{}")
                    (path / "adapter_model.safetensors").write_bytes(f"synthetic-{number}".encode())

            save_checkpoint(destination, Artifact(), SimpleNamespace(state_dict=lambda: {"synthetic_step": number}),
                            {**settings, "metrics": {"optimizer_step": number}}, number, number)

        with TemporaryDirectory() as tmp, \
                patch.object(online, "policy_service", service), patch.object(online, "collect", collect), \
                patch.object(online, "load_batch", return_value=[]), patch.object(online, "update", update), \
                patch("skill_rsi_rl.serve.checkpoint_identities", return_value=("base", "base")), \
                patch("skill_rsi_rl.serve.checkpoint_identity", side_effect=lambda model, adapter: adapter.parent.name), \
                patch("torch.cuda.is_available", return_value=False):
            output = Path(tmp) / "run"
            config = {"device": "cpu", "benchmark": {"harness": "codex"}}
            online.run(config, Path(tmp), None, output, 2, Path(sys.executable))
            self.assertEqual([p[1] for p in parents], ["base", "round-0001"])
            self.assertEqual(updates, [1, 2])
            records = {p: p.read_bytes() for p in output.rglob("result.json")}
            online.run(config, Path(tmp), None, output, 2, Path(sys.executable))
            self.assertEqual(updates, [1, 2])
            self.assertEqual(len(parents), 2)
            self.assertEqual(records, {p: p.read_bytes() for p in output.rglob("result.json")})


if __name__ == "__main__":
    unittest.main()
