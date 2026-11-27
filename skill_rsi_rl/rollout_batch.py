"""Validate native rollout receipts and update the finite-action LoRA policy."""

import math
import hashlib
import random
from statistics import fmean
from types import SimpleNamespace

from .contracts import apply_action, build_messages, digest, option_token_ids, propose_options, render_prompt


def check_group(rollouts, policy_id, expected_size, *, expected=None, sampling_seeds=None):
    """Return sequence rewards; reject an entire group if one member is incomplete."""
    if len(rollouts) != expected_size or expected_size < 2:
        raise ValueError("incomplete rollout group")
    sequence = first_state = None
    rewards = []
    queries = set()
    seeds = [row["sampling_seed"] for row in rollouts]
    if (any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds)
            or sampling_seeds is not None and seeds != sampling_seeds):
        raise ValueError("duplicate or unexpected replica sampling streams")
    header = {key: rollouts[0][key] for key in ("task", "harness", "environment_seed", "bench_commit",
                                                 "run_index", "benchmark_identity")}
    header.update(status="completed", split="train", policy_id=policy_id, sample=True)
    header.update(expected or {})
    for row in rollouts:
        if row["sample"] is not True or any(row[k] != v for k, v in header.items()):
            raise ValueError("rollout does not match the frozen training group")
        decisions, outcomes = row["decisions"], row["outcomes"]
        native_memory = None
        if row.get("benchmark_kind") in {"past-bench", "workbuddy"}:
            from .harnesses.native_history import validate_history
            native_memory = validate_history(row)
        ids = [(o["instance_index"], o["instance_id"]) for o in outcomes]
        if (not ids or len(ids) != row["expected_instances"] or len(ids) != len(set(i for _, i in ids))
                or len(ids) != len(set(n for n, _ in ids))
                or any(type(n) is not int or n < 0 or not isinstance(i, str) or not i for n, i in ids)):
            raise ValueError("missing or duplicated official outcomes")
        instances = set(ids)
        if sequence is not None and ids != sequence:
            raise ValueError("rollouts have different instance schedules")
        sequence = ids
        if any(type(o["reward"]) not in (float, int) or not math.isfinite(o["reward"])
               or type(o["success"]) is not bool for o in outcomes):
            raise ValueError("invalid official outcome")
        rewards.append(fmean(o["reward"] for o in outcomes))
        rng = random.Random(row["sampling_seed"])
        previous_memory = None
        if not decisions:
            raise ValueError("rollout has no student decisions")
        initial = decisions[0]["state"]
        if first_state is not None and first_state != initial:
            raise ValueError("replicas did not start with the same visible student state")
        first_state = initial
        for step in decisions:
            query_id = step["query_id"]
            if not isinstance(query_id, str) or not query_id or query_id in queries:
                raise ValueError("missing or duplicate decision identity")
            queries.add(query_id)
            if (step["instance_index"], step["instance_id"]) not in instances or step["seed"] != rng.randrange(2**63):
                raise ValueError("decision belongs to a different instance or sampling stream")
            if step["policy_id"] != policy_id or step["applied"] is not True:
                raise ValueError("stale policy or unapplied decision")
            state, options = step["state"], step["options"]
            if native_memory is not None and state["memory"] != native_memory[query_id]:
                raise ValueError("student memory differs from the native episode boundary")
            if native_memory is None and previous_memory is not None and state["memory"] != previous_memory:
                raise ValueError("student memory does not continue the preceding decision")
            if options != propose_options(**state) or step["messages"] != build_messages(**state, options=options):
                raise ValueError("recorded student input differs from the pre-action state")
            index, logp = step["option_index"], step["behavior_logprobs"]
            if (type(index) is not int or not 0 <= index < len(options)
                    or step["action"] != options[index]["action"]
                    or len(logp) != len(options) or len(step["option_token_ids"]) != len(options)):
                raise ValueError("chosen action differs from the recorded support")
            if any(type(x) not in (int, float) or not math.isfinite(x) or x > 1e-6 for x in logp):
                raise ValueError("invalid log probabilities")
            if abs(math.fsum(math.exp(x) for x in logp) - 1) > 1e-5:
                raise ValueError("behavior distribution does not sum to one")
            updated, memo = apply_action(step["action"], state["memory"])
            previous_memory = updated
            receipt = step["receipt"]
            evidence = dict(checkpoint=policy_id, status="applied", op=step["action"]["op"],
                            input_sha256=digest(state), memory_before_sha256=digest(state["memory"]),
                            memory_after_sha256=digest(updated), injected=bool(memo),
                            memo_sha256=hashlib.sha256(memo.encode()).hexdigest() if memo else None)
            if any(receipt.get(k) != v for k, v in evidence.items()):
                raise ValueError("application receipt differs from the sampled state or action")
        if not any(len(d["options"]) > 1 for d in decisions):
            raise ValueError("rollout contains only forced actions")
    if max(rewards) == min(rewards):
        raise ValueError("all rollout rewards in group are equal")
    return rewards


def update_policy(config, model_path, parent, previous_optimizer, groups, policy_id, round_id, destination, settings):
    """Accumulate one rollout-weighted EasyR1 update, then atomically save weights and Adam state."""
    import torch

    from .easyr1_backend import _upstream
    from .train import load_model, option_logprobs
    from .training_artifacts import save_checkpoint

    _, core, _, _ = _upstream()
    device = torch.device(config["device"])
    torch.manual_seed(config["seed"] + round_id)
    loader = SimpleNamespace(model=model_path, dtype="auto", resume_from=None, init_adapter=parent,
                             lora_rank=config["lora_rank"], gradient_checkpointing=False)
    model, tokenizer, _ = load_model(loader, SimpleNamespace(device=device))
    # Keep sampling and training forward passes deterministic even for warm-start adapters with dropout.
    model.eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"], weight_decay=0.0)
    if previous_optimizer is not None:
        optimizer.load_state_dict(previous_optimizer)
    optimizer.zero_grad(set_to_none=True)
    total_rollouts = sum(map(len, groups))
    stats = dict(loss=0.0, reference_kl=0.0, max_logp_error=0.0, decisions=0, masked_decisions=0,
                 groups=len(groups), rollouts=total_rollouts, logp_tolerance=config["logp_tolerance"],
                 dtype=str(next(model.get_base_model().parameters()).dtype))
    for group in groups:
        scores = torch.tensor(check_group(group, policy_id, config["group_size"]), dtype=torch.float32)[:, None]
        # EasyR1 groups by Python dictionary key. Repeated Python integers share a key.
        advantages, _ = core.compute_grpo_outcome_advantage(scores, torch.ones_like(scores), [0] * len(group))
        for row, advantage in zip(group, advantages.flatten().tolist(), strict=True):
            steps = [d for d in row["decisions"] if len(d["options"]) > 1]
            stats["masked_decisions"] += len(row["decisions"]) - len(steps)
            weight = 1.0 / (total_rollouts * len(steps))
            for step in steps:
                prompts = render_prompt(tokenizer, step["messages"])
                labels = option_token_ids(tokenizer, [o["label"] for o in step["options"]])
                if prompts != step["prompt_token_ids"] or labels != step["option_token_ids"]:
                    raise ValueError("student tokenizer does not reproduce the sampled decision")
                ids = torch.tensor([prompts], device=device)
                options = torch.tensor([labels], device=device)
                args = (ids, torch.ones_like(ids), options, torch.ones_like(options, dtype=torch.bool))
                with torch.no_grad(), model.disable_adapter():
                    ref = option_logprobs(model, *args)
                pred = option_logprobs(model, *args)
                old = pred.new_tensor([step["behavior_logprobs"]])
                error = (pred.detach() - old).abs().max().item()
                if not math.isfinite(error) or error > config["logp_tolerance"]:
                    raise ValueError(f"frozen policy logp mismatch: {error} > {config['logp_tolerance']}")
                stats["max_logp_error"] = max(stats["max_logp_error"], error)
                i = step["option_index"]
                selected = pred[:, i:i + 1]
                loss, _ = core.compute_policy_loss(
                    old_log_probs=old[:, i:i + 1], log_probs=selected,
                    advantages=torch.full_like(selected, advantage), response_mask=torch.ones_like(selected),
                    clip_ratio_low=config["clip_ratio"], clip_ratio_high=config["clip_ratio"],
                    clip_ratio_dual=2.0, tau_positive=1.0, tau_negative=1.0,
                    loss_type="default", loss_avg_mode="seq",
                )
                kl = core.compute_kl(pred, ref, "full").mean()
                objective = loss + config["beta"] * kl
                if not torch.isfinite(objective):
                    raise ValueError("non-finite online objective")
                (objective * weight).backward()
                stats["loss"] += weight * objective.detach().item()
                stats["reference_kl"] += weight * kl.detach().item()
                stats["decisions"] += 1
    norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
    if norm.item() == 0:
        raise ValueError("no gradient for this online batch")
    previous = [p.detach().cpu().clone() for p in trainable]
    optimizer.step()
    delta = max((p.detach().cpu() - before).abs().max().item() for p, before in zip(trainable, previous, strict=True))
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("optimizer failed to produce a finite LoRA update")
    stats.update(grad_norm=norm.item(), max_parameter_change=delta, optimizer_step=round_id)
    save_checkpoint(destination, model, optimizer, {**settings, "metrics": stats}, round_id, round_id)
    return stats
