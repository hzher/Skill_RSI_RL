"""Train single-process, finite-action DPO from immutable student replay."""

import argparse
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace

from skill_rsi_rl.algorithms import api
from skill_rsi_rl.contracts import MODEL_PATH
from skill_rsi_rl.serve import checkpoint_identities
from .objective import dpo_loss, preference_pairs, score


def single_process():
    import torch

    if (int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("RANK", "0")) != 0
            or torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1):
        raise ValueError("DPO currently supports one process on CPU or one GPU")


def load_policy(model, adapter, device, lora_rank):
    from skill_rsi_rl.train import load_model

    return load_model(SimpleNamespace(model=model, dtype="auto", init_adapter=adapter, resume_from=None,
                                     lora_rank=lora_rank, gradient_checkpointing=False),
                      SimpleNamespace(device=device))


def train(args):
    import torch
    from skill_rsi_rl.training_artifacts import (
        TrainingProgress, append_record, initialize_run, lock_output, save_checkpoint,
    )

    single_process()
    if min(args.epochs, args.lora_rank, args.max_prompt_tokens) < 1:
        raise ValueError("epochs, LoRA rank and token limit must be positive")
    for name in ("learning_rate", "beta"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.min_reward_gap) or args.min_reward_gap < 0:
        raise ValueError("minimum reward gap must be finite and nonnegative")
    manifest, groups, replay_sha = api.read_training_replay(args.replay)
    # Filter before loading model weights. Missing coverage is not invented.
    selected = [(group, preference_pairs(group, args.min_reward_gap)) for group in groups]
    examples = [(group, pairs) for group, pairs in selected if pairs]
    if not examples:
        raise ValueError("no strictly preferred pair of sampled actions")
    base_id, initial_id = checkpoint_identities(args.model, args.init_adapter)
    if (Path(manifest["behavior_model"]).expanduser().resolve() == args.model.expanduser().resolve()
            and manifest.get("behavior_base_checkpoint_id", manifest["behavior_checkpoint_id"]) != base_id):
        raise ValueError("training base differs from recorded behavior base")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    settings = dict(algorithm="dpo", algorithm_version=1, mode="offline", beta=args.beta,
                    min_reward_gap=args.min_reward_gap, replay_sha256=replay_sha,
                    base_checkpoint_id=base_id, reference_policy_id=base_id, initial_policy_id=initial_id,
                    dataset_id=manifest["dataset_id"], behavior_checkpoint_id=manifest["behavior_checkpoint_id"],
                    learning_rate=args.learning_rate, seed=args.seed, epochs=args.epochs,
                    lora_rank=args.lora_rank, max_prompt_tokens=args.max_prompt_tokens,
                    preference_source="sampled_actions_with_frozen_rewards")
    with lock_output(args.output):
        if any(p.name != ".train.lock" for p in args.output.iterdir()):
            raise ValueError("choose an empty output directory for each DPO run")
        model, tokenizer, _ = load_policy(args.model, args.init_adapter, device, args.lora_rank)
        for group, _ in selected:
            api.validate_replay_tokens(group, tokenizer, vocab_size=model.config.vocab_size,
                                      max_prompt_tokens=min(args.max_prompt_tokens, model.config.max_position_embeddings))
        # Deterministic dropout-free scoring matches the student's sampling path.
        model.eval()
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
        metrics = initialize_run(args.output, {"settings": settings, "training_states": len(examples),
                                  "skipped_states": len(selected) - len(examples),
                                  "preference_pairs": sum(len(pairs) for _, pairs in examples)},
                                 TrainingProgress(), args.epochs)
        step = 0
        for epoch in range(1, args.epochs + 1):
            order = list(range(len(examples)))
            random.Random(args.seed + epoch).shuffle(order)
            total, correct, count = 0.0, 0, 0
            for index in order:
                group, pairs = examples[index]
                prompt, options = group["prompt_token_ids"], group["option_token_ids"]
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad(), model.disable_adapter():
                    reference = score(model, prompt, options, device)
                policy = score(model, prompt, options, device)
                loss = torch.stack([dpo_loss(policy, reference, w, l, args.beta) for w, l in pairs]).mean()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                if norm.item() == 0:
                    raise ValueError("DPO preference batch produced no gradient")
                optimizer.step()
                total += loss.detach().item()
                correct += sum((policy[0, w] - policy[0, l] > reference[0, w] - reference[0, l]).item()
                               for w, l in pairs)
                count += len(pairs)
                step += 1
            save_checkpoint(args.output / f"checkpoint-{epoch:04d}", model, optimizer, settings, epoch, step)
            append_record(metrics, {"epoch": epoch, "step": step, "loss": total / len(examples),
                                    "preference_accuracy_before_update": correct / count, "pairs": count})
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--min-reward-gap", type=float, default=0.0)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=17)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
