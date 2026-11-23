"""Fit a single-device LoRA to the uniquely highest-scoring action per replay state."""

import argparse
import math
import os
from pathlib import Path
import random

from skill_rsi_rl.algorithms import api
from skill_rsi_rl.contracts import MODEL_PATH
from skill_rsi_rl.serve import checkpoint_identity


ALGORITHM = Path(__file__).parent.name


def best_action(group):
    sampled = api.frozen_sample_scores(group)
    if group.get("option_scores") is not None:
        rewards = [score["reward"] for score in group["option_scores"]]
    else:
        if len(sampled) != len(group["options"]):
            return None
        rewards = [sampled[i][0] for i in range(len(group["options"]))]
    highest = max(rewards)
    return rewards.index(highest) if rewards.count(highest) == 1 else None


def training_examples(groups, tokenizer, max_tokens, vocab_size):
    examples = []
    skipped = 0
    for group in groups:
        api.validate_replay_tokens(group, tokenizer, vocab_size=vocab_size, max_prompt_tokens=max_tokens)
        target = best_action(group)
        if target is None:
            skipped += 1
        else:
            examples.append((group["prompt_token_ids"], group["option_token_ids"], target))
    if not examples:
        raise ValueError("no states have full score coverage and a unique highest reward")
    return examples, skipped


def example_loss(model, prompt, options, target, device):
    import torch

    ids = torch.tensor([prompt], device=device)
    candidates = torch.tensor([options], device=device)
    logprobs = api.option_logprobs(model, ids, torch.ones_like(ids), candidates,
                                   torch.ones_like(candidates, dtype=torch.bool))
    return -logprobs[0, target]


def train(args):
    import torch
    from skill_rsi_rl.training_artifacts import (
        TrainingProgress, append_record, initialize_run, lock_output, save_checkpoint,
    )

    if (int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("RANK", "0")) != 0
            or (torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1)):
        raise ValueError(f"{ALGORITHM} uses one process. Run with python and a single visible GPU.")
    if min(args.epochs, args.lora_rank, args.max_prompt_tokens) < 1:
        raise ValueError("epochs, LoRA rank and token limit must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning rate must be positive and finite")

    manifest, groups, replay_sha = api.read_training_replay(args.replay)
    behavior_id = manifest["behavior_checkpoint_id"]
    base_id = checkpoint_identity(args.model)
    if (Path(manifest["behavior_model"]).expanduser().resolve() == args.model.expanduser().resolve()
            and manifest.get("behavior_base_checkpoint_id", behavior_id) != base_id):
        raise ValueError("training base differs from the recorded behavior base")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    device = torch.device(device_name)
    torch.manual_seed(args.seed)
    model, tokenizer, _ = api.load_lora_policy(args.model, device, lora_rank=args.lora_rank)
    examples, skipped = training_examples(groups, tokenizer,
                                          min(args.max_prompt_tokens, model.config.max_position_embeddings),
                                          model.config.vocab_size)

    settings = dict(algorithm=ALGORITHM, algorithm_version=1, replay_sha256=replay_sha,
                    base_checkpoint_id=base_id, dataset_id=manifest["dataset_id"],
                    behavior_checkpoint_id=behavior_id, learning_rate=args.learning_rate,
                    seed=args.seed, epochs=args.epochs, lora_rank=args.lora_rank,
                    max_prompt_tokens=args.max_prompt_tokens)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    model.train()
    with lock_output(args.output):
        if any(path.name != ".train.lock" for path in args.output.iterdir()):
            raise ValueError(f"choose an empty output directory for each {ALGORITHM} run")
        run = {"settings": settings, "training_states": len(examples), "skipped_states": skipped}
        metrics = initialize_run(args.output, run, TrainingProgress(), args.epochs)
        step = 0
        for epoch in range(1, args.epochs + 1):
            order = list(range(len(examples)))
            random.Random(args.seed + epoch).shuffle(order)
            loss_sum = 0.0
            for index in order:
                optimizer.zero_grad(set_to_none=True)
                loss = example_loss(model, *examples[index], device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                loss_sum += loss.detach().item()
                step += 1
            save_checkpoint(args.output / f"checkpoint-{epoch:04d}", model, optimizer, settings, epoch, step)
            append_record(metrics, {"epoch": epoch, "step": step, "mean_loss": loss_sum / len(examples)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--output", type=Path, required=True, help="new or empty run directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    args = parser.parse_args()
    try:
        train(args)
    except (OSError, ValueError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
