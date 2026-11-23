"""Shared inputs and model operations for finite-action training extensions."""

from types import SimpleNamespace

from ...contracts import option_token_ids, render_prompt
from ...replay_io import open_hashed_replay
from ...verify_replay import _score_coverage, _validate_support

__all__ = ["frozen_sample_scores", "load_lora_policy", "option_logprobs", "read_training_replay",
           "validate_replay_tokens"]


def read_training_replay(path):
    """Read one complete training snapshot, without filtering by algorithm."""
    with open_hashed_replay(path) as (manifest, rows, digest):
        if manifest["split"] != "train":
            raise ValueError("training requires a replay with split=train")
        if manifest.get("fixture_only"):
            raise ValueError("fixture-only replay cannot train model weights")
        if manifest.get("shard", {"rank": 0, "world_size": 1}) != {"rank": 0, "world_size": 1}:
            raise ValueError("merge all replay shards before training")
        behavior_id = manifest.get("behavior_checkpoint_id")
        if not isinstance(behavior_id, str) or not behavior_id.strip():
            raise ValueError("training replay requires an immutable behavior_checkpoint_id")
        groups = list(rows)
        if manifest.get("state_count", len(groups)) != len(groups):
            raise ValueError("replay group count differs from the collection manifest")
        for group in groups:
            _validate_support(group)
        return manifest, groups, digest()


def frozen_sample_scores(group):
    """Validate fixed candidate scores and return observed index -> (reward, success)."""
    return _score_coverage(group, "train")


def validate_replay_tokens(row, tokenizer, *, vocab_size, max_prompt_tokens):
    """Check that the current model can reproduce a logged decision exactly."""
    state = row["state_id"]
    prompt, options = row["prompt_token_ids"], row["option_token_ids"]
    if not 0 < len(prompt) <= max_prompt_tokens:
        raise ValueError(f"{state}: prompt length {len(prompt)} outside 1..{max_prompt_tokens}")
    if any(token < 0 or token >= vocab_size for token in (*prompt, *options)):
        raise ValueError(f"{state}: token ID is outside the model vocabulary")
    labels = [option["label"] for option in row["options"]]
    if prompt != render_prompt(tokenizer, row["messages"]) or options != option_token_ids(tokenizer, labels):
        raise ValueError(f"{state}: replay tokenization differs from the model tokenizer")


def load_lora_policy(model_path, device, *, lora_rank, gradient_checkpointing=False):
    """Create a fresh trainable policy using the project's local model loader."""
    from ...train import load_model

    config = SimpleNamespace(model=model_path, dtype="auto", init_adapter=None, resume_from=None,
                             lora_rank=lora_rank, gradient_checkpointing=gradient_checkpointing)
    return load_model(config, SimpleNamespace(device=device))


def option_logprobs(model, input_ids, attention_mask, option_ids, option_mask):
    """Normalize next-token log probabilities over the offered action labels."""
    from ...train import option_logprobs as score_options

    return score_options(model, input_ids, attention_mask, option_ids, option_mask)
