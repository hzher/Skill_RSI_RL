"""Preference construction and the reference-relative DPO objective."""

import math

from skill_rsi_rl.algorithms import api
from skill_rsi_rl.contracts import digest


def preference_pairs(group, min_gap=0.0):
    """Use only actions genuinely sampled by the recorded student, once per pair."""
    if not math.isfinite(min_gap) or min_gap < 0:
        raise ValueError("minimum reward gap must be finite and nonnegative")
    scores = api.frozen_sample_scores(group)
    return [(winner, loser) for winner in sorted(scores) for loser in sorted(scores)
            if scores[winner][0] - scores[loser][0] > min_gap]


def dpo_loss(policy, reference, winner, loser, beta):
    import torch
    from torch.nn import functional as F

    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("DPO beta must be finite and positive")
    margin = beta * ((policy[..., winner] - policy[..., loser])
                     - (reference[..., winner] - reference[..., loser]).detach())
    if not torch.isfinite(margin).all():
        raise ValueError("non-finite DPO margin")
    return -F.logsigmoid(margin).mean()


def score(model, prompt, options, device):
    import torch

    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    candidates = torch.tensor([options], dtype=torch.long, device=device)
    return api.option_logprobs(model, ids, torch.ones_like(ids), candidates,
                              torch.ones_like(candidates, dtype=torch.bool))


def online_preferences(rollouts, rewards, min_gap=0.0):
    """Compare alternatives at identical pre-action states across independent replicas.

    Sequence reward labels are used only when two replicas reach the same decision
    input. States after memory or feedback diverges are never paired together.
    """
    if len(rollouts) != len(rewards) or not math.isfinite(min_gap) or min_gap < 0:
        raise ValueError("invalid online reward labels or minimum gap")
    buckets = {}
    for replica, (row, reward) in enumerate(zip(rollouts, rewards, strict=True)):
        if not math.isfinite(reward):
            raise ValueError("non-finite sequence reward")
        for position, step in enumerate(row["decisions"]):
            if len(step["options"]) < 2:
                continue
            key = digest([position, step["instance_id"], step["instance_index"], step["state"],
                          step["messages"], step["options"], step["prompt_token_ids"], step["option_token_ids"]])
            buckets.setdefault(key, []).append((replica, reward, step))
    pairs = []
    for rows in buckets.values():
        for replica, reward, winner in rows:
            for other, lower, loser in rows:
                if (replica != other and reward - lower > min_gap
                        and winner["option_index"] != loser["option_index"]):
                    pairs.append((winner, loser))
    return pairs
