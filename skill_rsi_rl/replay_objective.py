"""Clipped importance weighting for fixed replay."""

import torch


def clipping_ratios(log_ratio, sample_mask, clip_low, clip_high):
    # Keep the clipping indicator on the original ratio, including valid extreme draws.
    ratio = log_ratio.detach().masked_fill(~sample_mask, 0).exp()
    return ratio, ratio.clamp(1.0 - clip_low, 1.0 + clip_high)


def exact_clipped_policy_loss(log_ratio, advantages, sample_mask, ratio, clipped):
    # For positive advantages the high bound can win the minimum, and for negative
    # advantages the low bound can win. Only the selected raw terms need a gradient.
    use_clipped = ((advantages > 0) & (ratio > clipped)) | ((advantages < 0) & (ratio < clipped))
    use_raw = sample_mask & (advantages != 0) & ~use_clipped
    raw_ratio = log_ratio.masked_fill(~use_raw, 0).exp()
    selected_ratio = torch.where(use_clipped, clipped, raw_ratio)
    surrogate = (selected_ratio * advantages).masked_fill(~sample_mask, 0)
    return -surrogate.sum(-1) / sample_mask.sum(-1)
