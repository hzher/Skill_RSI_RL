"""Use pinned EasyR1 batch and policy-loss primitives with immutable replay groups."""

from functools import lru_cache
from hashlib import sha256
import math
from pathlib import Path
import sys

from .sources import verify_source


EASYR1_COMMIT = "a73346d028ba0335aa42da104972c86e6cd0fe69"
_LOG_RATIO_LIMIT = 20.0
_ADAPTER_SHA256 = sha256(Path(__file__).read_bytes()).hexdigest()
_BATCH_KEYS = (
    "input_ids", "attention_mask", "option_ids", "option_mask", "behavior_log_probs",
    "sampled_actions", "advantages", "sample_mask",
)


@lru_cache(maxsize=1)
def _upstream():
    checkout, commit = verify_source("easyr1")
    if commit != EASYR1_COMMIT:
        raise RuntimeError(f"EasyR1 commit must be {EASYR1_COMMIT}; found {commit}")
    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))
    try:
        from verl.protocol import DataProto
        from verl.trainer import core_algos
    except ImportError as exc:
        raise RuntimeError("The EasyR1 backend needs the pinned EasyR1 checkout and its Python dependencies") from exc

    from verl import protocol

    paths = [Path(module.__file__).resolve() for module in (protocol, core_algos, core_algos.VF)]
    expected = ["verl/protocol.py", "verl/trainer/core_algos.py", "verl/utils/torch_functional.py"]
    if paths != [checkout / name for name in expected] or DataProto is not protocol.DataProto:
        raise RuntimeError("EasyR1 primitives must be imported from the verified project source")
    sources = {path.relative_to(checkout).as_posix(): sha256(path.read_bytes()).hexdigest() for path in paths}
    return DataProto, core_algos, checkout, sources


def backend_provenance():
    """Return the exact upstream primitives active in this Python process."""
    _, _, checkout, sources = _upstream()
    return {"name": "easyr1", "commit": EASYR1_COMMIT, "checkout": str(checkout),
            "batch": "verl.protocol.DataProto", "policy_loss": "verl.trainer.core_algos.compute_policy_loss",
            "kl": "verl.trainer.core_algos.compute_kl", "sources_sha256": dict(sources),
            "adapter_sha256": _ADAPTER_SHA256,
            "objective_sha256": sha256(Path(__file__).with_name("replay_objective.py").read_bytes()).hexdigest()}


def pack_batch(batch):
    """Keep each padded replay state and all its sampled actions on one DataProto row."""
    DataProto, _, _, _ = _upstream()
    return DataProto.from_dict(tensors=dict(zip(_BATCH_KEYS, batch, strict=True)),
                               meta_info={"batch_unit": "replay_state"})


def roundtrip_batch(batch):
    data = pack_batch(batch)
    return tuple(data.batch[key] for key in _BATCH_KEYS)


def _upstream_policy_loss(core, old, selected, advantages, mask, clip_low, clip_high, average):
    loss, _ = core.compute_policy_loss(
        old_log_probs=old, log_probs=selected, advantages=advantages, response_mask=mask,
        clip_ratio_low=clip_low, clip_ratio_high=clip_high,
        clip_ratio_dual=2.0 * math.exp(_LOG_RATIO_LIMIT), tau_positive=1.0, tau_negative=1.0,
        loss_type="default", loss_avg_mode=average,
    )
    return loss


def group_losses(policy, behavior, reference, option_mask, actions, advantages, sample_mask,
                 clip_low, clip_high, beta, *, sum_groups=False):
    """Return per-state loss or its sum, with per-state KL and clipping diagnostics."""
    import torch

    from .replay_objective import clipping_ratios, exact_clipped_policy_loss

    _, core, _, _ = _upstream()
    selected = policy.gather(-1, actions).masked_fill(~sample_mask, 0)
    old = behavior.gather(-1, actions).masked_fill(~sample_mask, 0)
    log_ratio = selected - old
    ratio, clipped = clipping_ratios(log_ratio, sample_mask, clip_low, clip_high)
    counts = sample_mask.sum(-1)
    outside = ((clipped != ratio) & sample_mask).sum(-1) / counts
    tail = ((log_ratio.abs() >= _LOG_RATIO_LIMIT) & sample_mask).any(-1, keepdim=True)

    # Matching finite sentinels make excluded options contribute zero to EasyR1's exact full KL.
    kl = core.compute_kl(policy.masked_fill(~option_mask, 0), reference.masked_fill(~option_mask, 0), "full")
    if sum_groups:
        regular = sample_mask & ~tail
        # EasyR1's sequence reduction preserves equal weights for ragged states. Masked
        # tail rows contribute zero to the batch mean and are handled by the exact loss.
        regular_mean = _upstream_policy_loss(
            core, old.masked_fill(~regular, 0), selected.masked_fill(~regular, 0),
            advantages.masked_fill(~regular, 0), regular, clip_low, clip_high, "seq",
        )
        exact_tail = exact_clipped_policy_loss(
            log_ratio, advantages.masked_fill(~tail, 0), sample_mask, ratio, clipped,
        )
        return regular_mean * policy.shape[0] + (exact_tail + beta * kl).sum(), kl, outside

    group_policy_losses = []
    for row, use_exact_tail in enumerate(tail.flatten().tolist()):
        if use_exact_tail:
            # EasyR1 clamps the raw log ratio at ±20. Keep the replay gradient at and beyond that boundary.
            loss = exact_clipped_policy_loss(
                log_ratio[row], advantages[row], sample_mask[row], ratio[row], clipped[row],
            )
        else:
            start = slice(row, row + 1)
            loss = _upstream_policy_loss(
                core, old[start], selected[start], advantages[start], sample_mask[start],
                clip_low, clip_high, "token",
            )
        group_policy_losses.append(loss)

    return torch.stack(group_policy_losses) + beta * kl, kl, outside
