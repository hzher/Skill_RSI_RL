"""Train a finite-action context policy from frozen, scored replay groups."""

import argparse
import math
import os
import random
import time
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from itertools import count
from pathlib import Path
from statistics import fmean, pstdev

import torch
from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel

from .contracts import MODEL_PATH, option_token_ids, render_prompt
from .replay_io import open_hashed_replay
from .replay_objective import clipping_ratios, exact_clipped_policy_loss
from .training_artifacts import (
    append_record,
    file_sha256,
    initialize_run,
    load_training_state,
    lock_output,
    save_checkpoint as checkpoint,
)


@dataclass(frozen=True)
class Group:
    prompt: list[int]
    options: list[int]
    behavior: list[float]
    actions: list[int]
    advantages: list[float]
    reward_sum: float = 0.0
    native_successes: int = 0
    policy_weight: float = 1.0


@dataclass(frozen=True)
class Worker:
    rank: int
    world: int
    device: torch.device

    def sum(self, values):
        if self.world > 1:
            dist.all_reduce(values)
        return values

    def run_on_primary(self, stage, operation):
        """Broadcast startup or checkpoint failure before peers enter the next collective."""
        if self.world == 1:
            return operation()
        reply = [None]
        local_error = None
        if self.rank == 0:
            try:
                reply[0] = (True, operation())
            except Exception as error:
                local_error = error
                reply[0] = (False, f"{type(error).__name__}: {error}")
        dist.broadcast_object_list(reply, src=0, device=self.device)
        if local_error is not None:
            raise local_error
        succeeded, result = reply[0]
        if not succeeded:
            raise RuntimeError(f"{stage} failed on rank 0: {result}")
        return result

    def run_on_all_ranks(self, stage, operation, *, require_same_result=False):
        """Complete all local preparation before initializing DDP or using its collectives."""
        if self.world == 1:
            return operation()
        result = local_error = failure = None
        try:
            result = operation()
        except torch.cuda.OutOfMemoryError:
            # The rank may be unable to allocate the CUDA buffer needed for error propagation.
            raise
        except Exception as error:
            local_error = error
            failure = f"{type(error).__name__}: {error}"
        other_ranks = [None] * self.world
        dist.all_gather_object(other_ranks, (failure, result if require_same_result else None))
        if local_error is not None:
            raise local_error
        for rank, (rank_failure, _) in enumerate(other_ranks):
            if rank_failure is not None:
                raise RuntimeError(f"{stage} failed on rank {rank}: {rank_failure}")
        if require_same_result and any(rank_result != result for _, rank_result in other_ranks):
            raise RuntimeError(f"{stage} differs across ranks")
        return result


def prepare_groups(rows, max_prompt_tokens, vocab_size, tokenizer=None, *, advantage_std_floor=0.0,
                   recall_write_weight=1.0):
    groups = []
    skipped = Counter()
    for row in rows:
        state, prompt, options = (row[key] for key in
                                  ("state_id", "prompt_token_ids", "option_token_ids"))
        if not prompt or len(prompt) > max_prompt_tokens:
            raise ValueError(f"{state}: prompt length {len(prompt)} outside 1..{max_prompt_tokens}")
        if any(token < 0 or token >= vocab_size for token in (*prompt, *options)):
            raise ValueError(f"{state}: token ID is outside the model vocabulary")
        if tokenizer is not None:
            labels = [option["label"] for option in row["options"]]
            if (prompt != render_prompt(tokenizer, row["messages"])
                    or options != option_token_ids(tokenizer, labels)):
                raise ValueError(f"{state}: replay tokenization differs from the model tokenizer")
        samples = row["samples"]
        if len(samples) < 2:
            skipped["fewer_than_two_samples"] += 1
            continue
        rewards = [sample["reward"] for sample in samples]
        reward_std = pstdev(rewards)
        if reward_std == 0:
            skipped["zero_reward_variance"] += 1
            continue
        family = row.get("family")
        if recall_write_weight != 1.0 and family not in {"noop", "recall", "compact", "write"}:
            raise ValueError(f"{state}: recall/write weighting requires an identified scenario family")
        reward_mean = fmean(rewards)
        scale = max(reward_std + 1e-8, advantage_std_floor)
        groups.append(Group(prompt, options, row["behavior_logprobs"],
                            [sample["option_index"] for sample in samples],
                            [(reward - reward_mean) / scale for reward in rewards],
                            math.fsum(rewards), sum(sample.get("native_success", False) for sample in samples),
                            recall_write_weight if family in {"recall", "write"} else 1.0))
    if not groups:
        raise ValueError(f"no groups contain a reward comparison: {dict(skipped)}")
    return groups, dict(skipped)


def option_logprobs(model, input_ids, attention_mask, option_ids, option_mask):
    """Score the same labeled support used when sampling the logged behavior."""
    position_ids = attention_mask.cumsum(-1).sub(1).clamp_min(0)
    logits = model(input_ids=input_ids, attention_mask=attention_mask,
                   position_ids=position_ids, use_cache=False, logits_to_keep=1).logits
    selected = logits[:, -1].gather(-1, option_ids).float()
    return selected.masked_fill(~option_mask, -torch.inf).log_softmax(-1)


def group_losses(policy, behavior, reference, option_mask, actions, advantages, sample_mask,
                 clip_low, clip_high, beta):
    log_ratio = policy.gather(-1, actions) - behavior.gather(-1, actions)
    ratio, clipped = clipping_ratios(log_ratio, sample_mask, clip_low, clip_high)
    policy_loss = exact_clipped_policy_loss(log_ratio, advantages, sample_mask, ratio, clipped)
    counts = sample_mask.sum(-1)

    policy_support = policy.masked_fill(~option_mask, 0)
    reference_support = reference.masked_fill(~option_mask, 0)
    kl = (policy.exp() * (policy_support - reference_support)).sum(-1)
    outside = ((clipped != ratio) & sample_mask).sum(-1) / counts
    return policy_loss + beta * kl, kl, outside


def replay_policy_diagnostics(policy, behavior, option_mask, actions, advantages, sampled, clip_low, clip_high):
    probabilities = policy.exp()
    log_policy = policy.masked_fill(~option_mask, 0)
    log_behavior = behavior.masked_fill(~option_mask, 0)
    entropy = -(probabilities * log_policy).sum(-1)
    behavior_kl = (probabilities * (log_policy - log_behavior)).sum(-1)
    log_weights = (log_policy - log_behavior).gather(-1, actions).masked_fill(~sampled, -torch.inf)
    normalized_weights = log_weights.softmax(-1)
    counts = sampled.sum(-1)
    effective_fraction = (normalized_weights.square().sum(-1) * counts).reciprocal()
    active_clip = (((advantages > 0) & (log_weights > math.log1p(clip_high)))
                   | ((advantages < 0) & (log_weights < math.log1p(-clip_low)))) & sampled
    clipped_fraction = active_clip.sum(-1) / counts
    advantage_abs = advantages.abs().masked_fill(~sampled, 0).sum(-1) / counts
    return torch.stack([metric.sum() for metric in (
        entropy, behavior_kl, effective_fraction, clipped_fraction, advantage_abs,
    )])


def make_batch(groups, pad_token, device):
    prompt_width = max(len(group.prompt) for group in groups)
    option_width = max(len(group.options) for group in groups)
    sample_width = max(len(group.actions) for group in groups)
    prompts, attention, options, support, behavior, actions, advantages, sampled = (
        [] for _ in range(8)
    )
    for group in groups:
        prompt_pad = prompt_width - len(group.prompt)
        option_pad = option_width - len(group.options)
        sample_pad = sample_width - len(group.actions)
        prompts.append([pad_token] * prompt_pad + group.prompt)
        attention.append([0] * prompt_pad + [1] * len(group.prompt))
        options.append(group.options + [0] * option_pad)
        support.append([True] * len(group.options) + [False] * option_pad)
        behavior.append(group.behavior + [0.] * option_pad)
        actions.append(group.actions + [0] * sample_pad)
        advantages.append(group.advantages + [0.] * sample_pad)
        sampled.append([True] * len(group.actions) + [False] * sample_pad)
    return tuple(torch.tensor(values, device=device) for values in
                 (prompts, attention, options, support, behavior, actions, advantages, sampled))


def worker_for(device):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA devices are unavailable")
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if world > 1:
        if device.type == "cuda":
            os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return Worker(rank, world, device)


def model_dtype(config, worker):
    bf16 = config.dtype == "bf16" or (config.dtype == "auto" and worker.device.type == "cuda")
    return torch.bfloat16 if bf16 else torch.float32


def load_trainable_adapter(base, checkpoint):
    from peft import PeftModel
    from peft.utils import save_and_load
    from transformers.integrations import tensor_parallel

    shard = getattr(save_and_load, "_maybe_shard_state_dict_for_tp", None)
    if not dist.is_initialized() or shard is None or hasattr(tensor_parallel, "EmbeddingParallel"):
        return PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
    if any(getattr(layer, "_hf_device_mesh", None) is not None for layer in base.modules()):
        raise RuntimeError("This PEFT/Transformers combination cannot restore tensor-parallel adapters")

    # PEFT imports a newer Transformers TP class whenever a process group exists.
    # Our DDP ranks hold whole weights, so the sharding helper would do no work.
    save_and_load._maybe_shard_state_dict_for_tp = lambda model, state, adapter: None
    try:
        return PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
    finally:
        save_and_load._maybe_shard_state_dict_for_tp = shard


def load_model(config, worker):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = Path(config.model).expanduser().resolve(strict=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if not isinstance(pad, int):
        raise ValueError("model tokenizer must define pad_token_id or eos_token_id")
    base = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=model_dtype(config, worker),
        attn_implementation="sdpa", local_files_only=True,
        device_map={"": worker.device},
    )
    base.config.use_cache = False
    adapter = config.resume_from or config.init_adapter
    if adapter:
        model = load_trainable_adapter(base, adapter)
        if model.peft_config["default"].r != config.lora_rank:
            raise ValueError("Adapter rank differs from --lora-rank")
    else:
        model = get_peft_model(base, LoraConfig(r=config.lora_rank, lora_alpha=2 * config.lora_rank,
                                              lora_dropout=0, target_modules=["q_proj", "v_proj"],
                                              task_type="CAUSAL_LM"))
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, tokenizer, pad


def configuration():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--output", required=True, type=Path)
    initializer = parser.add_mutually_exclusive_group()
    initializer.add_argument("--resume-from", type=Path, help="Restore matching weights and optimizer state")
    initializer.add_argument("--init-adapter", type=Path, help="Start a new recipe from adapter weights and a fresh optimizer")
    parser.add_argument("--epochs", type=int, default=2)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--stop-at-unix", type=float,
                        help="stop at a checkpoint before the next epoch would exceed this Unix timestamp")
    budget.add_argument("--train-for-seconds", type=float,
                        help="train until committed training time reaches this duration, then finish the epoch")
    parser.add_argument("--profile-steps", type=int, help="limit a new run to full batches and skip checkpoints")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-epochs", type=int, default=0,
                        help="fixed cosine schedule horizon in epochs; zero disables decay")
    parser.add_argument("--min-lr-ratio", type=float, default=0.2)
    parser.add_argument("--advantage-std-floor", type=float, default=0.0,
                        help="optional reward-scale floor for group-standardized advantages")
    parser.add_argument("--recall-write-weight", type=float, default=1.0,
                        help="relative policy-loss weight for recall and write scenario groups")
    parser.add_argument("--clip-low", type=float, default=0.2)
    parser.add_argument("--clip-high", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp32"), default="auto")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backend", choices=("native", "easyr1"), default="native",
                        help="select the fixed-replay GRPO loss and batch backend")
    parser.add_argument("--monitor", choices=("off", "local", "cloud"), default="off")
    parser.add_argument("--monitor-every", type=int, default=5, help="steps between SwanLab updates")
    parser.add_argument("--swanlab-project", default="skill_rsi")
    parser.add_argument("--swanlab-workspace", default="hzh")
    parser.add_argument("--swanlab-name")
    config = parser.parse_args()
    sizes = (config.epochs, config.per_device_batch_size, config.max_prompt_tokens, config.lora_rank)
    floats = (config.learning_rate, config.clip_low, config.clip_high, config.beta)
    if (min(sizes) < 1 or not all(map(math.isfinite, floats)) or config.learning_rate <= 0
            or not 0 <= config.clip_low < 1 or config.clip_high < 0 or config.beta < 0):
        parser.error("expected positive sizes and learning rate, 0 <= clip-low < 1, clip-high and beta >= 0")
    if config.monitor_every < 1:
        parser.error("--monitor-every must be positive")
    if (min(config.warmup_steps, config.lr_decay_epochs) < 0 or not 0 < config.min_lr_ratio <= 1
            or not math.isfinite(config.advantage_std_floor) or config.advantage_std_floor < 0):
        parser.error("expected nonnegative warmup, decay and advantage floor, and min LR ratio in (0, 1]")
    if config.profile_steps is not None and (config.profile_steps < 1 or config.resume_from or config.train_for_seconds):
        parser.error("--profile-steps requires a positive count and a new output run")
    if not math.isfinite(config.recall_write_weight) or config.recall_write_weight < 1:
        parser.error("--recall-write-weight must be finite and at least one")
    if config.stop_at_unix is not None and (not math.isfinite(config.stop_at_unix) or config.stop_at_unix <= 0):
        parser.error("--stop-at-unix must be a positive, finite Unix timestamp")
    if config.train_for_seconds is not None and (not math.isfinite(config.train_for_seconds)
                                                 or config.train_for_seconds <= 0):
        parser.error("--train-for-seconds must be a positive, finite duration")
    return config


def training_backend(name):
    if name == "easyr1":
        from . import easyr1_backend

        return (partial(easyr1_backend.group_losses, sum_groups=True), easyr1_backend.roundtrip_batch,
                easyr1_backend.backend_provenance())
    return group_losses, None, None


def training_settings(config, worker, manifest, replay_sha256, base_identity, backend_provenance):
    model_path = Path(config.model).expanduser().resolve()
    behavior_path = Path(manifest["behavior_model"]).expanduser().resolve()
    behavior_base = manifest.get("behavior_base_checkpoint_id", manifest["behavior_checkpoint_id"])
    if behavior_path == model_path and behavior_base != base_identity:
        raise ValueError("the current base differs from the checkpoint that sampled this replay")
    optimizer_fields = ("lora_rank", "learning_rate", "clip_low", "clip_high", "beta",
                        "seed", "per_device_batch_size", "max_prompt_tokens", "gradient_checkpointing")
    settings = {key: getattr(config, key) for key in optimizer_fields}
    if config.warmup_steps or config.lr_decay_epochs:
        settings["lr_schedule"] = {"warmup_steps": config.warmup_steps, "decay_epochs": config.lr_decay_epochs,
                                   "min_lr_ratio": config.min_lr_ratio}
    if config.advantage_std_floor:
        settings["advantage_std_floor"] = config.advantage_std_floor
    if config.recall_write_weight != 1.0:
        settings["recall_write_policy_weight"] = config.recall_write_weight
    settings.update(dataset_id=manifest["dataset_id"], replay_sha256=replay_sha256,
                    base_checkpoint_id=base_identity,
                    behavior_checkpoint_id=manifest["behavior_checkpoint_id"],
                    dtype=str(model_dtype(config, worker)), world_size=worker.world)
    if backend_provenance is not None:
        settings["backend"] = {k: v for k, v in backend_provenance.items() if k != "checkout"}
    return settings


def learning_rate_at_step(peak, step, warmup, decay_steps, min_ratio):
    """Use completed optimizer updates so checkpoint resume needs no scheduler state."""
    if step < warmup:
        return peak * (step + 1) / warmup
    if not decay_steps:
        return peak
    progress = min(1.0, (step - warmup) / (decay_steps - warmup - 1))
    return peak * (min_ratio + (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress)) / 2)


def time_for_next_epoch(deadline, now, last_epoch_seconds):
    margin = max(10.0, 0.05 * last_epoch_seconds)
    return deadline - now >= last_epoch_seconds + margin


def prepare_training(config, worker, rows, progress):
    torch.manual_seed(config.seed)
    model, tokenizer, pad = load_model(config, worker)
    if next(model.get_base_model().parameters()).dtype != model_dtype(config, worker):
        raise ValueError("the loaded base model does not use the configured training dtype")
    max_prompt = min(config.max_prompt_tokens, model.config.max_position_embeddings)
    groups, skipped = prepare_groups(rows, max_prompt, model.config.vocab_size, tokenizer,
                                     advantage_std_floor=config.advantage_std_floor,
                                     recall_write_weight=config.recall_write_weight)
    if not 0 <= pad < model.config.vocab_size:
        raise ValueError("padding token ID is outside the model vocabulary")
    random.Random(config.seed).shuffle(groups)
    groups.sort(key=lambda group: len(group.prompt))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    if progress.optimizer is not None:
        steps_per_epoch = math.ceil(len(groups) / (config.per_device_batch_size * worker.world))
        if progress.step != progress.next_epoch * steps_per_epoch:
            raise ValueError("resume checkpoint step does not match the replay batch plan")
        optimizer.load_state_dict(progress.optimizer)
    return model, pad, groups, skipped, trainable, optimizer


def gpu_memory(worker, *, peak=False):
    if worker.device.type != "cuda":
        return []
    allocated = torch.cuda.max_memory_allocated if peak else torch.cuda.memory_allocated
    reserved = torch.cuda.max_memory_reserved if peak else torch.cuda.memory_reserved
    local = torch.tensor([allocated(worker.device), reserved(worker.device),
                          torch.cuda.get_device_properties(worker.device).total_memory],
                         dtype=torch.int64, device=worker.device)
    if worker.world > 1:
        gathered = torch.empty(worker.world * local.numel(), dtype=local.dtype, device=local.device)
        dist.all_gather_into_tensor(gathered, local)
    else:
        gathered = local
    if worker.rank != 0:
        return []
    return [{"rank": rank, "allocated_bytes": alloc, "reserved_bytes": reserve, "total_bytes": size}
            for rank, (alloc, reserve, size) in enumerate(gathered.view(-1, 3).tolist())]


def train(config, worker, manifest, rows, replay_sha256, contexts):
    from .serve import checkpoint_identity
    from .tracking import open_tracking, training_metrics

    losses_for, batch_adapter, backend_provenance = worker.run_on_all_ranks(
        "load training backend", lambda: training_backend(config.backend),
    )
    base_identity = worker.run_on_primary("fingerprint base checkpoint", lambda: checkpoint_identity(config.model))
    settings = worker.run_on_all_ranks(
        "freeze training settings",
        lambda: training_settings(config, worker, manifest, replay_sha256, base_identity, backend_provenance),
        require_same_result=True,
    )
    budget = config.train_for_seconds
    epochs = config.epochs if budget is None else None
    progress = worker.run_on_all_ranks(
        "read resume checkpoint", lambda: load_training_state(config.resume_from, settings, epochs),
    )
    worker.run_on_all_ranks(
        "agree on resume progress", lambda: (progress.next_epoch, progress.step, progress.training_seconds),
        require_same_result=True,
    )
    if budget is not None and progress.training_seconds >= budget:
        raise ValueError("Resume checkpoint already meets --train-for-seconds")
    model, pad, groups, skipped, trainable, optimizer = worker.run_on_all_ranks(
        "prepare model, replay groups, and optimizer", lambda: prepare_training(config, worker, rows, progress),
    )
    batch_width = config.per_device_batch_size * worker.world
    profile = config.profile_steps is not None
    decay_steps = config.lr_decay_epochs * math.ceil(len(groups) / batch_width)
    if decay_steps and decay_steps <= config.warmup_steps + 1:
        raise ValueError("Cosine decay needs at least two updates after warmup")
    if profile and len(groups) // batch_width < config.profile_steps:
        raise ValueError(f"Profiling needs {config.profile_steps} full batches of {batch_width} groups; "
                         f"replay has {len(groups)} usable groups")
    tracker = None

    def open_run():
        if config.stop_at_unix is not None and time.time() >= config.stop_at_unix:
            raise ValueError("--stop-at-unix elapsed before the first training epoch")
        run_data = {"settings": settings, "training_groups": len(groups), "skipped_groups": skipped,
                    "behavior_model": manifest["behavior_model"], "trainable_parameters": sum(
                        parameter.numel() for parameter in trainable),
                    "accelerator": (torch.cuda.get_device_name(worker.device)
                                    if worker.device.type == "cuda" else "cpu")}
        if config.init_adapter is not None:
            source = config.init_adapter.resolve(strict=True)
            run_data["initial_adapter"] = {"path": str(source), "sha256": file_sha256(source / "adapter_model.safetensors")}
        if profile:
            run_data["profile_steps"] = config.profile_steps
        if budget is not None:
            run_data["train_for_seconds"] = budget
        parameters = {key: value for key, value in settings.items() if key not in {"backend", "model"}}
        parameters.update(model=Path(config.model).name, backend=config.backend,
                          training_groups=len(groups), trainable_parameters=run_data["trainable_parameters"],
                          resumed_step=progress.step)
        parameters.update({"train_for_seconds": budget} if budget is not None else {"epochs": 1 if profile else epochs})
        if profile:
            parameters["profile_steps"] = config.profile_steps

        def start_tracking():
            nonlocal tracker
            tracker = contexts.enter_context(open_tracking(
                config.monitor, config.output, parameters, project=config.swanlab_project,
                workspace=config.swanlab_workspace, name=config.swanlab_name,
            ))

        return initialize_run(config.output, run_data, progress, epochs, before_commit=start_tracking)

    metrics_path = worker.run_on_primary("open training output", open_run)
    step = progress.step
    loss_ema = progress.loss_ema
    training_seconds = progress.training_seconds
    actor = model
    if worker.world > 1:
        device_ids = [worker.device.index] if worker.device.type == "cuda" else None
        actor = DistributedDataParallel(model, device_ids=device_ids, broadcast_buffers=False,
                                        gradient_as_bucket_view=True)
    model.train()

    limit = len(groups) - batch_width + 1 if profile else len(groups)
    starts = list(range(0, limit, batch_width))
    if profile:
        starts[-1] = len(groups) - batch_width
    reference_cache = {}
    epoch_indices = count(progress.next_epoch) if budget is not None else range(progress.next_epoch, 1 if profile else epochs)
    for epoch in epoch_indices:
        epoch_started = time.monotonic()
        torch.manual_seed(config.seed + epoch * worker.world + worker.rank)
        order = starts[:-1] if profile else starts.copy()
        random.Random(config.seed + epoch).shuffle(order)
        if profile:
            # Exercise the largest padded batch even when profiling only a few steps.
            order = [starts[-1], *order[:config.profile_steps - 1]]
        if worker.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(worker.device)
        epoch_loss_sum, epoch_groups = 0.0, 0
        for iteration, start in enumerate(order):
            step_start = time.monotonic()
            global_count = min(batch_width, len(groups) - start)
            local_start = start + config.per_device_batch_size * worker.rank
            local_end = min(local_start + config.per_device_batch_size, start + global_count)
            local = groups[local_start:local_end]
            batch = make_batch(local or groups[:1], pad, worker.device)
            if batch_adapter is not None:
                batch = batch_adapter(batch)
            x, attention, options, option_mask, behavior, actions, advantages, sampled = batch
            cache_hit = bool(local and start in reference_cache)
            if local and not cache_hit:
                model.eval()
                with torch.no_grad(), model.disable_adapter():
                    reference_cache[start] = option_logprobs(model, x, attention, options, option_mask)
                model.train()
            optimizer.zero_grad(set_to_none=True)
            policy = option_logprobs(actor, x, attention, options, option_mask)
            reference = reference_cache[start] if local else policy.detach()
            if not local:
                behavior = policy.detach()
            policy_advantages = advantages
            if config.recall_write_weight != 1.0 and local:
                global_weight = sum(group.policy_weight for group in groups[start:start + global_count])
                local_weights = policy.new_tensor([group.policy_weight for group in local])
                policy_advantages = advantages * (local_weights * (global_count / global_weight)).unsqueeze(-1)
            losses, kl, clipping = losses_for(policy, behavior, reference, option_mask, actions,
                                              policy_advantages, sampled, config.clip_low, config.clip_high, config.beta)
            # DDP averages rank gradients. Scale local sums to recover the mean over real states.
            weight = bool(local)
            (losses.sum() * (worker.world * weight / global_count)).backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            learning_rate = learning_rate_at_step(config.learning_rate, step, config.warmup_steps,
                                                   decay_steps, config.min_lr_ratio)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            optimizer.step()
            with torch.no_grad():
                measurements = torch.stack((losses.sum(), kl.sum(), clipping.sum(), attention.sum().float(),
                                             sampled.sum().float()))
                counts = policy.new_tensor((len(local) if cache_hit else 0, x.numel(),
                                            math.fsum(group.reward_sum for group in local),
                                            sum(group.native_successes for group in local)))
                diagnostics = replay_policy_diagnostics(
                    policy, behavior, option_mask, actions, advantages, sampled, config.clip_low, config.clip_high,
                )
                totals = worker.sum(torch.cat((measurements, counts, diagnostics)) * weight)
            step += 1
            should_track = config.monitor != "off" and (iteration == 0 or iteration == len(order) - 1
                                                       or step % config.monitor_every == 0)
            memory = gpu_memory(worker) if should_track else []
            if worker.rank == 0:
                (loss, kl, clipped, tokens, actions_seen, cached, padded, reward, successes,
                 entropy, behavior_kl, effective_fraction, clipped_fraction, advantage_abs) = totals.tolist()
                mean_loss = loss / global_count
                loss_ema = mean_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * mean_loss
                epoch_loss_sum += loss
                epoch_groups += global_count
                seconds = time.monotonic() - step_start
                metric = {"epoch": epoch + 1, "step": step, "groups": global_count,
                          "sampled_actions": int(actions_seen), "loss": mean_loss, "loss_ema": loss_ema,
                          "policy_loss": (loss - config.beta * kl) / global_count, "learning_rate": learning_rate,
                          "reference_kl": kl / global_count, "ratio_outside_clip": clipped / global_count,
                          "cached_reference_groups": int(cached),
                          "active_ranks": math.ceil(global_count / config.per_device_batch_size),
                          "padded_prompt_tokens": int(padded), "padding_fraction": 1.0 - tokens / padded,
                          "replay_reward_mean": reward / actions_seen,
                          "replay_native_success_rate": successes / actions_seen,
                          "policy_entropy_nats": entropy / global_count, "behavior_kl": behavior_kl / global_count,
                          "importance_ess_fraction": effective_fraction / global_count,
                          "active_clip_fraction": clipped_fraction / global_count,
                          "advantage_abs_mean": advantage_abs / global_count,
                          "grad_norm": norm.item(), "effective_prompt_tokens": int(tokens),
                          "step_seconds": seconds, "effective_prompt_tokens_per_second": tokens / seconds}
                if iteration == len(order) - 1:
                    metric["epoch_loss_mean"] = epoch_loss_sum / epoch_groups
                if memory:
                    metric["rank_memory"] = memory
                append_record(metrics_path, metric)
                if should_track:
                    tracker.log(training_metrics(metric), step=step)
        memory = gpu_memory(worker, peak=True)

        def commit_epoch():
            nonlocal training_seconds
            summary = {"epoch": epoch + 1, "step": step, "mean_loss": epoch_loss_sum / epoch_groups,
                       "learning_rate": learning_rate}
            stop = False
            if memory:
                summary.update(rank_peak_memory=memory,
                               cuda_peak_allocated_bytes=max(row["allocated_bytes"] for row in memory),
                               cuda_peak_reserved_bytes=max(row["reserved_bytes"] for row in memory))
            if profile:
                summary["profile_only"] = True
            else:
                elapsed = time.monotonic() - epoch_started
                training_seconds += elapsed
                destination = config.output / f"checkpoint-{epoch + 1:04d}"
                checkpoint(destination, model, optimizer, settings, epoch + 1, step,
                           loss_ema=loss_ema, training_seconds=training_seconds)
                summary.update(checkpoint=str(destination), epoch_seconds=elapsed, training_seconds=training_seconds)
                if config.stop_at_unix is not None:
                    elapsed = time.monotonic() - epoch_started
                    stop = epoch + 1 < epochs and not time_for_next_epoch(
                        config.stop_at_unix, time.time(), elapsed,
                    )
                    if stop:
                        summary.update(stop_reason="training_deadline", stop_at_unix=config.stop_at_unix)
                if budget is not None and training_seconds >= budget:
                    stop = True
                    summary.update(stop_reason="training_duration", train_for_seconds=budget)
            append_record(config.output / "epochs.jsonl", summary)
            return stop

        if worker.run_on_primary("commit training epoch", commit_epoch):
            break


@record
def main():
    config = configuration()
    with open_hashed_replay(config.replay) as (manifest, groups, replay_digest):
        rows = list(groups)
        replay_sha256 = replay_digest()
    if manifest["split"] != "train":
        raise ValueError("training requires a replay with split=train")
    if manifest.get("fixture_only"):
        raise ValueError("fixture-only replay can be validated but cannot train model weights")
    if manifest.get("shard", {"rank": 0, "world_size": 1}) != {"rank": 0, "world_size": 1}:
        raise ValueError("merge all replay shards before training")
    identity = manifest.get("behavior_checkpoint_id")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("training replay requires an immutable behavior_checkpoint_id")
    worker = worker_for(config.device)
    with ExitStack() as contexts:
        worker.run_on_primary("lock training output", lambda: contexts.enter_context(lock_output(config.output)))
        train(config, worker, manifest, rows, replay_sha256, contexts)
    # On failure let torchrun report the originating error and terminate peers in collectives.
    if worker.world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
