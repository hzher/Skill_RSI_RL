"""Collect and merge replay sampled from a frozen categorical behavior policy."""

from argparse import ArgumentParser
from contextlib import ExitStack
import heapq
from itertools import islice, zip_longest
import json
import math
import os
from pathlib import Path
import random
import re

from .contracts import (
    MODEL_PATH, digest, option_token_ids, render_prompt, write_replay,
)
from .replay_io import open_replay
from .replay_spec import load_spec, validate_scored_state
from .scenarios import FAMILIES, synthetic_state


def behavior_log_probs(log_probs, temperature, uniform_mix):
    tempered = (log_probs / temperature).log_softmax(-1)
    if uniform_mix == 0.0:
        return tempered
    uniform = tempered.new_full(tempered.shape, -math.log(tempered.numel()))
    if uniform_mix == 1.0:
        return uniform
    return (tempered + math.log1p(-uniform_mix)).logaddexp(uniform + math.log(uniform_mix))


class FrozenPolicy:
    def __init__(self, model, revision, device, dtype, temperature, max_prompt_tokens, checkpoint_id,
                 uniform_mix=0.0, *, adapter=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .train import option_logprobs

        if device == "auto":
            rank = int(os.environ.get("LOCAL_RANK", "0"))
            device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.option_logprobs = option_logprobs
        adapter_provenance = {}
        if adapter is not None:
            from .serve import checkpoint_identities

            adapter = Path(adapter).expanduser().resolve(strict=True)
            base_id, adapter_id = checkpoint_identities(model, adapter)
            if checkpoint_id is not None and checkpoint_id != adapter_id:
                raise ValueError("Behavior checkpoint ID does not match the base and adapter files")
            checkpoint_id = adapter_id
            adapter_provenance = {"behavior_adapter": str(adapter), "behavior_base_checkpoint_id": base_id}
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, dtype=dtype if dtype == "auto" else getattr(torch, dtype),
            device_map={"": self.device},
        )
        if adapter is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=False, local_files_only=True)
        self.model.requires_grad_(False).eval()
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Behavior tokenizer needs a padding or end-of-sequence token")
        revision = getattr(self.model.config, "_commit_hash", None) or revision
        if checkpoint_id is None:
            if Path(model).is_dir():
                from .serve import checkpoint_identity

                checkpoint_id = checkpoint_identity(model)
            elif isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision):
                checkpoint_id = "sha256:" + digest([model, revision])
            else:
                raise ValueError("A remote behavior model must resolve to an immutable revision")
        self.max_prompt_tokens = max_prompt_tokens
        self.temperature = temperature
        self.uniform_mix = uniform_mix
        self.manifest = {
            "behavior_model": str(model),
            "behavior_model_revision": revision,
            "behavior_checkpoint_id": checkpoint_id,
            "behavior_model_dtype": str(next(self.model.parameters()).dtype),
            "behavior_tokenizer_sha256": digest({
                "tokenizer": self.tokenizer.backend_tokenizer.to_str(),
                "chat_template": self.tokenizer.chat_template,
                "special_ids": self.tokenizer.all_special_ids,
            }),
            "behavior_temperature": temperature,
            "behavior_sampler": "frozen_model_categorical_with_replacement",
            **adapter_provenance,
        }
        if uniform_mix:
            self.manifest.update(behavior_uniform_mix=uniform_mix,
                                 behavior_sampler="frozen_model_uniform_mixture_with_replacement")

    def sample(self, messages, labels, count, seed):
        return self.sample_many([(messages, labels, seed)], count)[0]

    def sample_many(self, inputs, count):
        tokenized = [(render_prompt(self.tokenizer, messages), option_token_ids(self.tokenizer, labels), seed)
                     for messages, labels, seed in inputs]
        prompt_width = max(len(prompt) for prompt, _, _ in tokenized)
        option_width = max(len(options) for _, options, _ in tokenized)
        if prompt_width > self.max_prompt_tokens:
            raise ValueError(f"State needs {prompt_width} prompt tokens, above {self.max_prompt_tokens}")
        prompts, masks, choices, supports = [], [], [], []
        for prompt, options, _ in tokenized:
            pad_prompt = prompt_width - len(prompt)
            pad_options = option_width - len(options)
            prompts.append([self.pad_token_id] * pad_prompt + prompt)
            masks.append([0] * pad_prompt + [1] * len(prompt))
            choices.append(options + [0] * pad_options)
            supports.append([True] * len(options) + [False] * pad_options)
        torch = self.torch
        with torch.inference_mode():
            normalized = self.option_logprobs(
                self.model, *(torch.tensor(values, device=self.device)
                              for values in (prompts, masks, choices, supports)),
            ).to(device="cpu", dtype=torch.float64)
            results = []
            for (prompt, options, seed), row in zip(tokenized, normalized, strict=True):
                logged = behavior_log_probs(row[:len(options)], self.temperature, self.uniform_mix)
                probabilities = logged.exp()
                if not torch.isfinite(logged).all() or not (probabilities > 0).all():
                    raise ValueError("Behavior policy probabilities must be finite and positive")
                rng = torch.Generator(device="cpu").manual_seed(seed)
                draws = torch.multinomial(probabilities, count, replacement=True, generator=rng).tolist()
                results.append((prompt, options, logged.tolist(), draws))
        return results


class FixturePolicy:
    """Deterministic CPU fixtures for verifying replay tooling."""

    manifest = {
        "behavior_model": "fixture-uniform-v1",
        "behavior_sampler": "fixture_uniform_with_replacement",
        "fixture_only": True,
    }

    @staticmethod
    def sample(messages, labels, count, seed):
        rng = random.Random(seed)
        prompt_tokens = list(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
        return (prompt_tokens, [ord(label) for label in labels], [-math.log(len(labels))] * len(labels),
                [rng.randrange(len(labels)) for _ in range(count)])

    def sample_many(self, inputs, count):
        return [self.sample(messages, labels, count, seed) for messages, labels, seed in inputs]


def _load_replay_spec(path, split):
    spec, source_sha = load_spec(path, split)
    return spec["states"], {
        "source_kind": "authorized_replay",
        "dataset_id": spec["dataset_id"],
        "authorization_ref": spec["authorization_ref"],
        "score_protocol": spec["score_protocol"],
        "source_sha256": source_sha,
        **{key: spec[key] for key in ("plan_sha256", "source_specs") if key in spec},
    }


def _prepare_group(state, index, seed):
    messages = validate_scored_state(state)
    options = state["options"]
    scores = state["scores"]
    draw_seed = int(digest([seed, state["state_id"], "behavior-sampling-v1"])[:16], 16)
    inputs = messages, [option["label"] for option in options], draw_seed
    group = {
        "state_id": state["state_id"],
        "scenario_id": state["scenario_id"],
        "state_index": index,
        "messages": messages,
        "observation_sha256": digest(messages),
        "options": options,
    }
    if "contrast_id" in state:
        group.update(contrast_id=state["contrast_id"], family=state["family"])
    return group, scores, inputs


def _sampled_group(group, scores, result, *, include_scores=False):
    prompt_ids, label_ids, logged, draws = result
    group.update({
        "prompt_token_ids": prompt_ids,
        "option_token_ids": label_ids,
        "behavior_logprobs": logged,
        "samples": [{"option_index": i, "reward": scores[i]["reward"],
                     "native_success": scores[i]["native_success"]} for i in draws],
    })
    if include_scores:
        group["option_scores"] = [{"reward": score["reward"], "native_success": score["native_success"]}
                                  for score in scores]
    return group


def _collect_groups(policy, states, indices, samples, seed, batch_size, include_scores):
    prepared = (_prepare_group(states(index), index, seed) for index in indices)
    while batch := list(islice(prepared, batch_size)):
        outputs = policy.sample_many([inputs for _, _, inputs in batch], samples)
        for (group, scores, _), result in zip(batch, outputs, strict=True):
            yield _sampled_group(group, scores, result, include_scores=include_scores)


def collect(args):
    batch_size = getattr(args, "inference_batch_size", 1)
    suite = getattr(args, "scenario_suite", "v1")
    uniform_mix = getattr(args, "uniform_mix", 0.0)
    adapter = getattr(args, "adapter", None)
    if args.samples_per_state < 2 or args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("Collect requires at least two samples per state and a valid shard rank")
    if batch_size < 1:
        raise ValueError("Inference batch size must be positive")
    if args.max_prompt_tokens < 1 or not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("Prompt-token limit and behavior temperature must be positive")
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("Uniform behavior mixture must be between zero and one")
    if args.checkpoint_id and not re.fullmatch(r"sha256:[0-9a-f]{64}", args.checkpoint_id):
        raise ValueError("Checkpoint ID must use sha256:<64 hex digits>")
    if adapter is not None and (args.fixture or not Path(args.model).is_dir()):
        raise ValueError("Collecting from an adapter requires a local base model and non-fixture policy")
    if args.replay_spec:
        if suite != "v1":
            raise ValueError("Select a synthetic scenario suite or a replay spec")
        states, provenance = _load_replay_spec(args.replay_spec, args.split)
        state_count = len(states)
        get_state = states.__getitem__
    else:
        state_count = args.states
        if suite in ("v2", "v3"):
            if suite == "v2":
                from . import scenarios_v2 as scenarios
            else:
                from . import scenarios_v3 as scenarios

            if state_count % scenarios.STATES_PER_SCENARIO:
                raise ValueError(f"Scenario suite {suite} requires a multiple of {scenarios.STATES_PER_SCENARIO} states")
            get_state = lambda index: scenarios.synthetic_state(args.split, args.seed, index)
            provenance = {
                "source_kind": "synthetic", "scenario_suite": suite,
                "dataset_id": f"{scenarios.SUITE_ID}-{args.split}-{args.seed}-{state_count}",
                "score_protocol": scenarios.SCORE_PROTOCOL,
                "scenario_families": list(scenarios.FAMILIES),
                "states_per_scenario": scenarios.STATES_PER_SCENARIO,
            }
        else:
            get_state = lambda index: synthetic_state(args.split, args.seed, index)
            provenance = {
                "source_kind": "synthetic",
                "dataset_id": f"context-memory-fixtures-v1-{args.split}-{args.seed}-{state_count}",
                "score_protocol": "bounded_context_and_next_query_persistence_v1",
                "scenario_families": list(FAMILIES),
            }
    if state_count < args.world_size:
        raise ValueError("Number of states must be at least the shard count")
    if args.fixture and (args.replay_spec or uniform_mix):
        raise ValueError("Fixture policy only supports synthetic scenarios and is already uniform")
    if args.fixture:
        policy = FixturePolicy()
    else:
        policy = FrozenPolicy(args.model, args.revision, args.device, args.dtype, args.temperature,
                              args.max_prompt_tokens, args.checkpoint_id, uniform_mix, adapter=adapter)
    manifest = {
        "schema_version": 1,
        "training_permission": "weight_training",
        "split": args.split,
        "seed": args.seed,
        "sampled_outcomes_per_state": args.samples_per_state,
        "state_count": state_count,
        "inference_batch_size": batch_size,
        "shard": {"rank": args.rank, "world_size": args.world_size},
        **provenance,
        **policy.manifest,
    }
    indices = range(args.rank, state_count, args.world_size)
    write_replay(args.output, manifest, _collect_groups(
        policy, get_state, indices, args.samples_per_state, args.seed, batch_size, args.split == "dev",
    ))
    return {"output": str(args.output), "states": len(indices), "samples": len(indices) * args.samples_per_state}


def merge(output, shard_paths):
    manifest = None
    shards = {}
    world_size = None
    with ExitStack() as sources:
        for path in shard_paths:
            source, groups = sources.enter_context(open_replay(path))
            shard = source.pop("shard")
            rank, size = shard["rank"], shard["world_size"]
            if manifest is None:
                manifest, world_size = source, size
            if (source != manifest or type(size) is not int or size != world_size or type(rank) is not int or
                    not 0 <= rank < size or rank in shards):
                raise ValueError("Replay shard identity or collection settings differ")
            shards[rank] = groups
        if not manifest or len(shards) != world_size:
            raise ValueError("Every replay shard is required exactly once")
        count, samples = manifest["state_count"], manifest["sampled_outcomes_per_state"]
        if type(count) is not int or count < world_size or type(samples) is not int or samples < 2:
            raise ValueError("Invalid state or sample count in replay shard manifest")
        ordered = [_ordered_shard(groups, range(rank, count, world_size), samples, rank)
                   for rank, groups in shards.items()]
        manifest["shard"] = {"rank": 0, "world_size": 1}
        write_replay(output, manifest, heapq.merge(*ordered, key=lambda group: group["state_index"]))
    return {"output": str(output), "states": count, "shards": world_size}


def _ordered_shard(groups, indices, samples, rank):
    for group, index in zip_longest(groups, indices):
        if (group is None or index is None or type(group.get("state_index")) is not int or
                group["state_index"] != index or len(group["samples"]) != samples):
            raise ValueError(f"State order, coverage or sample count differs for shard rank {rank}")
        yield group


def main():
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="sample and freeze one replay shard")
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--model", default=MODEL_PATH)
    sample.add_argument("--revision")
    sample.add_argument("--adapter", type=Path, help="local frozen LoRA adapter to sample as the behavior policy")
    sample.add_argument("--checkpoint-id", help="precomputed base identity, or expected identity when using an adapter")
    sample.add_argument("--device", default="auto")
    sample.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    sample.add_argument("--temperature", type=float, default=1.0)
    sample.add_argument("--uniform-mix", type=float, default=0.0, help="mix the frozen policy with uniform exploration")
    sample.add_argument("--max-prompt-tokens", type=int, default=4096)
    sample.add_argument("--inference-batch-size", type=int, default=1, help="states per frozen behavior forward pass")
    sample.add_argument("--split", choices=("train", "dev"), default="train")
    sample.add_argument("--seed", type=int, default=17)
    sample.add_argument("--states", type=int, default=128)
    sample.add_argument("--scenario-suite", choices=("v1", "v2", "v3"), default="v1")
    sample.add_argument("--samples-per-state", type=int, default=8)
    sample.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    sample.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    sample.add_argument("--replay-spec", type=Path, help="authorized states and independently scored options JSON")
    sample.add_argument("--fixture", action="store_true", help="use a uniform CPU policy for regression tests")
    combine = commands.add_parser("merge", help="verify and combine all collection shards")
    combine.add_argument("--output", type=Path, required=True)
    combine.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        result = collect(args) if args.command == "sample" else merge(args.output, args.shards)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
