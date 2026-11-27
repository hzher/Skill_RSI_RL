"""Collect fresh native rollouts for each DPO update, then publish the next student."""

import argparse
import gc
import json
import math
from pathlib import Path

from skill_rsi_rl.algorithms import api
from skill_rsi_rl.contracts import MODEL_PATH, digest
from skill_rsi_rl.online import collect, configuration, load_batch, policy_service, read
from skill_rsi_rl.rollout_batch import check_group
from .objective import dpo_loss, online_preferences, score
from .offline import load_policy, single_process


def update(config, model_path, parent, previous_optimizer, groups, policy_id, round_id, destination, settings):
    import torch
    from skill_rsi_rl.training_artifacts import save_checkpoint

    pair_groups = []
    for group in groups:
        rewards = check_group(group, policy_id, config["group_size"])
        pairs = online_preferences(group, rewards, config["min_reward_gap"])
        if pairs:
            pair_groups.append(pairs)
    if not pair_groups:
        raise ValueError("no same-state preference pair of distinct sampled actions; collect a new batch")
    device = torch.device(config["device"])
    torch.manual_seed(config["seed"] + round_id)
    model, tokenizer, _ = load_policy(model_path, parent, device, config["lora_rank"])
    model.eval()
    # Check the whole frozen batch before any optimizer step or publication.
    max_error = 0.0
    for group in groups:
        for row in group:
            for step in row["decisions"]:
                api.validate_replay_tokens({**step, "state_id": step["query_id"]}, tokenizer,
                                          vocab_size=model.config.vocab_size,
                                          max_prompt_tokens=min(config["max_prompt_tokens"],
                                                                model.config.max_position_embeddings))
                with torch.no_grad():
                    pred = score(model, step["prompt_token_ids"], step["option_token_ids"], device)
                error = (pred[0] - pred.new_tensor(step["behavior_logprobs"])).abs().max().item()
                if not math.isfinite(error) or error > config["logp_tolerance"]:
                    raise ValueError(f"frozen policy logp mismatch: {error} > {config['logp_tolerance']}")
                max_error = max(error, max_error)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0.0)
    if previous_optimizer is not None:
        optimizer.load_state_dict(previous_optimizer)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for pairs in pair_groups:
        weight = 1.0 / (len(pair_groups) * len(pairs))
        for winner, loser in pairs:
            prompt, options = winner["prompt_token_ids"], winner["option_token_ids"]
            with torch.no_grad(), model.disable_adapter():
                reference = score(model, prompt, options, device)
            pred = score(model, prompt, options, device)
            loss = dpo_loss(pred, reference, winner["option_index"], loser["option_index"], config["beta"])
            (loss * weight).backward()
            loss_sum += weight * loss.detach().item()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    if norm.item() == 0:
        raise ValueError("DPO batch produced no gradient")
    previous = [p.detach().cpu().clone() for p in parameters]
    optimizer.step()
    delta = max((p.detach().cpu() - before).abs().max().item()
                for p, before in zip(parameters, previous, strict=True))
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("optimizer failed to produce a finite LoRA update")
    metrics = dict(loss=loss_sum, preference_pairs=sum(map(len, pair_groups)),
                   groups=len(groups), eligible_groups=len(pair_groups), max_logp_error=max_error,
                   grad_norm=norm.item(), max_parameter_change=delta, optimizer_step=round_id)
    save_checkpoint(destination, model, optimizer, {**settings, "metrics": metrics}, round_id, round_id)
    return metrics


def run(config, model, initial, output, rounds, bench_python):
    import torch
    from skill_rsi_rl.serve import checkpoint_identities, checkpoint_identity
    from skill_rsi_rl.sources import PROJECT, verify_source
    from skill_rsi_rl.training_artifacts import (
        _write_run_json, file_sha256, load_training_state, lock_output,
    )

    single_process()
    if rounds < 1 or not bench_python.is_file():
        raise ValueError("positive rounds and an existing benchmark Python executable are required")
    base_id, parent_id = checkpoint_identities(model, initial)
    from skill_rsi_rl.harnesses.native_spec import benchmark_commit
    bench_commit = benchmark_commit(config["benchmark"])
    package = Path(__file__).parent
    sources = {str(p.relative_to(PROJECT)): file_sha256(p)
               for root in (package, PROJECT / "skill_rsi_rl/harnesses")
               for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in {".py", ".json", ".mjs"}}
    for name in ("online.py", "rollout_batch.py", "serve.py", "train.py", "contracts.py", "poker_split.py"):
        path = PROJECT / "skill_rsi_rl" / name
        sources[str(path.relative_to(PROJECT))] = file_sha256(path)
    manifest = dict(schema_version=1, algorithm="dpo", algorithm_version=1, mode="online",
                    config={k: v for k, v in config.items() if k != "bench_python"},
                    base_policy_id=base_id, reference_policy_id=base_id, initial_policy_id=parent_id,
                    benchmark_commit=bench_commit, benchmark_kind=config["benchmark"].get("kind", "clbench"),
                    clbench_commit=bench_commit if config["benchmark"].get("kind", "clbench") == "clbench" else None,
                    sources=sources,
                    source_lock_sha256=file_sha256(PROJECT / "sources.lock.json"))
    identity = digest(manifest)
    parent, optimizer = initial, None
    with lock_output(output):
        manifest_path = output / "manifest.json"
        if manifest_path.exists() and read(manifest_path) != manifest:
            raise ValueError("output belongs to a different DPO experiment")
        if not manifest_path.exists():
            _write_run_json(manifest_path, manifest)
        for number in range(1, rounds + 1):
            rd = output / f"round-{number:04d}"
            rd.mkdir(exist_ok=True)
            batch_file = rd / "batch.json"
            if not batch_file.is_file():
                with policy_service(config, model, parent, parent_id) as endpoint:
                    collect(config, rd, number, parent_id, endpoint, bench_python, identity, bench_commit)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            groups = load_batch(batch_file, identity, parent_id, number, config, bench_commit)
            settings = dict(algorithm="dpo", algorithm_version=1, identity=identity, round=number,
                            parent_policy_id=parent_id, base_checkpoint_id=base_id,
                            reference_policy_id=base_id, batch_sha256=file_sha256(batch_file))
            checkpoint = rd / "checkpoint"
            if not checkpoint.exists():
                try:
                    update(config, model, parent, optimizer, groups, parent_id, number, checkpoint, settings)
                except ValueError as exc:
                    _write_run_json(rd / "update-rejected.json", {**settings, "reason": str(exc)})
                    raise
            raw = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
            saved = dict(raw["settings"])
            metrics = saved.pop("metrics")
            if saved != settings or metrics["optimizer_step"] != number:
                raise ValueError("checkpoint parent or sealed batch has changed")
            del raw
            progress = load_training_state(checkpoint, {**settings, "metrics": metrics}, epochs=None)
            if progress.step != number or progress.next_epoch != number:
                raise ValueError("checkpoint progress differs from round")
            new_id = checkpoint_identity(model, checkpoint)
            if new_id == parent_id:
                raise ValueError("checkpoint did not produce a new policy identity")
            result = {**settings, "policy_id": new_id, "checkpoint": "checkpoint", "metrics": metrics,
                      "training_state_sha256": file_sha256(checkpoint / "training_state.pt")}
            result_path = rd / "result.json"
            if result_path.exists() and read(result_path) != result:
                raise ValueError("published DPO round differs from committed weights or optimizer")
            if not result_path.exists():
                _write_run_json(result_path, result)
            parent, parent_id, optimizer = checkpoint, new_id, progress.optimizer
            print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--bench-python", type=Path)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--min-reward-gap", type=float, default=0.0)
    args = parser.parse_args()
    if not math.isfinite(args.beta) or args.beta <= 0 or not math.isfinite(args.min_reward_gap) or args.min_reward_gap < 0:
        parser.error("beta must be finite and positive; minimum reward gap must be finite and nonnegative")
    config = configuration(args.config, args.device)
    config.update(beta=args.beta, min_reward_gap=args.min_reward_gap)
    bench_python = args.bench_python or Path(config.get("bench_python", Path(__file__).resolve().parents[3] / ".venv-bench/bin/python"))
    run(config, args.model.expanduser().resolve(strict=True), args.adapter, args.output.resolve(), args.rounds,
        bench_python.expanduser().absolute())


if __name__ == "__main__":
    main()
