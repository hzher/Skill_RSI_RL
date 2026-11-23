#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s {--plan|--run} NEW_RUN_DIRECTORY\n' "${0##*/}"
    printf '%s\n' 'Run single-H20 EasyR1 GRPO and held-out evaluation.' \
        'Set MODEL_DIR to local weights. Uses .venv-train when installed.' \
        'Optional environment: SKILL_RSI_SCENARIO_SUITE=v1|v2 (default v2), SKILL_RSI_TRAIN_SPEC=FILE,' \
        '                      SKILL_RSI_MONITOR=off|local|cloud.'
}

mode="${1:-}"
case "$mode" in
    -h|--help) usage; exit 0 ;;
    --plan|--run) ;;
    *) usage >&2; exit 2 ;;
esac
if [[ $# != 2 || -z "$2" ]]; then
    usage >&2
    exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="$2"
if [[ "$run_dir" != /* ]]; then
    run_dir="$PWD/$run_dir"
fi
train_spec="${SKILL_RSI_TRAIN_SPEC:-}"
if [[ -n "$train_spec" && "$train_spec" != /* ]]; then
    train_spec="$PWD/$train_spec"
fi
suite="${SKILL_RSI_SCENARIO_SUITE:-v2}"
monitor="${SKILL_RSI_MONITOR:-off}"
case "$suite" in
    v1) train_states=32; dev_states=24; batch_groups=1; uniform_mix=0 ;;
    v2) train_states=128; dev_states=64; batch_groups=4; uniform_mix=0.35 ;;
    *) printf 'Unknown scenario suite: %s\n' "$suite" >&2; exit 2 ;;
esac
case "$monitor" in
    off|local|cloud) ;;
    *) printf 'Unknown monitoring mode: %s\n' "$monitor" >&2; exit 2 ;;
esac
model="${SKILL_RL_MODEL:-${SKILL_RSI_MODEL:-${MODEL_DIR:-$root/models/Qwen3-8B}}}"
if [[ "$model" != /* ]]; then model="$PWD/$model"; fi
cd "$root"
umask 077
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [[ -z "${SKILL_RSI_PYTHON:-}" && -x "$root/.venv-train/bin/python" ]]; then
    SKILL_RSI_PYTHON="$root/.venv-train/bin/python"
fi
export SKILL_RSI_PYTHON="${SKILL_RSI_PYTHON:-python3}"
export SKILL_RSI_MODEL="$model"
export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0
python=("$SKILL_RSI_PYTHON" -B)

run() {
    if [[ "$mode" == --plan ]]; then
        printf '$ '
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

if [[ "$mode" == --run && -e "$run_dir" ]]; then
    printf 'Choose a new run directory: %s\n' "$run_dir" >&2
    exit 2
fi
printf 'CUDA_VISIBLE_DEVICES=%s | EasyR1 | one H20 | dev suite %s\n' "$CUDA_VISIBLE_DEVICES" "$suite"
train_input=(--scenario-suite "$suite" --states "$train_states")
train_draws=16
if [[ -n "$train_spec" ]]; then
    run "${python[@]}" -c 'import sys
from skill_rsi_rl.replay_spec import load_spec, validate_scored_state
spec, _ = load_spec(sys.argv[1], "train")
for state in spec["states"]:
    validate_scored_state(state)
if not any(len({score["reward"] for score in state["scores"]}) > 1 for state in spec["states"]):
    sys.exit("No historical state has distinct candidate rewards. Score another verified boundary.")' "$train_spec"
    train_input=(--replay-spec "$train_spec")
    batch_groups=1
    uniform_mix=0.35
    # Each cut has few actions. More draws improve coverage without new executor calls.
    train_draws=96
fi
preflight=("${python[@]}" -m skill_rsi_rl.preflight --model "$SKILL_RSI_MODEL" --expected-gpus 1 --backend easyr1)
run "${preflight[@]}"
run mkdir -p -- "$(dirname "$run_dir")"
run mkdir -- "$run_dir"

identity=("${python[@]}" -c 'import sys; from skill_rsi_rl.serve import checkpoint_identity; print(checkpoint_identity(sys.argv[1]))' "$SKILL_RSI_MODEL")
if [[ "$mode" == --plan ]]; then
    run "${identity[@]}"
    checkpoint_id='<computed checkpoint identity>'
else
    checkpoint_id="$("${identity[@]}")"
fi
sample=("${python[@]}" -m skill_rsi_rl.collect sample --model "$SKILL_RSI_MODEL"
        --checkpoint-id "$checkpoint_id" --device cuda:0 --dtype bfloat16
        --temperature 1.5)
train_replay="$run_dir/train.jsonl"
dev_replay="$run_dir/dev.jsonl"
run "${sample[@]}" "${train_input[@]}" --samples-per-state "$train_draws" --uniform-mix "$uniform_mix" --split train --seed 17 --output "$train_replay"
run "${sample[@]}" --scenario-suite "$suite" --samples-per-state 16 --split dev --seed 221 --states "$dev_states" --output "$dev_replay"
run "${python[@]}" -m skill_rsi_rl.verify_replay --train "$train_replay" --dev "$dev_replay"
run "${preflight[@]}" --replay "$train_replay"
run "${python[@]}" -m skill_rsi_rl.train --model "$SKILL_RSI_MODEL" --device cuda --dtype bf16 \
    --backend easyr1 --replay "$train_replay" --output "$run_dir/train" --epochs 1 \
    --per-device-batch-size "$batch_groups" --monitor "$monitor" --swanlab-name "${run_dir##*/}"
adapter="$run_dir/train/checkpoint-0001"
run test -s "$adapter/adapter_model.safetensors"
run "${python[@]}" -m skill_rsi_rl.dev_eval --model "$SKILL_RSI_MODEL" --device cuda:0 \
    --adapter "$adapter" --train-replay "$train_replay" --replay "$dev_replay" \
    --output "$run_dir/dev-score.json"
