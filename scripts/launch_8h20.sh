#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
python="${SKILL_RSI_PYTHON:-$root/.venv-train/bin/python}"
if [[ -z "${SKILL_RSI_PYTHON:-}" && ! -x "$python" ]]; then python=python3; fi
print_command=false
case "${1:-}" in
    --print-command) print_command=true; shift ;;
    -h|--help) exec "$python" -m skill_rsi_rl.train "$@" ;;
esac

command=(
    "$python" -m torch.distributed.run
    --nproc-per-node="${SKILL_RSI_NPROC_PER_NODE:-8}"
    --master-addr=127.0.0.1 --master-port="${SKILL_RSI_MASTER_PORT:-29500}"
    -m -- skill_rsi_rl.train
    --device cuda --dtype bf16
    --backend "${SKILL_RSI_BACKEND:-easyr1}"
    --model "${SKILL_RL_MODEL:-${SKILL_RSI_MODEL:-${MODEL_DIR:-$root/models/Qwen3-8B}}}"
    "$@"
)
if "$print_command"; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
