#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="${BENCH_PYTHON:-$root/.venv-bench/bin/python}"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
exec "$py" -B -m skill_rsi_rl.evaluate --harness dsh "$@"
