#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

setup_bench() {
    local py="${BENCH_PYTHON:-python3.13}"
    local tooling="$root/.venv-setup"
    # uv sync removes extraneous packages from its target environment, including uv itself.
    "$py" -m venv "$tooling"
    "$tooling/bin/python" -m pip install uv
    UV_PROJECT_ENVIRONMENT="$root/.venv-bench" "$tooling/bin/uv" sync \
        --project "$root/vendor/continual-learning-bench" --python "$py" \
        --frozen --no-dev --extra "${CLBENCH_EXTRA:-poker}"
    "$tooling/bin/uv" pip install --python "$root/.venv-bench/bin/python" --no-deps -e "$root"
}

case "${1:-}" in
    train)
        py="${TRAIN_PYTHON:-python3.11}"
        "$py" -m venv .venv-train
        py="$root/.venv-train/bin/python"
        "$py" -m pip install -U pip setuptools wheel
        "$py" -m pip install torch==2.6.0 --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
        "$py" -m pip install -r env/train.txt -c env/easyr1-h20-constraints.txt
        "$py" -m pip install --no-deps -e ./EasyR1 -e .
        ;;
    bench)
        setup_bench
        ;;
    dsh)
        if [[ ! -x "$root/.venv-bench/bin/nodeenv" ]]; then
            setup_bench
        fi
        # The official benchmark lock already includes nodeenv as a dependency of pre-commit.
        prefix="$root/.venv-dsh"
        if [[ ! -x "$prefix/bin/node" ]]; then
            "$root/.venv-bench/bin/nodeenv" --node=24.22.3 --prebuilt "$prefix"
        fi
        PATH="$prefix/bin:$PATH" "$prefix/bin/npm" install --global --prefix "$prefix" \
            --engine-strict '@deepseek-ai/dsh@0.2.2-rc.2'
        ;;
    *)
        echo "Usage: $0 train|bench|dsh" >&2
        exit 2 ;;
esac
