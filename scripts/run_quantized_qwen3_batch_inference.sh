#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "${REPO_ROOT}/scripts/iron_env.sh"

DEFAULT_MODEL_DIR="/home/taowen/models/qwen3-0.6b-w4a16-autogptq-script-smoke/c1899de289a04d12100db370d81485cdf75e47ca-w4g128"
MODEL_DIR="${MODEL_DIR:-${DEFAULT_MODEL_DIR}}"
PROMPT_LEN="${PROMPT_LEN:-16}"
NUM_TOKENS="${NUM_TOKENS:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"

require_path "quantized Qwen3 model directory" "${MODEL_DIR}"

args=(
    "${MODEL_DIR}"
    --prompt-len "${PROMPT_LEN}"
    --num-tokens "${NUM_TOKENS}"
    --batch-size "${BATCH_SIZE}"
)

exec "${IRON_PYTHON}" -m models.quantized_qwen3.qwen_npu "${args[@]}" "$@"
