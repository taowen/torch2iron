#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "${REPO_ROOT}/scripts/iron_env.sh"

DEFAULT_MODEL_DIR="/home/taowen/models/qwen3-0.6b"
if [[ ! -e "${DEFAULT_MODEL_DIR}" ]]; then
    DEFAULT_MODEL_DIR="/home/taowen/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
fi

MODEL_DIR="${MODEL_DIR:-${DEFAULT_MODEL_DIR}}"
WEIGHTS_PATH="${WEIGHTS_PATH:-${MODEL_DIR}/model.safetensors}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_DIR}}"
PACKED_WEIGHTS_DIR="${PACKED_WEIGHTS_DIR:-${MODEL_DIR}/qwen_iron_packed}"

PROMPT_LEN="${PROMPT_LEN:-16}"
NUM_TOKENS="${NUM_TOKENS:-2}"
REQUIRE_PACKED_WEIGHTS="${REQUIRE_PACKED_WEIGHTS:-0}"
PREPARE_WEIGHTS="${PREPARE_WEIGHTS:-0}"

require_path "weights" "${WEIGHTS_PATH}"
require_path "tokenizer" "${TOKENIZER_PATH}"

if [[ "${REQUIRE_PACKED_WEIGHTS}" == "1" ]]; then
    require_path "packed weights directory" "${PACKED_WEIGHTS_DIR}"
fi

args=(
    "${WEIGHTS_PATH}"
    "${TOKENIZER_PATH}"
    --prompt-len "${PROMPT_LEN}"
    --num-tokens "${NUM_TOKENS}"
    --packed-weights-dir "${PACKED_WEIGHTS_DIR}"
)

if [[ "${REQUIRE_PACKED_WEIGHTS}" == "1" ]]; then
    args+=(--require-packed-weights)
fi

if [[ "${PREPARE_WEIGHTS}" == "1" ]]; then
    args+=(--prepare-weights)
fi

exec "${IRON_PYTHON}" -m models.exported_qwen3.qwen_npu "${args[@]}" "$@"
