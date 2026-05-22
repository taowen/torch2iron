#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IRON_PYTHON="${IRON_PYTHON:-/home/taowen/projects/IRON/.venv/bin/python}"
XRT_ROOT="${XRT_ROOT:-/home/taowen/projects/xdna-driver/xrt/build/Release/opt/xilinx/xrt}"
XRT_PYTHONPATH="${XRT_PYTHONPATH:-${XRT_ROOT}/python}"

MODEL_DIR="${MODEL_DIR:-/home/taowen/models/llama3.2-1b}"
WEIGHTS_PATH="${WEIGHTS_PATH:-${MODEL_DIR}/model.safetensors}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_DIR}/tokenizer.model}"
PACKED_WEIGHTS_DIR="${PACKED_WEIGHTS_DIR:-${MODEL_DIR}/llama_iron_packed}"

PROMPT_LEN="${PROMPT_LEN:-16}"
NUM_TOKENS="${NUM_TOKENS:-2}"
REQUIRE_PACKED_WEIGHTS="${REQUIRE_PACKED_WEIGHTS:-1}"
PREPARE_WEIGHTS="${PREPARE_WEIGHTS:-0}"

require_path() {
    local label="$1"
    local path="$2"
    if [[ ! -e "${path}" ]]; then
        echo "missing ${label}: ${path}" >&2
        exit 1
    fi
}

require_path "IRON python" "${IRON_PYTHON}"
require_path "XRT python directory" "${XRT_PYTHONPATH}"
require_path "weights" "${WEIGHTS_PATH}"
require_path "tokenizer" "${TOKENIZER_PATH}"

if [[ "${REQUIRE_PACKED_WEIGHTS}" == "1" ]]; then
    require_path "packed weights directory" "${PACKED_WEIGHTS_DIR}"
fi

export PYTHONPATH="${REPO_ROOT}/src:${XRT_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${XRT_ROOT}/lib:${XRT_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

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

exec "${IRON_PYTHON}" -m models.exported_llama3.llama_npu "${args[@]}" "$@"
