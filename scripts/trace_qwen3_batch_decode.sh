#!/usr/bin/env bash

set -euo pipefail

TRACE_SIZE="${TRACE_SIZE:-1048576}"
TRACE_OP_INDEX="${TRACE_OP_INDEX:-0}"
TRACE_DDR_ID="${TRACE_DDR_ID:-}"
TRACE_DIR="${TRACE_DIR:-build_trace}"
NUM_TOKENS="${NUM_TOKENS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PROMPT_LEN="${PROMPT_LEN:-16}"

export TORCH2IRON_TRACE_SIZE="${TRACE_SIZE}"
export TORCH2IRON_TRACE_OP_INDEX="${TRACE_OP_INDEX}"
export TORCH2IRON_TRACE_DIR="${TRACE_DIR}"
if [[ -n "${TRACE_DDR_ID}" ]]; then
  export TORCH2IRON_TRACE_DDR_ID="${TRACE_DDR_ID}"
fi
export NUM_TOKENS
export BATCH_SIZE
export PROMPT_LEN

exec "$(dirname "${BASH_SOURCE[0]}")/run_exported_qwen3_batch_inference.sh" "$@"
