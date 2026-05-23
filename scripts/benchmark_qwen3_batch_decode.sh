#!/usr/bin/env bash

set -euo pipefail

PROMPT_LEN="${PROMPT_LEN:-16}"
NUM_TOKENS="${NUM_TOKENS:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"

export PROMPT_LEN
export NUM_TOKENS
export BATCH_SIZE

exec "$(dirname "${BASH_SOURCE[0]}")/run_exported_qwen3_batch_inference.sh" \
  --prompt "Count from 1 to 10:" \
  --prompt "The capital of France is" \
  "$@"
