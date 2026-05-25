#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/iron_env.sh"

MODEL_DIR="${FAST_QWEN3_MODEL_DIR:-/home/taowen/models/qwen3-0.6b-w4a16-autogptq-script-smoke}"

exec "${IRON_PYTHON}" -m models.fast_qwen3.o_projection_residual_operator_smoke "${MODEL_DIR}" "$@"
