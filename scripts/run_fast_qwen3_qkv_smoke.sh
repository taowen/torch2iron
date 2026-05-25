#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/iron_env.sh"

MODEL_DIR="${MODEL_DIR:-/home/taowen/models/qwen3-0.6b-w4a16-autogptq-script-smoke}"
LAYER="${LAYER:-0}"
ROWS="${ROWS:-1}"

exec "${IRON_PYTHON}" -m models.fast_qwen3.qkv_smoke "${MODEL_DIR}" --layer "${LAYER}" --rows "${ROWS}" "$@"
