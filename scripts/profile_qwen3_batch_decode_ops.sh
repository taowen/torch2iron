#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "${REPO_ROOT}/scripts/iron_env.sh"

exec "${IRON_PYTHON}" "${REPO_ROOT}/scripts/profile_qwen3_batch_decode_ops.py" "$@"
