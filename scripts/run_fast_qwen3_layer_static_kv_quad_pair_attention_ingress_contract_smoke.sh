#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/iron_env.sh"

exec "${IRON_PYTHON}" -m models.fast_qwen3.qwen3_layer_static_kv_quad_pair_attention_ingress_contract_smoke "$@"
