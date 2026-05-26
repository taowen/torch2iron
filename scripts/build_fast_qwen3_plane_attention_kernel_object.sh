#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/iron_env.sh"

PEANO_DIR="$("${IRON_PYTHON}" - <<'PY'
import aie.utils.config as config
print(config.peano_install_dir())
PY
)"
MLIR_AIE_DIR="$("${IRON_PYTHON}" - <<'PY'
import aie.utils.config as config
print(config.root_path())
PY
)"

mkdir -p "${REPO_ROOT}/build"

exec "${PEANO_DIR}/bin/clang++" \
  -O2 \
  -std=c++20 \
  --target=aie2p-none-unknown-elf \
  -Wno-parentheses \
  -Wno-attributes \
  -Wno-macro-redefined \
  -Wno-empty-body \
  -Wno-missing-template-arg-list-after-template-kw \
  -I"${MLIR_AIE_DIR}/include" \
  -I"${MLIR_AIE_DIR}/aie_runtime_lib/AIE2P" \
  -DLLAMA_HEAD_DIM=128 \
  -DLLAMA_CHUNK_SIZE=16 \
  -DLLAMA_Q_HEADS_PER_GROUP=2 \
  -DLLAMA_ATTN_SCALE=0.08838834764831845f \
  -DLLAMA_VEC_SIZE=32 \
  -c "${REPO_ROOT}/src/models/fast_qwen3/operators/qwen_chunked_attention_current/kernel.cc" \
  -o "${REPO_ROOT}/build/qwen_plane_attention_current_hd128_tile16_qh2_kv32.o"
