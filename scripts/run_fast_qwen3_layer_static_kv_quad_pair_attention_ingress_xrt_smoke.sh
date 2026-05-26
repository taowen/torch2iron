#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/iron_env.sh"

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-10}"
MLIR="${REPO_ROOT}/build/fast_qwen3_qwen3_layer_static_kv_quad_pair_attention_ingress_contract.mlir"
XCLBIN="${REPO_ROOT}/build/fast_qwen3_qwen3_layer_static_kv_quad_pair_attention_ingress_contract_run.xclbin"
INSTS="${REPO_ROOT}/build/fast_qwen3_qwen3_layer_static_kv_quad_pair_attention_ingress_contract_run.bin"

"${REPO_ROOT}/scripts/build_fast_qwen3_plane_attention_kernel_object.sh"
"${REPO_ROOT}/scripts/run_fast_qwen3_layer_static_kv_quad_pair_attention_ingress_contract_smoke.sh" \
  --attend-seq-len 128 \
  --tile-size 16 \
  --output "${MLIR}" >/dev/null

MLIR="${MLIR}" XCLBIN="${XCLBIN}" INSTS="${INSTS}" "${IRON_PYTHON}" - <<'PY'
import os
import subprocess
from pathlib import Path

import aie.utils.config as config

env = os.environ.copy()
candidates = [
    Path(env.get("XILINX_XRT", "")) / "bin",
    Path("/var/opt/xilinx/xrt/bin"),
    Path("/opt/xilinx/xrt/bin"),
    Path("/var/home/taowen/projects/xdna-driver/xrt/build/Release/opt/xilinx/xrt/bin"),
]
xrt_bins = [str(path) for path in candidates if path.is_dir()]
if xrt_bins:
    env["PATH"] = os.pathsep.join(xrt_bins + [env.get("PATH", "")])

cmd = [
    str(Path(config.root_path()) / "bin" / "aiecc"),
    "-v",
    "-j1",
    "--no-compile-host",
    "--no-xchesscc",
    "--no-xbridge",
    "--peano",
    str(Path(config.peano_install_dir())),
    "--dynamic-objFifos",
    "--aie-generate-xclbin",
    f"--xclbin-name={Path(os.environ['XCLBIN']).resolve()}",
    "--xclbin-kernel-name=MLIR_AIE",
    "--aie-generate-npu-insts",
    f"--npu-insts-name={Path(os.environ['INSTS']).resolve()}",
    str(Path(os.environ["MLIR"]).resolve()),
]
subprocess.run(cmd, check=True, env=env)
PY

exec timeout "${TIMEOUT_SECONDS}s" "${IRON_PYTHON}" -m models.fast_qwen3.qwen3_layer_static_kv_xrt_smoke \
  --xclbin "${XCLBIN}" \
  --insts "${INSTS}" \
  "$@"
