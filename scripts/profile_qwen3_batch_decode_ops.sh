#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IRON_PYTHON="${IRON_PYTHON:-/home/taowen/projects/IRON/.venv/bin/python}"
XRT_ROOT="${XRT_ROOT:-/home/taowen/projects/xdna-driver/xrt/build/Release/opt/xilinx/xrt}"
XRT_PYTHONPATH="${XRT_PYTHONPATH:-${XRT_ROOT}/python}"

export PYTHONPATH="${REPO_ROOT}/src:${XRT_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${XRT_ROOT}/lib:${XRT_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec "${IRON_PYTHON}" "${REPO_ROOT}/scripts/profile_qwen3_batch_decode_ops.py" "$@"
