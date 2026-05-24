#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${REPO_ROOT:-}" ]]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

IRON_ROOT="${IRON_ROOT:-${REPO_ROOT}/third_party/IRON}"
if [[ -z "${IRON_PYTHON:-}" ]]; then
    if [[ -x "${IRON_ROOT}/.venv/bin/python" ]]; then
        IRON_PYTHON="${IRON_ROOT}/.venv/bin/python"
    else
        IRON_PYTHON="${REPO_ROOT}/.venv/bin/python"
    fi
fi

find_xrt_root() {
    local candidate
    for candidate in \
        "${XILINX_XRT:-}" \
        "/var/opt/xilinx/xrt" \
        "/opt/xilinx/xrt" \
        "/home/taowen/projects/xdna-driver/xrt/build/Release/opt/xilinx/xrt"; do
        if [[ -n "${candidate}" && -d "${candidate}/python" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

XRT_ROOT="${XRT_ROOT:-$(find_xrt_root)}"
XRT_PYTHONPATH="${XRT_PYTHONPATH:-${XRT_ROOT}/python}"

require_path() {
    local label="$1"
    local path="$2"
    if [[ ! -e "${path}" ]]; then
        echo "missing ${label}: ${path}" >&2
        exit 1
    fi
}

require_executable() {
    local label="$1"
    local command_path="$2"
    if [[ "${command_path}" == */* ]]; then
        if [[ ! -x "${command_path}" ]]; then
            echo "missing ${label}: ${command_path}" >&2
            exit 1
        fi
    elif ! command -v "${command_path}" >/dev/null 2>&1; then
        echo "missing ${label}: ${command_path}" >&2
        exit 1
    fi
}

require_path "IRON checkout" "${IRON_ROOT}"
require_path "XRT python directory" "${XRT_PYTHONPATH}"
require_executable "Python runtime" "${IRON_PYTHON}"

export PYTHONPATH="${REPO_ROOT}/src:${IRON_ROOT}:${XRT_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${XRT_ROOT}/lib:${XRT_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
