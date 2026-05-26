#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an already-compiled static-KV contract xclbin through XRT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import aie.utils as aie_utils
import numpy as np
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a static-KV contract xclbin")
    parser.add_argument("--xclbin", type=Path, required=True)
    parser.add_argument("--insts", type=Path, required=True)
    parser.add_argument("--kernel", default="MLIR_AIE")
    parser.add_argument("--kv-cache-bytes", type=int, default=16 * 1024 * 1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kv_cache = XRTTensor((args.kv_cache_bytes,), dtype=np.uint8)
    kv_cache.data.fill(0)
    kv_cache.to("npu")

    kernel = NPUKernel(
        xclbin_path=args.xclbin,
        insts_path=args.insts,
        kernel_name=args.kernel,
    )
    handle = aie_utils.DefaultNPURuntime.load(kernel)
    result = aie_utils.DefaultNPURuntime.run(handle, [kv_cache])
    print(
        json.dumps(
            {
                "kernel": args.kernel,
                "kv_cache_bytes": args.kv_cache_bytes,
                "npu_time_ns": result.npu_time,
                "ret": str(result.ret),
                "success": result.is_success(),
                "xclbin": str(args.xclbin.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
