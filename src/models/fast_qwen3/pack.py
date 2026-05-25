#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the Fast Qwen3 Q4NX-style packed artifact."""

from __future__ import annotations

import argparse

from models.fast_qwen3.fast_packed_format import write_fast_qwen3_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack Qwen3 weights for fast fused layer kernels")
    parser.add_argument("model_dir", help="Qwen3 AutoGPTQ/AutoRound checkpoint or parent directory")
    parser.add_argument("--output-dir", default=None, help="Output artifact directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = write_fast_qwen3_artifact(args.model_dir, args.output_dir)
    print(f"fast_qwen3_format: {manifest['format']}")
    print(f"fast_qwen3_total_bytes: {manifest['total_bytes']}")


if __name__ == "__main__":
    main()

