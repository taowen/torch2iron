#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import an AutoRound/AutoGPTQ Qwen3 checkpoint into packed W4A16."""

from __future__ import annotations

import argparse

from models.quantized_qwen3.model import find_model_dir
from models.quantized_qwen3.packed_format import write_packed_inference_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the packed W4A16 Qwen3 inference format")
    parser.add_argument("model_dir", help="AutoRound/AutoGPTQ checkpoint or parent output directory")
    parser.add_argument("--output-dir", default=None, help="Packed artifact directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = find_model_dir(args.model_dir)
    manifest = write_packed_inference_artifact(model_dir, args.output_dir)
    print(f"packed_total_bytes: {manifest['total_bytes']}")


if __name__ == "__main__":
    main()
