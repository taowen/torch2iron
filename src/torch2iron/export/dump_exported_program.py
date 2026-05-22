#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a model package's tiny PyTorch modules for graph inspection."""

from __future__ import annotations

import argparse

from torch2iron.export.model_tools import (
    DEFAULT_MODEL_PACKAGE,
    export_program,
    load_model_export_tools,
    make_llama_export_config,
)


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Inspect torch.export graphs")
    parser.add_argument(
        "--model-package",
        default=DEFAULT_MODEL_PACKAGE,
        help="Model package containing pytorch_modules.py and runtime_config.py.",
    )
    parser.add_argument(
        "--mode",
        choices=("prefill", "decode"),
        default="decode",
        help="Which exported graph to print.",
    )
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-groups", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    tools = load_model_export_tools(args.model_package)
    config = make_llama_export_config(
        tools,
        vocab_size=args.vocab_size,
        layers=args.layers,
        heads=args.heads,
        kv_groups=args.kv_groups,
        head_dim=args.head_dim,
        hidden_dim=args.hidden_dim,
        max_seq_len=args.max_seq_len,
        chunk_size=args.chunk_size,
    )
    exported_program = export_program(tools, config, args.mode)
    print(exported_program.graph_module.code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
