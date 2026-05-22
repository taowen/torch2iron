#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export tiny Llama PyTorch modules for graph inspection."""

from __future__ import annotations

import argparse

import torch

from models.exported_llama3.pytorch_modules import (
    ExportLlamaDecodeModel,
    ExportLlamaPrefillModel,
    LlamaExportConfig,
    example_decode_args,
    example_prefill_args,
)


def export_program(
    config: LlamaExportConfig,
    mode: str,
) -> torch.export.ExportedProgram:
    if mode == "prefill":
        model = ExportLlamaPrefillModel(config).eval()
        args = example_prefill_args(config)
    elif mode == "decode":
        model = ExportLlamaDecodeModel(config).eval()
        args = example_decode_args(config)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    exported_program: torch.export.ExportedProgram = torch.export.export(model, args)
    return exported_program


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Inspect exported Llama graphs")
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
    config = LlamaExportConfig(
        vocab_size=args.vocab_size,
        emb_dim=args.heads * args.head_dim,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_groups=args.kv_groups,
        head_dim=args.head_dim,
        hidden_dim=args.hidden_dim,
        max_seq_len=args.max_seq_len,
        chunk_size=args.chunk_size,
    )
    exported_program = export_program(config, args.mode)
    print(exported_program.graph_module.code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
