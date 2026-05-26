#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate and validate the Qwen3 layer static-KV K/V-pair attention ingress contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.fast_qwen3.operators.qwen3_layer_fused.static_kv_reader import (
    build_static_kv_pair_attention_ingress_contract_mlir,
    check_static_kv_pair_attention_ingress_contract_mlir,
    history_length_dwords,
    history_tiles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Qwen3LayerFusedMLIROperator static KV K/V-pair attention ingress contract"
    )
    parser.add_argument("--attend-seq-len", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "build/fast_qwen3_qwen3_layer_static_kv_pair_attention_ingress_contract.mlir"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = build_static_kv_pair_attention_ingress_contract_mlir(
        args.attend_seq_len,
        args.tile_size,
    )
    messages = check_static_kv_pair_attention_ingress_contract_mlir(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "attend_seq_len": args.attend_seq_len,
                "history_length_dwords": history_length_dwords(
                    args.attend_seq_len,
                    args.tile_size,
                ),
                "history_tiles": history_tiles(args.attend_seq_len, args.tile_size),
                "messages": messages,
                "output": str(args.output.resolve()),
                "tile_size": args.tile_size,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
