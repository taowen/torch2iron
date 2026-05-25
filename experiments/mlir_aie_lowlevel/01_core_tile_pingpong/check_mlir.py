#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


REQUIRED = [
    "aie.mem(%core)",
    "aie.dma_start(S2MM, 0",
    "aie.dma_start(MM2S, 0",
    "aie.use_lock(%in_empty, AcquireGreaterEqual, 1)",
    "aie.use_lock(%in_full, Release, 1)",
    "aie.use_lock(%out_full, AcquireGreaterEqual, 1)",
    "aie.use_lock(%out_empty, Release, 1)",
    "aiex.dma_configure_task_for @in_shim_alloc",
    "aiex.dma_configure_task_for @out_shim_alloc",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mlir", type=Path)
    args = parser.parse_args()
    text = args.mlir.read_text(encoding="utf-8")

    for item in REQUIRED:
        require(item in text, f"missing required MLIR fragment: {item}")

    bd_pairs = re.findall(r"bd_id = (\d+) : i32, next_bd_id = (\d+) : i32", text)
    require(bd_pairs == [("0", "1"), ("1", "0"), ("2", "3"), ("3", "2")], f"unexpected BD ring: {bd_pairs}")

    print("core tile ping-pong check passed")
    print("bd_ring=input 0->1->0, output 2->3->2")


if __name__ == "__main__":
    main()
