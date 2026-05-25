#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


EXPECTED_BD = {
    0: 4096,
    1: 4096,
    2: 2048,
    3: 2048,
    24: 2048,
    25: 2048,
    26: 4096,
    27: 4096,
    4: 2048,
    5: 2048,
    28: 2048,
    29: 2048,
}

EXPECTED_MLIR_LOCKS = {
    "ring_a": (4, 5, 6),
    "ring_b": (7, 8, 9),
}

EXPECTED_FASTFLOW_LOCK_COMMENTS = {
    "ring_a": (64, 65, 66),
    "ring_b": (67, 68, 69),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mlir", type=Path)
    args = parser.parse_args()
    text = args.mlir.read_text(encoding="utf-8")

    require("aie.memtile_dma(%mem)" in text, "missing memtile DMA region")
    for prefix, locks in EXPECTED_MLIR_LOCKS.items():
        for lock_id in locks:
            require(f"aie.lock(%mem, {lock_id})" in text, f"missing {prefix} MLIR lock {lock_id}")
    for prefix, locks in EXPECTED_FASTFLOW_LOCK_COMMENTS.items():
        comment = f"// {prefix}: fastflow_reference_locks={locks[0]},{locks[1]},{locks[2]}"
        require(comment in text, f"missing {prefix} FastFlow lock annotation")

    rows = re.findall(r"memref<(\d+)xbf16>, 0, (\d+)\).*bd_id = (\d+) : i32, next_bd_id = (\d+) : i32", text)
    found = {int(bd_id): int(length) for _memref_len, length, bd_id, _next_bd in rows}
    require(found == EXPECTED_BD, f"unexpected BD length table: {found}")

    expected_pairs = {
        0: 1,
        1: 0,
        2: 3,
        3: 2,
        24: 25,
        25: 24,
        26: 27,
        27: 26,
        4: 5,
        5: 4,
        28: 29,
        29: 28,
    }
    pairs = {int(bd_id): int(next_bd) for _memref_len, _length, bd_id, next_bd in rows}
    require(pairs == expected_pairs, f"unexpected next-BD table: {pairs}")

    print("memtile KV plane ring check passed")
    print("ring_a mlir_locks=4->5->6->4 fastflow_ref=64->65->66->64 bd=0/1,2/3,24/25")
    print("ring_b mlir_locks=7->8->9->7 fastflow_ref=67->68->69->67 bd=26/27,4/5,28/29")


if __name__ == "__main__":
    main()
