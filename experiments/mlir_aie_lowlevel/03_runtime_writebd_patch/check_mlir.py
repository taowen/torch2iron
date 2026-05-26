#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


EXPECTED = {
    "k03": (0x000000, 12),
    "v03": (0x400000, 13),
    "k47": (0x800000, 14),
    "v47": (0xC00000, 15),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mlir", type=Path)
    args = parser.parse_args()
    text = args.mlir.read_text(encoding="utf-8")

    require("aie.runtime_sequence" in text, "missing runtime_sequence")
    require(text.count("aiex.npu.writebd") == 4, "expected exactly four npu.writebd ops")

    for name, (offset, bd_id) in EXPECTED.items():
        require(f"// {name}:" in text, f"missing plane comment {name}")
        pattern = rf"// {name}:.*?bd_id = (\d+) : i32, buffer_length = (\d+) : i32, buffer_offset = (\d+) : i32"
        match = re.search(pattern, text, flags=re.S)
        require(match is not None, f"missing writebd fields for {name}")
        found_bd, found_len, found_offset = map(int, match.groups())
        require(found_bd == bd_id, f"{name} bd_id {found_bd} != {bd_id}")
        require(found_offset == offset, f"{name} offset {found_offset} != {offset}")
        require(found_len % 0x1000 == 0, f"{name} length should be 16-token tile aligned")

    require("use_next_bd = 0 : i32" in text, "dynamic shim descriptor should not chain by default")
    print("runtime writebd patch check passed")
    print("planes=k03,v03,k47,v47; length is ceil(L/16)*0x1000 dwords")


if __name__ == "__main__":
    main()
