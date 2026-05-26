#!/usr/bin/env python3
import argparse
from pathlib import Path


PLANES = [
    ("k03", 0x000000, 12),
    ("v03", 0x400000, 13),
    ("k47", 0x800000, 14),
    ("v47", 0xC00000, 15),
]


def writebd(name: str, base_bytes: int, bd_id: int, length_dwords: int, token_index: int) -> str:
    return f"""      // {name}: base_bytes=0x{base_bytes:06x}, length_dwords={length_dwords}
      aiex.npu.writebd {{bd_id = {bd_id} : i32, buffer_length = {length_dwords} : i32, buffer_offset = {base_bytes} : i32, burst_length = 64 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = {token_index} : i32, packet_id = {token_index} : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}"""


def build_mlir(length_dwords: int) -> str:
    patches = "\n".join(writebd(name, base, bd_id, length_dwords, index) for index, (name, base, bd_id) in enumerate(PLANES))
    return f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile(0, 0)

    aie.runtime_sequence(%kv_cache: memref<4194304xbf16>) {{
{patches}
    }}
  }}
}}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("build/runtime_writebd_patch.mlir"))
    parser.add_argument("--history-tiles", type=int, default=1)
    args = parser.parse_args()

    length_dwords = args.history_tiles * 0x1000
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_mlir(length_dwords), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
