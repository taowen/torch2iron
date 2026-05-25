#!/usr/bin/env python3
import argparse
from pathlib import Path


def build_mlir(tile_elements: int, total_elements: int, col: int, row: int) -> str:
    return f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile({col}, 0)
    %core = aie.tile({col}, {row})

    %in_ping = aie.buffer(%core) {{address = 1024 : i32, mem_bank = 0 : i32, sym_name = "in_ping"}} : memref<{tile_elements}xbf16>
    %in_pong = aie.buffer(%core) {{address = 16384 : i32, mem_bank = 1 : i32, sym_name = "in_pong"}} : memref<{tile_elements}xbf16>
    %out_ping = aie.buffer(%core) {{address = 32768 : i32, mem_bank = 2 : i32, sym_name = "out_ping"}} : memref<{tile_elements}xbf16>
    %out_pong = aie.buffer(%core) {{address = 49152 : i32, mem_bank = 3 : i32, sym_name = "out_pong"}} : memref<{tile_elements}xbf16>

    %in_empty = aie.lock(%core, 0) {{init = 2 : i32, sym_name = "in_empty"}}
    %in_full = aie.lock(%core, 1) {{init = 0 : i32, sym_name = "in_full"}}
    %out_empty = aie.lock(%core, 2) {{init = 2 : i32, sym_name = "out_empty"}}
    %out_full = aie.lock(%core, 3) {{init = 0 : i32, sym_name = "out_full"}}

    aie.flow(%shim, DMA : 0, %core, DMA : 0)
    aie.flow(%core, DMA : 0, %shim, DMA : 0)

    aie.runtime_sequence(%src: memref<{total_elements}xbf16>, %dst: memref<{total_elements}xbf16>) {{
      %0 = aiex.dma_configure_task_for @in_shim_alloc {{
        aie.dma_bd(%src : memref<{total_elements}xbf16>, 0, {total_elements}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {total_elements}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @out_shim_alloc {{
        aie.dma_bd(%dst : memref<{total_elements}xbf16>, 0, {total_elements}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {total_elements}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
    }}

    aie.shim_dma_allocation @in_shim_alloc(%shim, MM2S, 0)
    aie.shim_dma_allocation @out_shim_alloc(%shim, S2MM, 0)

    %mem = aie.mem(%core) {{
      %0 = aie.dma_start(S2MM, 0, ^in_ping_bd, ^after_in)
    ^in_ping_bd:
      aie.use_lock(%in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_ping : memref<{tile_elements}xbf16>, 0, {tile_elements}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%in_full, Release, 1)
      aie.next_bd ^in_pong_bd
    ^in_pong_bd:
      aie.use_lock(%in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_pong : memref<{tile_elements}xbf16>, 0, {tile_elements}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%in_full, Release, 1)
      aie.next_bd ^in_ping_bd
    ^after_in:
      %1 = aie.dma_start(MM2S, 0, ^out_ping_bd, ^done)
    ^out_ping_bd:
      aie.use_lock(%out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_ping : memref<{tile_elements}xbf16>, 0, {tile_elements}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%out_empty, Release, 1)
      aie.next_bd ^out_pong_bd
    ^out_pong_bd:
      aie.use_lock(%out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_pong : memref<{tile_elements}xbf16>, 0, {tile_elements}) {{bd_id = 3 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%out_empty, Release, 1)
      aie.next_bd ^out_ping_bd
    ^done:
      aie.end
    }}
  }}
}}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("build/core_tile_pingpong.mlir"))
    parser.add_argument("--tile-elements", type=int, default=256)
    parser.add_argument("--total-elements", type=int, default=4096)
    parser.add_argument("--col", type=int, default=0)
    parser.add_argument("--row", type=int, default=2)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build_mlir(args.tile_elements, args.total_elements, args.col, args.row),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()

