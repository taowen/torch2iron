#!/usr/bin/env python3
import argparse
from pathlib import Path


def ring_lock_decls(prefix: str, mlir_ids: tuple[int, int, int], fastflow_ids: tuple[int, int, int]) -> str:
    return "\n".join(
        [
            f"    // {prefix}: fastflow_reference_locks={fastflow_ids[0]},{fastflow_ids[1]},{fastflow_ids[2]}",
            f'    %{prefix}_empty = aie.lock(%mem, {mlir_ids[0]}) {{init = 2 : i32, sym_name = "{prefix}_empty"}}',
            f'    %{prefix}_loaded = aie.lock(%mem, {mlir_ids[1]}) {{init = 0 : i32, sym_name = "{prefix}_loaded"}}',
            f'    %{prefix}_split = aie.lock(%mem, {mlir_ids[2]}) {{init = 0 : i32, sym_name = "{prefix}_split"}}',
        ]
    )


def ring_buffers(prefix: str) -> str:
    return "\n".join(
        [
            f'    %{prefix}_plane_ping = aie.buffer(%mem) {{sym_name = "{prefix}_plane_ping"}} : memref<4096xbf16>',
            f'    %{prefix}_plane_pong = aie.buffer(%mem) {{sym_name = "{prefix}_plane_pong"}} : memref<4096xbf16>',
            f'    %{prefix}_half0_ping = aie.buffer(%mem) {{sym_name = "{prefix}_half0_ping"}} : memref<2048xbf16>',
            f'    %{prefix}_half0_pong = aie.buffer(%mem) {{sym_name = "{prefix}_half0_pong"}} : memref<2048xbf16>',
            f'    %{prefix}_half1_ping = aie.buffer(%mem) {{sym_name = "{prefix}_half1_ping"}} : memref<2048xbf16>',
            f'    %{prefix}_half1_pong = aie.buffer(%mem) {{sym_name = "{prefix}_half1_pong"}} : memref<2048xbf16>',
        ]
    )


def load_ring(prefix: str, channel: int, bd_ping: int, bd_pong: int) -> str:
    return f"""      %{prefix}_load = aie.dma_start(S2MM, {channel}, ^{prefix}_load_ping, ^{prefix}_after_load)
    ^{prefix}_load_ping:
      aie.use_lock(%{prefix}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_plane_ping : memref<4096xbf16>, 0, 4096) {{bd_id = {bd_ping} : i32, next_bd_id = {bd_pong} : i32}}
      aie.use_lock(%{prefix}_loaded, Release, 1)
      aie.next_bd ^{prefix}_load_pong
    ^{prefix}_load_pong:
      aie.use_lock(%{prefix}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_plane_pong : memref<4096xbf16>, 0, 4096) {{bd_id = {bd_pong} : i32, next_bd_id = {bd_ping} : i32}}
      aie.use_lock(%{prefix}_loaded, Release, 1)
      aie.next_bd ^{prefix}_load_ping
    ^{prefix}_after_load:
"""


def half0_ring(prefix: str, channel: int, bd_ping: int, bd_pong: int) -> str:
    return f"""      %{prefix}_half0 = aie.dma_start(MM2S, {channel}, ^{prefix}_half0_ping_bd, ^{prefix}_after_half0)
    ^{prefix}_half0_ping_bd:
      aie.use_lock(%{prefix}_loaded, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_half0_ping : memref<2048xbf16>, 0, 2048) {{bd_id = {bd_ping} : i32, next_bd_id = {bd_pong} : i32}}
      aie.use_lock(%{prefix}_split, Release, 1)
      aie.next_bd ^{prefix}_half0_pong_bd
    ^{prefix}_half0_pong_bd:
      aie.use_lock(%{prefix}_loaded, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_half0_pong : memref<2048xbf16>, 0, 2048) {{bd_id = {bd_pong} : i32, next_bd_id = {bd_ping} : i32}}
      aie.use_lock(%{prefix}_split, Release, 1)
      aie.next_bd ^{prefix}_half0_ping_bd
    ^{prefix}_after_half0:
"""


def half1_ring(prefix: str, channel: int, bd_ping: int, bd_pong: int) -> str:
    return f"""      %{prefix}_half1 = aie.dma_start(MM2S, {channel}, ^{prefix}_half1_ping_bd, ^{prefix}_after_half1)
    ^{prefix}_half1_ping_bd:
      aie.use_lock(%{prefix}_split, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_half1_ping : memref<2048xbf16>, 0, 2048) {{bd_id = {bd_ping} : i32, next_bd_id = {bd_pong} : i32}}
      aie.use_lock(%{prefix}_empty, Release, 1)
      aie.next_bd ^{prefix}_half1_pong_bd
    ^{prefix}_half1_pong_bd:
      aie.use_lock(%{prefix}_split, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_half1_pong : memref<2048xbf16>, 0, 2048) {{bd_id = {bd_pong} : i32, next_bd_id = {bd_ping} : i32}}
      aie.use_lock(%{prefix}_empty, Release, 1)
      aie.next_bd ^{prefix}_half1_ping_bd
    ^{prefix}_after_half1:
"""


def build_mlir() -> str:
    return f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %worker0 = aie.tile(0, 2)
    %worker1 = aie.tile(1, 2)

{ring_lock_decls("ring_a", (4, 5, 6), (64, 65, 66))}
{ring_lock_decls("ring_b", (7, 8, 9), (67, 68, 69))}

{ring_buffers("ring_a")}
{ring_buffers("ring_b")}

    aie.flow(%shim, DMA : 0, %mem, DMA : 0)
    aie.flow(%shim, DMA : 1, %mem, DMA : 1)
    aie.flow(%mem, DMA : 0, %worker0, DMA : 0)
    aie.flow(%mem, DMA : 1, %worker1, DMA : 0)

    %memdma = aie.memtile_dma(%mem) {{
{load_ring("ring_a", 0, 0, 1)}
{half0_ring("ring_a", 0, 2, 3)}
{half1_ring("ring_a", 1, 24, 25)}
{load_ring("ring_b", 1, 26, 27)}
{half0_ring("ring_b", 2, 4, 5)}
{half1_ring("ring_b", 3, 28, 29)}
      aie.end
    }}
  }}
}}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("build/memtile_kv_plane_ring.mlir"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_mlir(), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
