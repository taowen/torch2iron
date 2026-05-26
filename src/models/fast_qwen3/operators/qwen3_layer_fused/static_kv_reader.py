#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static KV reader contract for ``Qwen3LayerFusedMLIROperator``.

This module keeps the FastFlowLM-style row1 memtile ring and runtime shim BD
patching in the fast_qwen3 code path.  It is intentionally text-level MLIR for
now because this is the part that high-level IRON ObjectFIFO/Runtime.fill should
not own in the final layer engine.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


KV_TILE_SIZE = 16
KV_TILE_LENGTH_DWORDS = 0x1000
Q_HEADS_PER_GROUP = 2
HEAD_DIM = 128
Q_ELEMENTS_PER_GROUP = Q_HEADS_PER_GROUP * HEAD_DIM
Q_CURRENT_ELEMENTS_PER_GROUP = Q_ELEMENTS_PER_GROUP + 2 * HEAD_DIM
PLANE_GROUP_PAIR_CHUNK_ELEMENTS = 2 * KV_TILE_SIZE * HEAD_DIM
STATE_ELEMENTS = Q_HEADS_PER_GROUP * 2
ATTENTION_STACK_SIZE = 0xD00
ATTENTION_KERNEL_OBJECT = (
    "qwen_plane_attention_current_hd128_tile16_qh2_kv32.o"
)


@dataclass(frozen=True)
class PlaneDescriptor:
    name: str
    base_bytes: int
    shim_bd_id: int
    packet_id: int


@dataclass(frozen=True)
class RingDescriptor:
    name: str
    mlir_locks: tuple[int, int, int]
    fastflow_locks: tuple[int, int, int]
    load_channel: int
    load_bds: tuple[int, int]
    half0_channel: int
    half0_bds: tuple[int, int]
    half1_channel: int
    half1_bds: tuple[int, int]


@dataclass(frozen=True)
class WorkerStreamDescriptor:
    group: int
    plane_pair: int
    group_in_plane: int
    key_plane: str
    value_plane: str
    q_split_tile: tuple[int, int]
    plane_split_tile: tuple[int, int]
    worker_tile: tuple[int, int]


@dataclass(frozen=True)
class FifoRouteDescriptor:
    name: str
    producer_tile: tuple[int, int]
    consumer_tile: tuple[int, int]
    elements: int
    depth: int


PLANES: tuple[PlaneDescriptor, ...] = (
    PlaneDescriptor("k03", 0x000000, 12, 0),
    PlaneDescriptor("v03", 0x400000, 13, 1),
    PlaneDescriptor("k47", 0x800000, 14, 2),
    PlaneDescriptor("v47", 0xC00000, 15, 3),
)

RINGS: tuple[RingDescriptor, ...] = (
    RingDescriptor("ring_a", (4, 5, 6), (64, 65, 66), 0, (0, 1), 0, (2, 3), 1, (24, 25)),
    RingDescriptor("ring_b", (7, 8, 9), (67, 68, 69), 1, (26, 27), 2, (4, 5), 3, (28, 29)),
)

WORKER_STREAMS: tuple[WorkerStreamDescriptor, ...] = tuple(
    WorkerStreamDescriptor(
        group=group,
        plane_pair=group // 4,
        group_in_plane=group % 4,
        key_plane="k03" if group < 4 else "k47",
        value_plane="v03" if group < 4 else "v47",
        q_split_tile=(1 if group < 4 else 5, 1),
        plane_split_tile=(0 if group < 4 else 4, 1),
        worker_tile=(group, 2),
    )
    for group in range(8)
)

EXPECTED_BD_LENGTHS: dict[int, int] = {
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

Q_ROUTES: tuple[FifoRouteDescriptor, ...] = tuple(
    FifoRouteDescriptor(
        name=f"qwen_layer_q_g{stream.group}",
        producer_tile=stream.q_split_tile,
        consumer_tile=stream.worker_tile,
        elements=Q_CURRENT_ELEMENTS_PER_GROUP,
        depth=1,
    )
    for stream in WORKER_STREAMS
)

KV_ROUTES: tuple[FifoRouteDescriptor, ...] = tuple(
    FifoRouteDescriptor(
        name=f"qwen_layer_kv_pair_mem_g{stream.group}",
        producer_tile=stream.plane_split_tile,
        consumer_tile=stream.worker_tile,
        elements=PLANE_GROUP_PAIR_CHUNK_ELEMENTS,
        depth=2,
    )
    for stream in WORKER_STREAMS
)

OUT_ROUTES: tuple[FifoRouteDescriptor, ...] = tuple(
    FifoRouteDescriptor(
        name=f"qwen_layer_context_g{stream.group}",
        producer_tile=stream.worker_tile,
        consumer_tile=stream.plane_split_tile,
        elements=Q_ELEMENTS_PER_GROUP,
        depth=1,
    )
    for stream in WORKER_STREAMS
)


def history_tiles(attend_seq_len: int, tile_size: int = 16) -> int:
    if attend_seq_len <= 0:
        raise ValueError("attend_seq_len must be positive")
    if tile_size != KV_TILE_SIZE:
        raise ValueError("static KV reader contract is fixed to tile_size=16")
    return (attend_seq_len + tile_size - 1) // tile_size


def history_length_dwords(attend_seq_len: int, tile_size: int = 16) -> int:
    return history_tiles(attend_seq_len, tile_size) * KV_TILE_LENGTH_DWORDS


def _ring_lock_decls(ring: RingDescriptor, mem_value: str = "%mem") -> str:
    empty, loaded, split = ring.mlir_locks
    ff_empty, ff_loaded, ff_split = ring.fastflow_locks
    return "\n".join(
        [
            f"    // {ring.name}: fastflow_reference_locks={ff_empty},{ff_loaded},{ff_split}",
            f'    %{ring.name}_empty = aie.lock({mem_value}, {empty}) {{init = 2 : i32, sym_name = "{ring.name}_empty"}}',
            f'    %{ring.name}_loaded = aie.lock({mem_value}, {loaded}) {{init = 0 : i32, sym_name = "{ring.name}_loaded"}}',
            f'    %{ring.name}_split = aie.lock({mem_value}, {split}) {{init = 0 : i32, sym_name = "{ring.name}_split"}}',
        ]
    )


def _ring_buffers(ring: RingDescriptor, mem_value: str = "%mem") -> str:
    return "\n".join(
        [
            f'    %{ring.name}_plane_ping = aie.buffer({mem_value}) {{sym_name = "{ring.name}_plane_ping"}} : memref<4096xbf16>',
            f'    %{ring.name}_plane_pong = aie.buffer({mem_value}) {{sym_name = "{ring.name}_plane_pong"}} : memref<4096xbf16>',
            f'    %{ring.name}_half0_ping = aie.buffer({mem_value}) {{sym_name = "{ring.name}_half0_ping"}} : memref<2048xbf16>',
            f'    %{ring.name}_half0_pong = aie.buffer({mem_value}) {{sym_name = "{ring.name}_half0_pong"}} : memref<2048xbf16>',
            f'    %{ring.name}_half1_ping = aie.buffer({mem_value}) {{sym_name = "{ring.name}_half1_ping"}} : memref<2048xbf16>',
            f'    %{ring.name}_half1_pong = aie.buffer({mem_value}) {{sym_name = "{ring.name}_half1_pong"}} : memref<2048xbf16>',
        ]
    )


def _load_ring(ring: RingDescriptor) -> str:
    bd_ping, bd_pong = ring.load_bds
    return f"""      %{ring.name}_load = aie.dma_start(S2MM, {ring.load_channel}, ^{ring.name}_load_ping, ^{ring.name}_after_load)
    ^{ring.name}_load_ping:
      aie.use_lock(%{ring.name}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{ring.name}_plane_ping : memref<4096xbf16>, 0, 4096) {{bd_id = {bd_ping} : i32, next_bd_id = {bd_pong} : i32}}
      aie.use_lock(%{ring.name}_loaded, Release, 1)
      aie.next_bd ^{ring.name}_load_pong
    ^{ring.name}_load_pong:
      aie.use_lock(%{ring.name}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%{ring.name}_plane_pong : memref<4096xbf16>, 0, 4096) {{bd_id = {bd_pong} : i32, next_bd_id = {bd_ping} : i32}}
      aie.use_lock(%{ring.name}_loaded, Release, 1)
      aie.next_bd ^{ring.name}_load_ping
    ^{ring.name}_after_load:
"""


def _half_ring(
    ring: RingDescriptor,
    half_name: str,
    channel: int,
    bds: tuple[int, int],
    acquire_lock: str,
    release_lock: str,
) -> str:
    bd_ping, bd_pong = bds
    return f"""      %{ring.name}_{half_name} = aie.dma_start(MM2S, {channel}, ^{ring.name}_{half_name}_ping_bd, ^{ring.name}_after_{half_name})
    ^{ring.name}_{half_name}_ping_bd:
      aie.use_lock(%{ring.name}_{acquire_lock}, AcquireGreaterEqual, 1)
      aie.dma_bd(%{ring.name}_{half_name}_ping : memref<2048xbf16>, 0, 2048) {{bd_id = {bd_ping} : i32, next_bd_id = {bd_pong} : i32}}
      aie.use_lock(%{ring.name}_{release_lock}, Release, 1)
      aie.next_bd ^{ring.name}_{half_name}_pong_bd
    ^{ring.name}_{half_name}_pong_bd:
      aie.use_lock(%{ring.name}_{acquire_lock}, AcquireGreaterEqual, 1)
      aie.dma_bd(%{ring.name}_{half_name}_pong : memref<2048xbf16>, 0, 2048) {{bd_id = {bd_pong} : i32, next_bd_id = {bd_ping} : i32}}
      aie.use_lock(%{ring.name}_{release_lock}, Release, 1)
      aie.next_bd ^{ring.name}_{half_name}_ping_bd
    ^{ring.name}_after_{half_name}:
"""


def _full_forward_ring(
    ring: RingDescriptor,
    forward_name: str,
    channel: int,
    bds: tuple[int, int],
    acquire_lock: str,
    release_lock: str,
) -> str:
    bd_ping, bd_pong = bds
    return f"""      %{ring.name}_{forward_name} = aie.dma_start(MM2S, {channel}, ^{ring.name}_{forward_name}_ping_bd, ^{ring.name}_after_{forward_name})
    ^{ring.name}_{forward_name}_ping_bd:
      aie.use_lock(%{ring.name}_{acquire_lock}, AcquireGreaterEqual, 1)
      aie.dma_bd(%{ring.name}_plane_ping : memref<4096xbf16>, 0, 4096) {{bd_id = {bd_ping} : i32, next_bd_id = {bd_pong} : i32}}
      aie.use_lock(%{ring.name}_{release_lock}, Release, 1)
      aie.next_bd ^{ring.name}_{forward_name}_pong_bd
    ^{ring.name}_{forward_name}_pong_bd:
      aie.use_lock(%{ring.name}_{acquire_lock}, AcquireGreaterEqual, 1)
      aie.dma_bd(%{ring.name}_plane_pong : memref<4096xbf16>, 0, 4096) {{bd_id = {bd_pong} : i32, next_bd_id = {bd_ping} : i32}}
      aie.use_lock(%{ring.name}_{release_lock}, Release, 1)
      aie.next_bd ^{ring.name}_{forward_name}_ping_bd
    ^{ring.name}_after_{forward_name}:
"""


def _writebd(plane: PlaneDescriptor, length_dwords: int, column: int = 0) -> str:
    return f"""      // {plane.name}: base_bytes=0x{plane.base_bytes:06x}, length_dwords={length_dwords}
      aiex.npu.writebd {{bd_id = {plane.shim_bd_id} : i32, buffer_length = {length_dwords} : i32, buffer_offset = {plane.base_bytes} : i32, burst_length = 64 : i32, column = {column} : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = {plane.packet_id} : i32, packet_id = {plane.packet_id} : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}"""


def _worker_contract_comments() -> str:
    lines = ["    // worker-facing attention stream contract:"]
    for stream in WORKER_STREAMS:
        q_col, q_row = stream.q_split_tile
        p_col, p_row = stream.plane_split_tile
        w_col, w_row = stream.worker_tile
        lines.append(
            "    // "
            f"worker_g{stream.group}: "
            f"planes={stream.key_plane}/{stream.value_plane} "
            f"pair={stream.plane_pair} group_in_plane={stream.group_in_plane} "
            f"q_split=c{q_col}r{q_row} plane_split=c{p_col}r{p_row} "
            f"worker=c{w_col}r{w_row}"
        )
    return "\n".join(lines)


def _worker_tile_decls() -> str:
    tiles = {
        "q_split_p0": (1, 1),
        "q_split_p1": (5, 1),
        "plane_split_p1": (4, 1),
    }
    for stream in WORKER_STREAMS:
        tiles[f"worker_g{stream.group}"] = stream.worker_tile
    return "\n".join(
        f"    %{name} = aie.tile({col}, {row})" for name, (col, row) in tiles.items()
    )


def _attention_kernel_decls() -> str:
    return f"""    func.func private @llama_chunked_attention_init_f32(memref<{STATE_ELEMENTS}xf32>, memref<{Q_ELEMENTS_PER_GROUP}xf32>, i32, i32) attributes {{link_with = "{ATTENTION_KERNEL_OBJECT}"}}
    func.func private @qwen_plane_group_attention_update_bf16(memref<{Q_CURRENT_ELEMENTS_PER_GROUP}xbf16>, memref<{PLANE_GROUP_PAIR_CHUNK_ELEMENTS}xbf16>, memref<{STATE_ELEMENTS}xf32>, memref<{Q_ELEMENTS_PER_GROUP}xf32>, i32, i32, i32, i32, i32) attributes {{link_with = "{ATTENTION_KERNEL_OBJECT}"}}
    func.func private @llama_chunked_attention_finalize_bf16(memref<{STATE_ELEMENTS}xf32>, memref<{Q_ELEMENTS_PER_GROUP}xf32>, memref<{Q_ELEMENTS_PER_GROUP}xbf16>, i32, i32) attributes {{link_with = "{ATTENTION_KERNEL_OBJECT}"}}"""


def _split_attention_kernel_decls() -> str:
    return f"""    func.func private @llama_chunked_attention_init_f32(memref<{STATE_ELEMENTS}xf32>, memref<{Q_ELEMENTS_PER_GROUP}xf32>, i32, i32) attributes {{link_with = "{ATTENTION_KERNEL_OBJECT}"}}
    func.func private @qwen_plane_group_attention_update_split_bf16(memref<{Q_CURRENT_ELEMENTS_PER_GROUP}xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<{STATE_ELEMENTS}xf32>, memref<{Q_ELEMENTS_PER_GROUP}xf32>, i32, i32, i32, i32, i32) attributes {{link_with = "{ATTENTION_KERNEL_OBJECT}"}}"""


def _bounded_attention_kernel_decls() -> str:
    return f"""    func.func private @qwen_zero_q_current_bf16(memref<{Q_CURRENT_ELEMENTS_PER_GROUP}xbf16>, i32) attributes {{link_with = "{ATTENTION_KERNEL_OBJECT}"}}
    func.func private @llama_chunked_attention_init_f32(memref<{STATE_ELEMENTS}xf32>, memref<{Q_ELEMENTS_PER_GROUP}xf32>, i32, i32) attributes {{link_with = "{ATTENTION_KERNEL_OBJECT}"}}
    func.func private @qwen_plane_group_attention_update_split_bf16(memref<{Q_CURRENT_ELEMENTS_PER_GROUP}xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<{STATE_ELEMENTS}xf32>, memref<{Q_ELEMENTS_PER_GROUP}xf32>, i32, i32, i32, i32, i32) attributes {{link_with = "{ATTENTION_KERNEL_OBJECT}"}}
    func.func private @llama_chunked_attention_finalize_bf16(memref<{STATE_ELEMENTS}xf32>, memref<{Q_ELEMENTS_PER_GROUP}xf32>, memref<{Q_ELEMENTS_PER_GROUP}xbf16>, i32, i32) attributes {{link_with = "{ATTENTION_KERNEL_OBJECT}"}}"""


def _tile_value_name(tile: tuple[int, int]) -> str:
    if tile == (1, 1):
        return "%q_split_p0"
    if tile == (5, 1):
        return "%q_split_p1"
    if tile == (0, 1):
        return "%mem"
    if tile == (4, 1):
        return "%plane_split_p1"
    for stream in WORKER_STREAMS:
        if tile == stream.worker_tile:
            return f"%worker_g{stream.group}"
    raise ValueError(f"unknown tile {tile}")


def _fifo_route_comments(routes: tuple[FifoRouteDescriptor, ...]) -> str:
    lines = []
    for route in routes:
        producer = _tile_value_name(route.producer_tile)
        consumer = _tile_value_name(route.consumer_tile)
        lines.append(
            "    // "
            f"fifo_route {route.name}: {producer} -> {consumer}, "
            f"elements={route.elements}, depth={route.depth}"
        )
    return "\n".join(lines)


def _attention_fifo_decls() -> str:
    return "\n".join(
        [
            "    // worker FIFO boundary contract: q_current, static KV tile, context output",
            "    // Do not lower the KV routes to high-level aie.objectfifo here: combining",
            "    // c0r1 objectfifo fanout with the explicit static memtile DMA ring exceeds",
            "    // memtile output DMA resources. The final layer engine should keep this",
            "    // boundary as explicit BD/lock/stream routes.",
            _fifo_route_comments(Q_ROUTES),
            _fifo_route_comments(KV_ROUTES),
            _fifo_route_comments(OUT_ROUTES),
        ]
    )


def _attention_worker_buffers() -> str:
    blocks = []
    for stream in WORKER_STREAMS:
        group = stream.group
        blocks.append(
            f"""    %q_current_g{group} = aie.buffer(%worker_g{group}) {{sym_name = "q_current_g{group}"}} : memref<{Q_CURRENT_ELEMENTS_PER_GROUP}xbf16>
    %plane_pair_g{group} = aie.buffer(%worker_g{group}) {{sym_name = "plane_pair_g{group}"}} : memref<{PLANE_GROUP_PAIR_CHUNK_ELEMENTS}xbf16>
    %state_g{group} = aie.buffer(%worker_g{group}) {{sym_name = "state_g{group}"}} : memref<{STATE_ELEMENTS}xf32>
    %acc_g{group} = aie.buffer(%worker_g{group}) {{sym_name = "acc_g{group}"}} : memref<{Q_ELEMENTS_PER_GROUP}xf32>
    %out_g{group} = aie.buffer(%worker_g{group}) {{sym_name = "out_g{group}"}} : memref<{Q_ELEMENTS_PER_GROUP}xbf16>"""
        )
    return "\n".join(blocks)


def _attention_core_stubs() -> str:
    blocks = []
    for stream in WORKER_STREAMS:
        group = stream.group
        blocks.append(
            f"""    // attention worker ABI contract for worker_g{group}
    %core_g{group} = aie.core(%worker_g{group}) {{
      %q_heads_g{group} = arith.constant {Q_HEADS_PER_GROUP} : i32
      %head_dim_g{group} = arith.constant {HEAD_DIM} : i32
      %tile_size_g{group} = arith.constant {KV_TILE_SIZE} : i32
      %current_row_g{group} = arith.constant -1 : i32
      %valid_rows_g{group} = arith.constant {KV_TILE_SIZE} : i32
      func.call @llama_chunked_attention_init_f32(%state_g{group}, %acc_g{group}, %q_heads_g{group}, %head_dim_g{group}) : (memref<{STATE_ELEMENTS}xf32>, memref<{Q_ELEMENTS_PER_GROUP}xf32>, i32, i32) -> ()
      func.call @qwen_plane_group_attention_update_bf16(%q_current_g{group}, %plane_pair_g{group}, %state_g{group}, %acc_g{group}, %current_row_g{group}, %valid_rows_g{group}, %q_heads_g{group}, %tile_size_g{group}, %head_dim_g{group}) : (memref<{Q_CURRENT_ELEMENTS_PER_GROUP}xbf16>, memref<{PLANE_GROUP_PAIR_CHUNK_ELEMENTS}xbf16>, memref<{STATE_ELEMENTS}xf32>, memref<{Q_ELEMENTS_PER_GROUP}xf32>, i32, i32, i32, i32, i32) -> ()
      func.call @llama_chunked_attention_finalize_bf16(%state_g{group}, %acc_g{group}, %out_g{group}, %q_heads_g{group}, %head_dim_g{group}) : (memref<{STATE_ELEMENTS}xf32>, memref<{Q_ELEMENTS_PER_GROUP}xf32>, memref<{Q_ELEMENTS_PER_GROUP}xbf16>, i32, i32) -> ()
      aie.end
    }} {{stack_size = {ATTENTION_STACK_SIZE} : i32}}"""
        )
    return "\n".join(blocks)


def build_static_kv_reader_contract_mlir(attend_seq_len: int, tile_size: int = 16) -> str:
    length_dwords = history_length_dwords(attend_seq_len, tile_size)
    ring_locks = "\n".join(_ring_lock_decls(ring) for ring in RINGS)
    ring_buffers = "\n".join(_ring_buffers(ring) for ring in RINGS)
    ring_dma = "\n".join(
        "\n".join(
            [
                _load_ring(ring),
                _half_ring(ring, "half0", ring.half0_channel, ring.half0_bds, "loaded", "split"),
                _half_ring(ring, "half1", ring.half1_channel, ring.half1_bds, "split", "empty"),
            ]
        )
        for ring in RINGS
    )
    patches = "\n".join(_writebd(plane, length_dwords) for plane in PLANES)
    worker_contract = _worker_contract_comments()
    worker_tiles = _worker_tile_decls()
    attention_kernel_decls = _attention_kernel_decls()
    attention_fifos = _attention_fifo_decls()
    attention_buffers = _attention_worker_buffers()
    attention_cores = _attention_core_stubs()
    return f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
{worker_tiles}

{worker_contract}

{attention_kernel_decls}

{attention_fifos}

{ring_locks}

{ring_buffers}

{attention_buffers}

    aie.flow(%shim, DMA : 0, %mem, DMA : 0)
    aie.flow(%shim, DMA : 1, %mem, DMA : 1)
    aie.flow(%mem, DMA : 0, %worker_g0, DMA : 0)
    aie.flow(%mem, DMA : 1, %worker_g1, DMA : 0)

{attention_cores}

    %memdma = aie.memtile_dma(%mem) {{
{ring_dma}
      aie.end
    }}

    aie.runtime_sequence(%kv_cache: memref<4194304xbf16>) {{
{patches}
    }}
  }}
}}
"""


def _core_dma_worker_decls() -> str:
    return """    %worker_g0 = aie.tile(0, 2)
    %worker_g1 = aie.tile(1, 2)
    %worker_g0_kv_ping = aie.buffer(%worker_g0) {sym_name = "worker_g0_kv_ping"} : memref<2048xbf16>
    %worker_g0_kv_pong = aie.buffer(%worker_g0) {sym_name = "worker_g0_kv_pong"} : memref<2048xbf16>
    %worker_g1_kv_ping = aie.buffer(%worker_g1) {sym_name = "worker_g1_kv_ping"} : memref<2048xbf16>
    %worker_g1_kv_pong = aie.buffer(%worker_g1) {sym_name = "worker_g1_kv_pong"} : memref<2048xbf16>
    %worker_g0_empty = aie.lock(%worker_g0, 0) {init = 2 : i32, sym_name = "worker_g0_empty"}
    %worker_g0_full = aie.lock(%worker_g0, 1) {init = 0 : i32, sym_name = "worker_g0_full"}
    %worker_g1_empty = aie.lock(%worker_g1, 0) {init = 2 : i32, sym_name = "worker_g1_empty"}
    %worker_g1_full = aie.lock(%worker_g1, 1) {init = 0 : i32, sym_name = "worker_g1_full"}"""


def _core_dma_worker_core(group: int) -> str:
    return f"""    %core_g{group} = aie.core(%worker_g{group}) {{
      %c0_g{group} = arith.constant 0 : index
      %c9223372036854775807_g{group} = arith.constant 9223372036854775807 : index
      %c1_g{group} = arith.constant 1 : index
      scf.for %iv_g{group} = %c0_g{group} to %c9223372036854775807_g{group} step %c1_g{group} {{
        aie.use_lock(%worker_g{group}_full, AcquireGreaterEqual, 1)
        aie.use_lock(%worker_g{group}_empty, Release, 1)
      }}
      aie.end
    }}"""


def _core_dma_worker_mem(group: int) -> str:
    return f"""    %worker_g{group}_mem = aie.mem(%worker_g{group}) {{
      %worker_g{group}_rx = aie.dma_start(S2MM, 0, ^worker_g{group}_rx_ping, ^worker_g{group}_after_rx)
    ^worker_g{group}_rx_ping:
      aie.use_lock(%worker_g{group}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%worker_g{group}_kv_ping : memref<2048xbf16>, 0, 2048) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%worker_g{group}_full, Release, 1)
      aie.next_bd ^worker_g{group}_rx_pong
    ^worker_g{group}_rx_pong:
      aie.use_lock(%worker_g{group}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%worker_g{group}_kv_pong : memref<2048xbf16>, 0, 2048) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%worker_g{group}_full, Release, 1)
      aie.next_bd ^worker_g{group}_rx_ping
    ^worker_g{group}_after_rx:
      aie.end
    }}"""


def _kv_pair_core_dma_worker_decls() -> str:
    return """    %worker_g0 = aie.tile(0, 2)
    %worker_g1 = aie.tile(1, 2)
    %worker_g0_k_ping = aie.buffer(%worker_g0) {sym_name = "worker_g0_k_ping"} : memref<2048xbf16>
    %worker_g0_k_pong = aie.buffer(%worker_g0) {sym_name = "worker_g0_k_pong"} : memref<2048xbf16>
    %worker_g0_v_ping = aie.buffer(%worker_g0) {sym_name = "worker_g0_v_ping"} : memref<2048xbf16>
    %worker_g0_v_pong = aie.buffer(%worker_g0) {sym_name = "worker_g0_v_pong"} : memref<2048xbf16>
    %worker_g1_k_ping = aie.buffer(%worker_g1) {sym_name = "worker_g1_k_ping"} : memref<2048xbf16>
    %worker_g1_k_pong = aie.buffer(%worker_g1) {sym_name = "worker_g1_k_pong"} : memref<2048xbf16>
    %worker_g1_v_ping = aie.buffer(%worker_g1) {sym_name = "worker_g1_v_ping"} : memref<2048xbf16>
    %worker_g1_v_pong = aie.buffer(%worker_g1) {sym_name = "worker_g1_v_pong"} : memref<2048xbf16>
    %worker_g0_k_empty = aie.lock(%worker_g0, 0) {init = 2 : i32, sym_name = "worker_g0_k_empty"}
    %worker_g0_k_full = aie.lock(%worker_g0, 1) {init = 0 : i32, sym_name = "worker_g0_k_full"}
    %worker_g0_v_empty = aie.lock(%worker_g0, 2) {init = 2 : i32, sym_name = "worker_g0_v_empty"}
    %worker_g0_v_full = aie.lock(%worker_g0, 3) {init = 0 : i32, sym_name = "worker_g0_v_full"}
    %worker_g1_k_empty = aie.lock(%worker_g1, 0) {init = 2 : i32, sym_name = "worker_g1_k_empty"}
    %worker_g1_k_full = aie.lock(%worker_g1, 1) {init = 0 : i32, sym_name = "worker_g1_k_full"}
    %worker_g1_v_empty = aie.lock(%worker_g1, 2) {init = 2 : i32, sym_name = "worker_g1_v_empty"}
    %worker_g1_v_full = aie.lock(%worker_g1, 3) {init = 0 : i32, sym_name = "worker_g1_v_full"}"""


def _kv_pair_attention_buffers() -> str:
    return "\n".join(
        [
            _kv_pair_core_dma_worker_decls(),
            f'    %q_current_g0 = aie.buffer(%worker_g0) {{sym_name = "q_current_g0"}} : memref<{Q_CURRENT_ELEMENTS_PER_GROUP}xbf16>',
            f'    %state_g0 = aie.buffer(%worker_g0) {{sym_name = "state_g0"}} : memref<{STATE_ELEMENTS}xf32>',
            f'    %acc_g0 = aie.buffer(%worker_g0) {{sym_name = "acc_g0"}} : memref<{Q_ELEMENTS_PER_GROUP}xf32>',
            f'    %q_current_g1 = aie.buffer(%worker_g1) {{sym_name = "q_current_g1"}} : memref<{Q_CURRENT_ELEMENTS_PER_GROUP}xbf16>',
            f'    %state_g1 = aie.buffer(%worker_g1) {{sym_name = "state_g1"}} : memref<{STATE_ELEMENTS}xf32>',
            f'    %acc_g1 = aie.buffer(%worker_g1) {{sym_name = "acc_g1"}} : memref<{Q_ELEMENTS_PER_GROUP}xf32>',
        ]
    )


def _dual_pair_attention_buffers(groups: tuple[int, ...]) -> str:
    blocks = []
    for group in groups:
        blocks.append(
            f"""    %worker_g{group} = aie.tile({group}, 2)
    %worker_g{group}_k_ping = aie.buffer(%worker_g{group}) {{sym_name = "worker_g{group}_k_ping"}} : memref<2048xbf16>
    %worker_g{group}_k_pong = aie.buffer(%worker_g{group}) {{sym_name = "worker_g{group}_k_pong"}} : memref<2048xbf16>
    %worker_g{group}_v_ping = aie.buffer(%worker_g{group}) {{sym_name = "worker_g{group}_v_ping"}} : memref<2048xbf16>
    %worker_g{group}_v_pong = aie.buffer(%worker_g{group}) {{sym_name = "worker_g{group}_v_pong"}} : memref<2048xbf16>
    %worker_g{group}_k_empty = aie.lock(%worker_g{group}, 0) {{init = 2 : i32, sym_name = "worker_g{group}_k_empty"}}
    %worker_g{group}_k_full = aie.lock(%worker_g{group}, 1) {{init = 0 : i32, sym_name = "worker_g{group}_k_full"}}
    %worker_g{group}_v_empty = aie.lock(%worker_g{group}, 2) {{init = 2 : i32, sym_name = "worker_g{group}_v_empty"}}
    %worker_g{group}_v_full = aie.lock(%worker_g{group}, 3) {{init = 0 : i32, sym_name = "worker_g{group}_v_full"}}
    %q_current_g{group} = aie.buffer(%worker_g{group}) {{sym_name = "q_current_g{group}"}} : memref<{Q_CURRENT_ELEMENTS_PER_GROUP}xbf16>
    %state_g{group} = aie.buffer(%worker_g{group}) {{sym_name = "state_g{group}"}} : memref<{STATE_ELEMENTS}xf32>
    %acc_g{group} = aie.buffer(%worker_g{group}) {{sym_name = "acc_g{group}"}} : memref<{Q_ELEMENTS_PER_GROUP}xf32>"""
        )
    return "\n".join(blocks)


def _bounded_attention_buffers(groups: tuple[int, ...]) -> str:
    blocks = []
    for group in groups:
        blocks.append(
            f"""    %worker_g{group} = aie.tile({group}, 2)
    %worker_g{group}_k_ping = aie.buffer(%worker_g{group}) {{sym_name = "worker_g{group}_k_ping"}} : memref<2048xbf16>
    %worker_g{group}_k_pong = aie.buffer(%worker_g{group}) {{sym_name = "worker_g{group}_k_pong"}} : memref<2048xbf16>
    %worker_g{group}_v_ping = aie.buffer(%worker_g{group}) {{sym_name = "worker_g{group}_v_ping"}} : memref<2048xbf16>
    %worker_g{group}_v_pong = aie.buffer(%worker_g{group}) {{sym_name = "worker_g{group}_v_pong"}} : memref<2048xbf16>
    %worker_g{group}_k_empty = aie.lock(%worker_g{group}, 0) {{init = 2 : i32, sym_name = "worker_g{group}_k_empty"}}
    %worker_g{group}_k_full = aie.lock(%worker_g{group}, 1) {{init = 0 : i32, sym_name = "worker_g{group}_k_full"}}
    %worker_g{group}_v_empty = aie.lock(%worker_g{group}, 2) {{init = 2 : i32, sym_name = "worker_g{group}_v_empty"}}
    %worker_g{group}_v_full = aie.lock(%worker_g{group}, 3) {{init = 0 : i32, sym_name = "worker_g{group}_v_full"}}
    %worker_g{group}_out_empty = aie.lock(%worker_g{group}, 4) {{init = 1 : i32, sym_name = "worker_g{group}_out_empty"}}
    %worker_g{group}_out_full = aie.lock(%worker_g{group}, 5) {{init = 0 : i32, sym_name = "worker_g{group}_out_full"}}
    %q_current_g{group} = aie.buffer(%worker_g{group}) {{sym_name = "q_current_g{group}"}} : memref<{Q_CURRENT_ELEMENTS_PER_GROUP}xbf16>
    %state_g{group} = aie.buffer(%worker_g{group}) {{sym_name = "state_g{group}"}} : memref<{STATE_ELEMENTS}xf32>
    %acc_g{group} = aie.buffer(%worker_g{group}) {{sym_name = "acc_g{group}"}} : memref<{Q_ELEMENTS_PER_GROUP}xf32>
    %out_g{group} = aie.buffer(%worker_g{group}) {{sym_name = "out_g{group}"}} : memref<{Q_ELEMENTS_PER_GROUP}xbf16>"""
        )
    return "\n".join(blocks)


def _renamed_ring(name: str, ring: RingDescriptor) -> RingDescriptor:
    return RingDescriptor(
        name=name,
        mlir_locks=ring.mlir_locks,
        fastflow_locks=ring.fastflow_locks,
        load_channel=ring.load_channel,
        load_bds=ring.load_bds,
        half0_channel=ring.half0_channel,
        half0_bds=ring.half0_bds,
        half1_channel=ring.half1_channel,
        half1_bds=ring.half1_bds,
    )


def _kv_pair_ring_dma(key_ring: RingDescriptor, value_ring: RingDescriptor) -> str:
    return "\n".join(
        [
            _load_ring(key_ring),
            _half_ring(
                key_ring,
                "half0",
                key_ring.half0_channel,
                key_ring.half0_bds,
                "loaded",
                "split",
            ),
            _half_ring(
                key_ring,
                "half1",
                key_ring.half1_channel,
                key_ring.half1_bds,
                "split",
                "empty",
            ),
            _load_ring(value_ring),
            _half_ring(
                value_ring,
                "half0",
                value_ring.half0_channel,
                value_ring.half0_bds,
                "loaded",
                "split",
            ),
            _half_ring(
                value_ring,
                "half1",
                value_ring.half1_channel,
                value_ring.half1_bds,
                "split",
                "empty",
            ),
        ]
    )


def _kv_pair_source_fanout_ring_dma(
    key_ring: RingDescriptor,
    value_ring: RingDescriptor,
) -> str:
    return "\n".join(
        [
            _load_ring(key_ring),
            _full_forward_ring(
                key_ring,
                "to_left",
                key_ring.half0_channel,
                key_ring.half0_bds,
                "loaded",
                "split",
            ),
            _full_forward_ring(
                key_ring,
                "to_right",
                key_ring.half1_channel,
                key_ring.half1_bds,
                "split",
                "empty",
            ),
            _load_ring(value_ring),
            _full_forward_ring(
                value_ring,
                "to_left",
                value_ring.half0_channel,
                value_ring.half0_bds,
                "loaded",
                "split",
            ),
            _full_forward_ring(
                value_ring,
                "to_right",
                value_ring.half1_channel,
                value_ring.half1_bds,
                "split",
                "empty",
            ),
        ]
    )


def _kv_pair_core_dma_worker_core(group: int) -> str:
    return f"""    %core_g{group} = aie.core(%worker_g{group}) {{
      %c0_g{group} = arith.constant 0 : index
      %c9223372036854775807_g{group} = arith.constant 9223372036854775807 : index
      %c1_g{group} = arith.constant 1 : index
      scf.for %iv_g{group} = %c0_g{group} to %c9223372036854775807_g{group} step %c1_g{group} {{
        aie.use_lock(%worker_g{group}_k_full, AcquireGreaterEqual, 1)
        aie.use_lock(%worker_g{group}_v_full, AcquireGreaterEqual, 1)
        aie.use_lock(%worker_g{group}_k_empty, Release, 1)
        aie.use_lock(%worker_g{group}_v_empty, Release, 1)
      }}
      aie.end
    }}"""


def _kv_pair_attention_worker_core(group: int) -> str:
    return f"""    %core_g{group} = aie.core(%worker_g{group}) {{
      %q_heads_g{group} = arith.constant {Q_HEADS_PER_GROUP} : i32
      %head_dim_g{group} = arith.constant {HEAD_DIM} : i32
      %tile_size_g{group} = arith.constant {KV_TILE_SIZE} : i32
      %current_row_g{group} = arith.constant -1 : i32
      %valid_rows_g{group} = arith.constant {KV_TILE_SIZE} : i32
      %c0_g{group} = arith.constant 0 : index
      %c9223372036854775807_g{group} = arith.constant 9223372036854775807 : index
      %c1_g{group} = arith.constant 1 : index
      func.call @llama_chunked_attention_init_f32(%state_g{group}, %acc_g{group}, %q_heads_g{group}, %head_dim_g{group}) : (memref<{STATE_ELEMENTS}xf32>, memref<{Q_ELEMENTS_PER_GROUP}xf32>, i32, i32) -> ()
      scf.for %iv_g{group} = %c0_g{group} to %c9223372036854775807_g{group} step %c1_g{group} {{
        aie.use_lock(%worker_g{group}_k_full, AcquireGreaterEqual, 1)
        aie.use_lock(%worker_g{group}_v_full, AcquireGreaterEqual, 1)
        func.call @qwen_plane_group_attention_update_split_bf16(%q_current_g{group}, %worker_g{group}_k_ping, %worker_g{group}_v_ping, %state_g{group}, %acc_g{group}, %current_row_g{group}, %valid_rows_g{group}, %q_heads_g{group}, %tile_size_g{group}, %head_dim_g{group}) : (memref<{Q_CURRENT_ELEMENTS_PER_GROUP}xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<{STATE_ELEMENTS}xf32>, memref<{Q_ELEMENTS_PER_GROUP}xf32>, i32, i32, i32, i32, i32) -> ()
        aie.use_lock(%worker_g{group}_k_empty, Release, 1)
        aie.use_lock(%worker_g{group}_v_empty, Release, 1)
        aie.use_lock(%worker_g{group}_k_full, AcquireGreaterEqual, 1)
        aie.use_lock(%worker_g{group}_v_full, AcquireGreaterEqual, 1)
        func.call @qwen_plane_group_attention_update_split_bf16(%q_current_g{group}, %worker_g{group}_k_pong, %worker_g{group}_v_pong, %state_g{group}, %acc_g{group}, %current_row_g{group}, %valid_rows_g{group}, %q_heads_g{group}, %tile_size_g{group}, %head_dim_g{group}) : (memref<{Q_CURRENT_ELEMENTS_PER_GROUP}xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<{STATE_ELEMENTS}xf32>, memref<{Q_ELEMENTS_PER_GROUP}xf32>, i32, i32, i32, i32, i32) -> ()
        aie.use_lock(%worker_g{group}_k_empty, Release, 1)
        aie.use_lock(%worker_g{group}_v_empty, Release, 1)
      }}
      aie.end
    }} {{stack_size = {ATTENTION_STACK_SIZE} : i32}}"""


def _bounded_attention_update_blocks(group: int, attend_seq_len: int, tile_size: int) -> str:
    blocks = []
    for tile_idx in range(history_tiles(attend_seq_len, tile_size)):
        buffer_name = "ping" if tile_idx % 2 == 0 else "pong"
        valid_rows = min(tile_size, attend_seq_len - tile_idx * tile_size)
        blocks.append(
            f"""      %valid_rows_g{group}_{tile_idx} = arith.constant {valid_rows} : i32
      aie.use_lock(%worker_g{group}_k_full, AcquireGreaterEqual, 1)
      aie.use_lock(%worker_g{group}_v_full, AcquireGreaterEqual, 1)
      func.call @qwen_plane_group_attention_update_split_bf16(%q_current_g{group}, %worker_g{group}_k_{buffer_name}, %worker_g{group}_v_{buffer_name}, %state_g{group}, %acc_g{group}, %current_row_g{group}, %valid_rows_g{group}_{tile_idx}, %q_heads_g{group}, %tile_size_g{group}, %head_dim_g{group}) : (memref<{Q_CURRENT_ELEMENTS_PER_GROUP}xbf16>, memref<2048xbf16>, memref<2048xbf16>, memref<{STATE_ELEMENTS}xf32>, memref<{Q_ELEMENTS_PER_GROUP}xf32>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%worker_g{group}_k_empty, Release, 1)
      aie.use_lock(%worker_g{group}_v_empty, Release, 1)"""
        )
    return "\n".join(blocks)


def _bounded_attention_worker_core(group: int, attend_seq_len: int, tile_size: int) -> str:
    updates = _bounded_attention_update_blocks(group, attend_seq_len, tile_size)
    return f"""    %core_g{group} = aie.core(%worker_g{group}) {{
      %q_heads_g{group} = arith.constant {Q_HEADS_PER_GROUP} : i32
      %head_dim_g{group} = arith.constant {HEAD_DIM} : i32
      %tile_size_g{group} = arith.constant {KV_TILE_SIZE} : i32
      %current_row_g{group} = arith.constant -1 : i32
      %q_current_elements_g{group} = arith.constant {Q_CURRENT_ELEMENTS_PER_GROUP} : i32
      func.call @qwen_zero_q_current_bf16(%q_current_g{group}, %q_current_elements_g{group}) : (memref<{Q_CURRENT_ELEMENTS_PER_GROUP}xbf16>, i32) -> ()
      func.call @llama_chunked_attention_init_f32(%state_g{group}, %acc_g{group}, %q_heads_g{group}, %head_dim_g{group}) : (memref<{STATE_ELEMENTS}xf32>, memref<{Q_ELEMENTS_PER_GROUP}xf32>, i32, i32) -> ()
{updates}
      aie.use_lock(%worker_g{group}_out_empty, AcquireGreaterEqual, 1)
      func.call @llama_chunked_attention_finalize_bf16(%state_g{group}, %acc_g{group}, %out_g{group}, %q_heads_g{group}, %head_dim_g{group}) : (memref<{STATE_ELEMENTS}xf32>, memref<{Q_ELEMENTS_PER_GROUP}xf32>, memref<{Q_ELEMENTS_PER_GROUP}xbf16>, i32, i32) -> ()
      aie.use_lock(%worker_g{group}_out_full, Release, 1)
      aie.end
    }} {{stack_size = {ATTENTION_STACK_SIZE} : i32}}"""


def _kv_pair_worker_rx_ring(group: int, role: str, channel: int) -> str:
    return f"""      %worker_g{group}_{role}_rx = aie.dma_start(S2MM, {channel}, ^worker_g{group}_{role}_rx_ping, ^worker_g{group}_{role}_after_rx)
    ^worker_g{group}_{role}_rx_ping:
      aie.use_lock(%worker_g{group}_{role}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%worker_g{group}_{role}_ping : memref<2048xbf16>, 0, 2048) {{bd_id = {channel * 2} : i32, next_bd_id = {channel * 2 + 1} : i32}}
      aie.use_lock(%worker_g{group}_{role}_full, Release, 1)
      aie.next_bd ^worker_g{group}_{role}_rx_pong
    ^worker_g{group}_{role}_rx_pong:
      aie.use_lock(%worker_g{group}_{role}_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%worker_g{group}_{role}_pong : memref<2048xbf16>, 0, 2048) {{bd_id = {channel * 2 + 1} : i32, next_bd_id = {channel * 2} : i32}}
      aie.use_lock(%worker_g{group}_{role}_full, Release, 1)
      aie.next_bd ^worker_g{group}_{role}_rx_ping
    ^worker_g{group}_{role}_after_rx:
"""


def _bounded_attention_context_tx_ring(group: int) -> str:
    return f"""      %worker_g{group}_out_tx = aie.dma_start(MM2S, 0, ^worker_g{group}_out_tx_bd, ^worker_g{group}_after_out_tx)
    ^worker_g{group}_out_tx_bd:
      aie.use_lock(%worker_g{group}_out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_g{group} : memref<{Q_ELEMENTS_PER_GROUP}xbf16>, 0, {Q_ELEMENTS_PER_GROUP}) {{bd_id = 4 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%worker_g{group}_out_empty, Release, 1)
      aie.next_bd ^worker_g{group}_out_tx_bd
    ^worker_g{group}_after_out_tx:
"""


def _kv_pair_core_dma_worker_mem(group: int) -> str:
    return f"""    %worker_g{group}_mem = aie.mem(%worker_g{group}) {{
{_kv_pair_worker_rx_ring(group, "k", 0)}
{_kv_pair_worker_rx_ring(group, "v", 1)}
      aie.end
    }}"""


def _bounded_attention_worker_mem(group: int) -> str:
    return f"""    %worker_g{group}_mem = aie.mem(%worker_g{group}) {{
{_kv_pair_worker_rx_ring(group, "k", 0)}
{_kv_pair_worker_rx_ring(group, "v", 1)}
{_bounded_attention_context_tx_ring(group)}
      aie.end
    }}"""


def build_static_kv_core_dma_contract_mlir(
    attend_seq_len: int,
    tile_size: int = 16,
) -> str:
    length_dwords = history_length_dwords(attend_seq_len, tile_size)
    ring = RINGS[0]
    plane = PLANES[0]
    return f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
{_core_dma_worker_decls()}

    // core-DMA contract: k03 plane -> row1 ring_a -> worker_g0/g1 half buffers
{_ring_lock_decls(ring)}

{_ring_buffers(ring)}

    aie.flow(%shim, DMA : 0, %mem, DMA : 0)
    aie.flow(%mem, DMA : 0, %worker_g0, DMA : 0)
    aie.flow(%mem, DMA : 1, %worker_g1, DMA : 0)

{_core_dma_worker_core(0)}
{_core_dma_worker_core(1)}

{_core_dma_worker_mem(0)}
{_core_dma_worker_mem(1)}

    %memdma = aie.memtile_dma(%mem) {{
{_load_ring(ring)}
{_half_ring(ring, "half0", ring.half0_channel, ring.half0_bds, "loaded", "split")}
{_half_ring(ring, "half1", ring.half1_channel, ring.half1_bds, "split", "empty")}
      aie.end
    }}

    aie.runtime_sequence(%kv_cache: memref<4194304xbf16>) {{
{_writebd(plane, length_dwords)}
    }}
  }}
}}
"""


def build_static_kv_pair_core_dma_contract_mlir(
    attend_seq_len: int,
    tile_size: int = 16,
) -> str:
    length_dwords = history_length_dwords(attend_seq_len, tile_size)
    key_ring = RINGS[0]
    value_ring = RINGS[1]
    key_plane = PLANES[0]
    value_plane = PLANES[1]
    ring_dma = "\n".join(
        [
            _load_ring(key_ring),
            _half_ring(
                key_ring,
                "half0",
                key_ring.half0_channel,
                key_ring.half0_bds,
                "loaded",
                "split",
            ),
            _half_ring(
                key_ring,
                "half1",
                key_ring.half1_channel,
                key_ring.half1_bds,
                "split",
                "empty",
            ),
            _load_ring(value_ring),
            _half_ring(
                value_ring,
                "half0",
                value_ring.half0_channel,
                value_ring.half0_bds,
                "loaded",
                "split",
            ),
            _half_ring(
                value_ring,
                "half1",
                value_ring.half1_channel,
                value_ring.half1_bds,
                "split",
                "empty",
            ),
        ]
    )
    patches = "\n".join(
        [
            _writebd(key_plane, length_dwords),
            _writebd(value_plane, length_dwords),
        ]
    )
    return f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
{_kv_pair_core_dma_worker_decls()}

    // K/V-pair core-DMA contract: k03 ring_a and v03 ring_b -> worker_g0/g1 K/V buffers
{_ring_lock_decls(key_ring)}
{_ring_lock_decls(value_ring)}

{_ring_buffers(key_ring)}
{_ring_buffers(value_ring)}

    aie.flow(%shim, DMA : 0, %mem, DMA : 0)
    aie.flow(%shim, DMA : 1, %mem, DMA : 1)
    aie.flow(%mem, DMA : 0, %worker_g0, DMA : 0)
    aie.flow(%mem, DMA : 1, %worker_g1, DMA : 0)
    aie.flow(%mem, DMA : 2, %worker_g0, DMA : 1)
    aie.flow(%mem, DMA : 3, %worker_g1, DMA : 1)

{_kv_pair_core_dma_worker_core(0)}
{_kv_pair_core_dma_worker_core(1)}

{_kv_pair_core_dma_worker_mem(0)}
{_kv_pair_core_dma_worker_mem(1)}

    %memdma = aie.memtile_dma(%mem) {{
{ring_dma}
      aie.end
    }}

    aie.runtime_sequence(%kv_cache: memref<4194304xbf16>) {{
{patches}
    }}
  }}
}}
"""


def build_static_kv_pair_attention_ingress_contract_mlir(
    attend_seq_len: int,
    tile_size: int = 16,
) -> str:
    length_dwords = history_length_dwords(attend_seq_len, tile_size)
    key_ring = RINGS[0]
    value_ring = RINGS[1]
    key_plane = PLANES[0]
    value_plane = PLANES[1]
    ring_dma = "\n".join(
        [
            _load_ring(key_ring),
            _half_ring(
                key_ring,
                "half0",
                key_ring.half0_channel,
                key_ring.half0_bds,
                "loaded",
                "split",
            ),
            _half_ring(
                key_ring,
                "half1",
                key_ring.half1_channel,
                key_ring.half1_bds,
                "split",
                "empty",
            ),
            _load_ring(value_ring),
            _half_ring(
                value_ring,
                "half0",
                value_ring.half0_channel,
                value_ring.half0_bds,
                "loaded",
                "split",
            ),
            _half_ring(
                value_ring,
                "half1",
                value_ring.half1_channel,
                value_ring.half1_bds,
                "split",
                "empty",
            ),
        ]
    )
    patches = "\n".join(
        [
            _writebd(key_plane, length_dwords),
            _writebd(value_plane, length_dwords),
        ]
    )
    return f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
{_kv_pair_attention_buffers()}

    // K/V-pair core-DMA contract: k03 ring_a and v03 ring_b -> worker_g0/g1 K/V buffers
    // K/V-pair attention ingress contract: static ring streams call split attention update
{_split_attention_kernel_decls()}

{_ring_lock_decls(key_ring)}
{_ring_lock_decls(value_ring)}

{_ring_buffers(key_ring)}
{_ring_buffers(value_ring)}

    aie.flow(%shim, DMA : 0, %mem, DMA : 0)
    aie.flow(%shim, DMA : 1, %mem, DMA : 1)
    aie.flow(%mem, DMA : 0, %worker_g0, DMA : 0)
    aie.flow(%mem, DMA : 1, %worker_g1, DMA : 0)
    aie.flow(%mem, DMA : 2, %worker_g0, DMA : 1)
    aie.flow(%mem, DMA : 3, %worker_g1, DMA : 1)

{_kv_pair_attention_worker_core(0)}
{_kv_pair_attention_worker_core(1)}

{_kv_pair_core_dma_worker_mem(0)}
{_kv_pair_core_dma_worker_mem(1)}

    %memdma = aie.memtile_dma(%mem) {{
{ring_dma}
      aie.end
    }}

    aie.runtime_sequence(%kv_cache: memref<4194304xbf16>) {{
{patches}
    }}
  }}
}}
"""


def build_static_kv_dual_pair_attention_ingress_contract_mlir(
    attend_seq_len: int,
    tile_size: int = 16,
) -> str:
    length_dwords = history_length_dwords(attend_seq_len, tile_size)
    left_key_ring = _renamed_ring("left_key_ring", RINGS[0])
    left_value_ring = _renamed_ring("left_value_ring", RINGS[1])
    right_key_ring = _renamed_ring("right_key_ring", RINGS[0])
    right_value_ring = _renamed_ring("right_value_ring", RINGS[1])
    left_ring_dma = "\n".join(
        [
            _load_ring(left_key_ring),
            _half_ring(
                left_key_ring,
                "half0",
                left_key_ring.half0_channel,
                left_key_ring.half0_bds,
                "loaded",
                "split",
            ),
            _half_ring(
                left_key_ring,
                "half1",
                left_key_ring.half1_channel,
                left_key_ring.half1_bds,
                "split",
                "empty",
            ),
            _load_ring(left_value_ring),
            _half_ring(
                left_value_ring,
                "half0",
                left_value_ring.half0_channel,
                left_value_ring.half0_bds,
                "loaded",
                "split",
            ),
            _half_ring(
                left_value_ring,
                "half1",
                left_value_ring.half1_channel,
                left_value_ring.half1_bds,
                "split",
                "empty",
            ),
        ]
    )
    right_ring_dma = "\n".join(
        [
            _load_ring(right_key_ring),
            _half_ring(
                right_key_ring,
                "half0",
                right_key_ring.half0_channel,
                right_key_ring.half0_bds,
                "loaded",
                "split",
            ),
            _half_ring(
                right_key_ring,
                "half1",
                right_key_ring.half1_channel,
                right_key_ring.half1_bds,
                "split",
                "empty",
            ),
            _load_ring(right_value_ring),
            _half_ring(
                right_value_ring,
                "half0",
                right_value_ring.half0_channel,
                right_value_ring.half0_bds,
                "loaded",
                "split",
            ),
            _half_ring(
                right_value_ring,
                "half1",
                right_value_ring.half1_channel,
                right_value_ring.half1_bds,
                "split",
                "empty",
            ),
        ]
    )
    patches = "\n".join(
        [
            _writebd(PLANES[0], length_dwords, column=0),
            _writebd(PLANES[1], length_dwords, column=0),
            _writebd(PLANES[2], length_dwords, column=4),
            _writebd(PLANES[3], length_dwords, column=4),
        ]
    )
    return f"""module {{
  aie.device(npu2) {{
    %shim0 = aie.tile(0, 0)
    %shim4 = aie.tile(4, 0)
    %mem0 = aie.tile(0, 1)
    %mem4 = aie.tile(4, 1)
{_dual_pair_attention_buffers((0, 1, 4, 5))}

    // Dual K/V-pair attention ingress contract:
    //   mem0 reads k03/v03 from shim column 0 and feeds worker_g0/g1.
    //   mem4 reads k47/v47 from shim column 4 and feeds worker_g4/g5.
{_split_attention_kernel_decls()}

{_ring_lock_decls(left_key_ring, "%mem0")}
{_ring_lock_decls(left_value_ring, "%mem0")}
{_ring_lock_decls(right_key_ring, "%mem4")}
{_ring_lock_decls(right_value_ring, "%mem4")}

{_ring_buffers(left_key_ring, "%mem0")}
{_ring_buffers(left_value_ring, "%mem0")}
{_ring_buffers(right_key_ring, "%mem4")}
{_ring_buffers(right_value_ring, "%mem4")}

    aie.flow(%shim0, DMA : 0, %mem0, DMA : 0)
    aie.flow(%shim0, DMA : 1, %mem0, DMA : 1)
    aie.flow(%mem0, DMA : 0, %worker_g0, DMA : 0)
    aie.flow(%mem0, DMA : 1, %worker_g1, DMA : 0)
    aie.flow(%mem0, DMA : 2, %worker_g0, DMA : 1)
    aie.flow(%mem0, DMA : 3, %worker_g1, DMA : 1)
    aie.flow(%shim4, DMA : 0, %mem4, DMA : 0)
    aie.flow(%shim4, DMA : 1, %mem4, DMA : 1)
    aie.flow(%mem4, DMA : 0, %worker_g4, DMA : 0)
    aie.flow(%mem4, DMA : 1, %worker_g5, DMA : 0)
    aie.flow(%mem4, DMA : 2, %worker_g4, DMA : 1)
    aie.flow(%mem4, DMA : 3, %worker_g5, DMA : 1)

{_kv_pair_attention_worker_core(0)}
{_kv_pair_attention_worker_core(1)}
{_kv_pair_attention_worker_core(4)}
{_kv_pair_attention_worker_core(5)}

{_kv_pair_core_dma_worker_mem(0)}
{_kv_pair_core_dma_worker_mem(1)}
{_kv_pair_core_dma_worker_mem(4)}
{_kv_pair_core_dma_worker_mem(5)}

    %memdma0 = aie.memtile_dma(%mem0) {{
{left_ring_dma}
      aie.end
    }}

    %memdma4 = aie.memtile_dma(%mem4) {{
{right_ring_dma}
      aie.end
    }}

    aie.runtime_sequence(%kv_cache: memref<4194304xbf16>) {{
{patches}
    }}
  }}
}}
"""


def build_static_kv_quad_pair_attention_ingress_contract_mlir(
    attend_seq_len: int,
    tile_size: int = 16,
) -> str:
    length_dwords = history_length_dwords(attend_seq_len, tile_size)
    p0_key_ring = _renamed_ring("p0_key_ring", RINGS[0])
    p0_value_ring = _renamed_ring("p0_value_ring", RINGS[1])
    p1_key_ring = _renamed_ring("p1_key_ring", RINGS[0])
    p1_value_ring = _renamed_ring("p1_value_ring", RINGS[1])
    p2_key_ring = _renamed_ring("p2_key_ring", RINGS[0])
    p2_value_ring = _renamed_ring("p2_value_ring", RINGS[1])
    p3_key_ring = _renamed_ring("p3_key_ring", RINGS[0])
    p3_value_ring = _renamed_ring("p3_value_ring", RINGS[1])
    p0_ring_dma = _kv_pair_ring_dma(p0_key_ring, p0_value_ring)
    p1_ring_dma = _kv_pair_ring_dma(p1_key_ring, p1_value_ring)
    p2_ring_dma = _kv_pair_ring_dma(p2_key_ring, p2_value_ring)
    p3_ring_dma = _kv_pair_ring_dma(p3_key_ring, p3_value_ring)
    patches = "\n".join(
        [
            _writebd(PLANES[0], length_dwords, column=0),
            _writebd(PLANES[1], length_dwords, column=0),
            _writebd(PLANES[0], length_dwords, column=2),
            _writebd(PLANES[1], length_dwords, column=2),
            _writebd(PLANES[2], length_dwords, column=4),
            _writebd(PLANES[3], length_dwords, column=4),
            _writebd(PLANES[2], length_dwords, column=6),
            _writebd(PLANES[3], length_dwords, column=6),
        ]
    )
    return f"""module {{
  aie.device(npu2) {{
    %shim0 = aie.tile(0, 0)
    %shim2 = aie.tile(2, 0)
    %shim4 = aie.tile(4, 0)
    %shim6 = aie.tile(6, 0)
    %mem0 = aie.tile(0, 1)
    %mem2 = aie.tile(2, 1)
    %mem4 = aie.tile(4, 1)
    %mem6 = aie.tile(6, 1)
{_dual_pair_attention_buffers((0, 1, 2, 3, 4, 5, 6, 7))}

    // Quad K/V-pair attention ingress contract:
    //   mem0 reads k03/v03 from shim column 0 and feeds worker_g0/g1.
    //   mem2 reads k03/v03 from shim column 2 and feeds worker_g2/g3.
    //   mem4 reads k47/v47 from shim column 4 and feeds worker_g4/g5.
    //   mem6 reads k47/v47 from shim column 6 and feeds worker_g6/g7.
    // This covers all 8 workers, but intentionally duplicates each plane-pair read once.
{_split_attention_kernel_decls()}

{_ring_lock_decls(p0_key_ring, "%mem0")}
{_ring_lock_decls(p0_value_ring, "%mem0")}
{_ring_lock_decls(p1_key_ring, "%mem2")}
{_ring_lock_decls(p1_value_ring, "%mem2")}
{_ring_lock_decls(p2_key_ring, "%mem4")}
{_ring_lock_decls(p2_value_ring, "%mem4")}
{_ring_lock_decls(p3_key_ring, "%mem6")}
{_ring_lock_decls(p3_value_ring, "%mem6")}

{_ring_buffers(p0_key_ring, "%mem0")}
{_ring_buffers(p0_value_ring, "%mem0")}
{_ring_buffers(p1_key_ring, "%mem2")}
{_ring_buffers(p1_value_ring, "%mem2")}
{_ring_buffers(p2_key_ring, "%mem4")}
{_ring_buffers(p2_value_ring, "%mem4")}
{_ring_buffers(p3_key_ring, "%mem6")}
{_ring_buffers(p3_value_ring, "%mem6")}

    aie.flow(%shim0, DMA : 0, %mem0, DMA : 0)
    aie.flow(%shim0, DMA : 1, %mem0, DMA : 1)
    aie.flow(%mem0, DMA : 0, %worker_g0, DMA : 0)
    aie.flow(%mem0, DMA : 1, %worker_g1, DMA : 0)
    aie.flow(%mem0, DMA : 2, %worker_g0, DMA : 1)
    aie.flow(%mem0, DMA : 3, %worker_g1, DMA : 1)
    aie.flow(%shim2, DMA : 0, %mem2, DMA : 0)
    aie.flow(%shim2, DMA : 1, %mem2, DMA : 1)
    aie.flow(%mem2, DMA : 0, %worker_g2, DMA : 0)
    aie.flow(%mem2, DMA : 1, %worker_g3, DMA : 0)
    aie.flow(%mem2, DMA : 2, %worker_g2, DMA : 1)
    aie.flow(%mem2, DMA : 3, %worker_g3, DMA : 1)
    aie.flow(%shim4, DMA : 0, %mem4, DMA : 0)
    aie.flow(%shim4, DMA : 1, %mem4, DMA : 1)
    aie.flow(%mem4, DMA : 0, %worker_g4, DMA : 0)
    aie.flow(%mem4, DMA : 1, %worker_g5, DMA : 0)
    aie.flow(%mem4, DMA : 2, %worker_g4, DMA : 1)
    aie.flow(%mem4, DMA : 3, %worker_g5, DMA : 1)
    aie.flow(%shim6, DMA : 0, %mem6, DMA : 0)
    aie.flow(%shim6, DMA : 1, %mem6, DMA : 1)
    aie.flow(%mem6, DMA : 0, %worker_g6, DMA : 0)
    aie.flow(%mem6, DMA : 1, %worker_g7, DMA : 0)
    aie.flow(%mem6, DMA : 2, %worker_g6, DMA : 1)
    aie.flow(%mem6, DMA : 3, %worker_g7, DMA : 1)

{_kv_pair_attention_worker_core(0)}
{_kv_pair_attention_worker_core(1)}
{_kv_pair_attention_worker_core(2)}
{_kv_pair_attention_worker_core(3)}
{_kv_pair_attention_worker_core(4)}
{_kv_pair_attention_worker_core(5)}
{_kv_pair_attention_worker_core(6)}
{_kv_pair_attention_worker_core(7)}

{_kv_pair_core_dma_worker_mem(0)}
{_kv_pair_core_dma_worker_mem(1)}
{_kv_pair_core_dma_worker_mem(2)}
{_kv_pair_core_dma_worker_mem(3)}
{_kv_pair_core_dma_worker_mem(4)}
{_kv_pair_core_dma_worker_mem(5)}
{_kv_pair_core_dma_worker_mem(6)}
{_kv_pair_core_dma_worker_mem(7)}

    %memdma0 = aie.memtile_dma(%mem0) {{
{p0_ring_dma}
      aie.end
    }}

    %memdma2 = aie.memtile_dma(%mem2) {{
{p1_ring_dma}
      aie.end
    }}

    %memdma4 = aie.memtile_dma(%mem4) {{
{p2_ring_dma}
      aie.end
    }}

    %memdma6 = aie.memtile_dma(%mem6) {{
{p3_ring_dma}
      aie.end
    }}

    aie.runtime_sequence(%kv_cache: memref<4194304xbf16>) {{
{patches}
    }}
  }}
}}
"""


def _context_output_allocation(group: int) -> str:
    return f"    aie.shim_dma_allocation @context_g{group}_alloc(%shim{group}, S2MM, 0)"


def _context_output_task(group: int) -> str:
    offset = group * Q_ELEMENTS_PER_GROUP
    return f"""      %context_task_g{group} = aiex.dma_configure_task_for @context_g{group}_alloc {{
        aie.dma_bd(%context : memref<{len(WORKER_STREAMS) * Q_ELEMENTS_PER_GROUP}xbf16>, {offset}, {Q_ELEMENTS_PER_GROUP}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {Q_ELEMENTS_PER_GROUP}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%context_task_g{group})"""


def _context_output_await(group: int) -> str:
    return f"""      aiex.dma_await_task(%context_task_g{group})
      aiex.dma_free_task(%context_task_g{group})"""


def _kv_runtime_input_allocation(name: str, column: int, channel: int) -> str:
    return f"    aie.shim_dma_allocation @{name}_alloc(%shim{column}, MM2S, {channel})"


def _kv_runtime_input_task(
    name: str,
    offset_elements: int,
    length_elements: int,
) -> str:
    return f"""      %{name}_task = aiex.dma_configure_task_for @{name}_alloc {{
        aie.dma_bd(%kv_cache : memref<{kv_plane_total_bf16_elements()}xbf16>, {offset_elements}, {length_elements}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {length_elements}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%{name}_task)"""


def _kv_runtime_input_free(name: str) -> str:
    return f"      aiex.dma_free_task(%{name}_task)"


def _shim_bd_address(column: int, bd_id: int) -> int:
    return column * 0x02000000 + 0x1D004 + bd_id * 0x20


def _npu_writebd(
    bd_id: int,
    column: int,
    buffer_length_dwords: int,
    buffer_offset_bytes: int,
) -> str:
    return f"      aiex.npu.writebd {{bd_id = {bd_id} : i32, buffer_length = {buffer_length_dwords} : i32, buffer_offset = {buffer_offset_bytes} : i32, column = {column} : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}"


def _npu_address_patch(column: int, bd_id: int, arg_idx: int, arg_plus_bytes: int) -> str:
    addr = _shim_bd_address(column, bd_id)
    return f"      aiex.npu.address_patch {{addr = {addr} : ui32, arg_idx = {arg_idx} : i32, arg_plus = {arg_plus_bytes} : i32}}"


def _npu_push_queue(
    column: int,
    direction: str,
    channel: int,
    bd_id: int,
    issue_token: bool,
) -> str:
    token = "true" if issue_token else "false"
    return f"      aiex.npu.push_queue({column}, 0, {direction} : {channel}) {{bd_id = {bd_id} : i32, issue_token = {token}, repeat_count = 0 : i32}}"


def _npu_sync_s2mm(column: int, channel: int = 0) -> str:
    return f"      aiex.npu.sync {{channel = {channel} : i32, column = {column} : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}}"


def _npu_input_sequence(
    name: str,
    column: int,
    channel: int,
    bd_id: int,
    offset_bytes: int,
    length_dwords: int,
) -> str:
    return "\n".join(
        [
            f"      // {name}: arg0+0x{offset_bytes:x} -> shim{column} MM2S{channel} bd{bd_id}",
            _npu_writebd(bd_id, column, length_dwords, offset_bytes),
            _npu_address_patch(column, bd_id, arg_idx=0, arg_plus_bytes=offset_bytes),
            _npu_push_queue(column, "MM2S", channel, bd_id, issue_token=False),
        ]
    )


def _npu_context_sequence(group: int, bd_id: int) -> str:
    offset_bytes = group * Q_ELEMENTS_PER_GROUP * 2
    length_dwords = Q_ELEMENTS_PER_GROUP // 2
    return "\n".join(
        [
            f"      // context_g{group}: shim{group} S2MM0 bd{bd_id} -> arg1+0x{offset_bytes:x}",
            _npu_writebd(bd_id, group, length_dwords, offset_bytes),
            _npu_address_patch(group, bd_id, arg_idx=1, arg_plus_bytes=offset_bytes),
            _npu_push_queue(group, "S2MM", 0, bd_id, issue_token=True),
        ]
    )


def kv_plane_total_bf16_elements(packet_seq_len: int = 4096) -> int:
    return len(PLANES) * packet_seq_len * 4 * HEAD_DIM


def build_static_kv_quad_pair_attention_bounded_contract_mlir(
    attend_seq_len: int,
    tile_size: int = 16,
) -> str:
    length_elements = history_tiles(attend_seq_len, tile_size) * 4096
    p0_key_ring = _renamed_ring("p0_key_ring", RINGS[0])
    p0_value_ring = _renamed_ring("p0_value_ring", RINGS[1])
    p1_key_ring = _renamed_ring("p1_key_ring", RINGS[0])
    p1_value_ring = _renamed_ring("p1_value_ring", RINGS[1])
    p2_key_ring = _renamed_ring("p2_key_ring", RINGS[0])
    p2_value_ring = _renamed_ring("p2_value_ring", RINGS[1])
    p3_key_ring = _renamed_ring("p3_key_ring", RINGS[0])
    p3_value_ring = _renamed_ring("p3_value_ring", RINGS[1])
    p0_ring_dma = _kv_pair_ring_dma(p0_key_ring, p0_value_ring)
    p1_ring_dma = _kv_pair_ring_dma(p1_key_ring, p1_value_ring)
    p2_ring_dma = _kv_pair_ring_dma(p2_key_ring, p2_value_ring)
    p3_ring_dma = _kv_pair_ring_dma(p3_key_ring, p3_value_ring)
    input_specs = (
        ("kv_k03_c0", 0, 0, PLANES[0].base_bytes // 2),
        ("kv_v03_c0", 0, 1, PLANES[1].base_bytes // 2),
        ("kv_k03_c2", 2, 0, PLANES[0].base_bytes // 2),
        ("kv_v03_c2", 2, 1, PLANES[1].base_bytes // 2),
        ("kv_k47_c4", 4, 0, PLANES[2].base_bytes // 2),
        ("kv_v47_c4", 4, 1, PLANES[3].base_bytes // 2),
        ("kv_k47_c6", 6, 0, PLANES[2].base_bytes // 2),
        ("kv_v47_c6", 6, 1, PLANES[3].base_bytes // 2),
    )
    input_allocations = "\n".join(
        _kv_runtime_input_allocation(name, column, channel)
        for name, column, channel, _offset in input_specs
    )
    input_tasks = "\n".join(
        _kv_runtime_input_task(name, offset, length_elements)
        for name, _column, _channel, offset in input_specs
    )
    input_frees = "\n".join(
        _kv_runtime_input_free(name) for name, _column, _channel, _offset in input_specs
    )
    output_tasks = "\n".join(_context_output_task(group) for group in range(8))
    output_awaits = "\n".join(_context_output_await(group) for group in range(8))
    output_allocations = "\n".join(_context_output_allocation(group) for group in range(8))
    return f"""module {{
  aie.device(npu2) {{
    %shim0 = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %shim2 = aie.tile(2, 0)
    %shim3 = aie.tile(3, 0)
    %shim4 = aie.tile(4, 0)
    %shim5 = aie.tile(5, 0)
    %shim6 = aie.tile(6, 0)
    %shim7 = aie.tile(7, 0)
    %mem0 = aie.tile(0, 1)
    %mem2 = aie.tile(2, 1)
    %mem4 = aie.tile(4, 1)
    %mem6 = aie.tile(6, 1)
{_bounded_attention_buffers(tuple(range(8)))}

    // Quad K/V-pair bounded attention contract:
    //   Keep the duplicate-read quad baseline shape, but consume a finite
    //   history, finalize each worker context, and drain 8 context groups.
{_bounded_attention_kernel_decls()}

{_ring_lock_decls(p0_key_ring, "%mem0")}
{_ring_lock_decls(p0_value_ring, "%mem0")}
{_ring_lock_decls(p1_key_ring, "%mem2")}
{_ring_lock_decls(p1_value_ring, "%mem2")}
{_ring_lock_decls(p2_key_ring, "%mem4")}
{_ring_lock_decls(p2_value_ring, "%mem4")}
{_ring_lock_decls(p3_key_ring, "%mem6")}
{_ring_lock_decls(p3_value_ring, "%mem6")}

{_ring_buffers(p0_key_ring, "%mem0")}
{_ring_buffers(p0_value_ring, "%mem0")}
{_ring_buffers(p1_key_ring, "%mem2")}
{_ring_buffers(p1_value_ring, "%mem2")}
{_ring_buffers(p2_key_ring, "%mem4")}
{_ring_buffers(p2_value_ring, "%mem4")}
{_ring_buffers(p3_key_ring, "%mem6")}
{_ring_buffers(p3_value_ring, "%mem6")}

    aie.flow(%shim0, DMA : 0, %mem0, DMA : 0)
    aie.flow(%shim0, DMA : 1, %mem0, DMA : 1)
    aie.flow(%mem0, DMA : 0, %worker_g0, DMA : 0)
    aie.flow(%mem0, DMA : 1, %worker_g1, DMA : 0)
    aie.flow(%mem0, DMA : 2, %worker_g0, DMA : 1)
    aie.flow(%mem0, DMA : 3, %worker_g1, DMA : 1)
    aie.flow(%shim2, DMA : 0, %mem2, DMA : 0)
    aie.flow(%shim2, DMA : 1, %mem2, DMA : 1)
    aie.flow(%mem2, DMA : 0, %worker_g2, DMA : 0)
    aie.flow(%mem2, DMA : 1, %worker_g3, DMA : 0)
    aie.flow(%mem2, DMA : 2, %worker_g2, DMA : 1)
    aie.flow(%mem2, DMA : 3, %worker_g3, DMA : 1)
    aie.flow(%shim4, DMA : 0, %mem4, DMA : 0)
    aie.flow(%shim4, DMA : 1, %mem4, DMA : 1)
    aie.flow(%mem4, DMA : 0, %worker_g4, DMA : 0)
    aie.flow(%mem4, DMA : 1, %worker_g5, DMA : 0)
    aie.flow(%mem4, DMA : 2, %worker_g4, DMA : 1)
    aie.flow(%mem4, DMA : 3, %worker_g5, DMA : 1)
    aie.flow(%shim6, DMA : 0, %mem6, DMA : 0)
    aie.flow(%shim6, DMA : 1, %mem6, DMA : 1)
    aie.flow(%mem6, DMA : 0, %worker_g6, DMA : 0)
    aie.flow(%mem6, DMA : 1, %worker_g7, DMA : 0)
    aie.flow(%mem6, DMA : 2, %worker_g6, DMA : 1)
    aie.flow(%mem6, DMA : 3, %worker_g7, DMA : 1)
    aie.flow(%worker_g0, DMA : 0, %shim0, DMA : 0)
    aie.flow(%worker_g1, DMA : 0, %shim1, DMA : 0)
    aie.flow(%worker_g2, DMA : 0, %shim2, DMA : 0)
    aie.flow(%worker_g3, DMA : 0, %shim3, DMA : 0)
    aie.flow(%worker_g4, DMA : 0, %shim4, DMA : 0)
    aie.flow(%worker_g5, DMA : 0, %shim5, DMA : 0)
    aie.flow(%worker_g6, DMA : 0, %shim6, DMA : 0)
    aie.flow(%worker_g7, DMA : 0, %shim7, DMA : 0)

{_bounded_attention_worker_core(0, attend_seq_len, tile_size)}
{_bounded_attention_worker_core(1, attend_seq_len, tile_size)}
{_bounded_attention_worker_core(2, attend_seq_len, tile_size)}
{_bounded_attention_worker_core(3, attend_seq_len, tile_size)}
{_bounded_attention_worker_core(4, attend_seq_len, tile_size)}
{_bounded_attention_worker_core(5, attend_seq_len, tile_size)}
{_bounded_attention_worker_core(6, attend_seq_len, tile_size)}
{_bounded_attention_worker_core(7, attend_seq_len, tile_size)}

{_bounded_attention_worker_mem(0)}
{_bounded_attention_worker_mem(1)}
{_bounded_attention_worker_mem(2)}
{_bounded_attention_worker_mem(3)}
{_bounded_attention_worker_mem(4)}
{_bounded_attention_worker_mem(5)}
{_bounded_attention_worker_mem(6)}
{_bounded_attention_worker_mem(7)}

    %memdma0 = aie.memtile_dma(%mem0) {{
{p0_ring_dma}
      aie.end
    }}

    %memdma2 = aie.memtile_dma(%mem2) {{
{p1_ring_dma}
      aie.end
    }}

    %memdma4 = aie.memtile_dma(%mem4) {{
{p2_ring_dma}
      aie.end
    }}

    %memdma6 = aie.memtile_dma(%mem6) {{
{p3_ring_dma}
      aie.end
    }}

    aie.runtime_sequence(%kv_cache: memref<{kv_plane_total_bf16_elements()}xbf16>, %context: memref<{len(WORKER_STREAMS) * Q_ELEMENTS_PER_GROUP}xbf16>) {{
{input_tasks}
{output_tasks}
{output_awaits}
{input_frees}
    }}

{input_allocations}
{output_allocations}
  }}
}}
"""


def build_static_kv_one_pair_two_stage_attention_ingress_contract_mlir(
    attend_seq_len: int,
    tile_size: int = 16,
) -> str:
    length_dwords = history_length_dwords(attend_seq_len, tile_size)
    source_key_ring = _renamed_ring("source_key_ring", RINGS[0])
    source_value_ring = _renamed_ring("source_value_ring", RINGS[1])
    left_key_ring = _renamed_ring("left_key_ring", RINGS[0])
    left_value_ring = _renamed_ring("left_value_ring", RINGS[1])
    right_key_ring = _renamed_ring("right_key_ring", RINGS[0])
    right_value_ring = _renamed_ring("right_value_ring", RINGS[1])
    source_ring_dma = _kv_pair_source_fanout_ring_dma(
        source_key_ring,
        source_value_ring,
    )
    left_ring_dma = _kv_pair_ring_dma(left_key_ring, left_value_ring)
    right_ring_dma = _kv_pair_ring_dma(right_key_ring, right_value_ring)
    patches = "\n".join(
        [
            _writebd(PLANES[0], length_dwords, column=1),
            _writebd(PLANES[1], length_dwords, column=1),
        ]
    )
    return f"""module {{
  aie.device(npu2) {{
    %shim1 = aie.tile(1, 0)
    %mem0 = aie.tile(0, 1)
    %mem1 = aie.tile(1, 1)
    %mem2 = aie.tile(2, 1)
{_dual_pair_attention_buffers((0, 1, 2, 3))}

    // One-read two-stage K/V-pair attention ingress contract:
    //   mem1 reads k03/v03 once from shim column 1.
    //   mem1 forwards full K/V tiles to mem0 and mem2.
    //   mem0 splits to worker_g0/g1; mem2 splits to worker_g2/g3.
{_split_attention_kernel_decls()}

{_ring_lock_decls(source_key_ring, "%mem1")}
{_ring_lock_decls(source_value_ring, "%mem1")}
{_ring_lock_decls(left_key_ring, "%mem0")}
{_ring_lock_decls(left_value_ring, "%mem0")}
{_ring_lock_decls(right_key_ring, "%mem2")}
{_ring_lock_decls(right_value_ring, "%mem2")}

{_ring_buffers(source_key_ring, "%mem1")}
{_ring_buffers(source_value_ring, "%mem1")}
{_ring_buffers(left_key_ring, "%mem0")}
{_ring_buffers(left_value_ring, "%mem0")}
{_ring_buffers(right_key_ring, "%mem2")}
{_ring_buffers(right_value_ring, "%mem2")}

    aie.flow(%shim1, DMA : 0, %mem1, DMA : 0)
    aie.flow(%shim1, DMA : 1, %mem1, DMA : 1)
    aie.flow(%mem1, DMA : 0, %mem0, DMA : 0)
    aie.flow(%mem1, DMA : 1, %mem2, DMA : 0)
    aie.flow(%mem1, DMA : 2, %mem0, DMA : 1)
    aie.flow(%mem1, DMA : 3, %mem2, DMA : 1)
    aie.flow(%mem0, DMA : 0, %worker_g0, DMA : 0)
    aie.flow(%mem0, DMA : 1, %worker_g1, DMA : 0)
    aie.flow(%mem0, DMA : 2, %worker_g0, DMA : 1)
    aie.flow(%mem0, DMA : 3, %worker_g1, DMA : 1)
    aie.flow(%mem2, DMA : 0, %worker_g2, DMA : 0)
    aie.flow(%mem2, DMA : 1, %worker_g3, DMA : 0)
    aie.flow(%mem2, DMA : 2, %worker_g2, DMA : 1)
    aie.flow(%mem2, DMA : 3, %worker_g3, DMA : 1)

{_kv_pair_attention_worker_core(0)}
{_kv_pair_attention_worker_core(1)}
{_kv_pair_attention_worker_core(2)}
{_kv_pair_attention_worker_core(3)}

{_kv_pair_core_dma_worker_mem(0)}
{_kv_pair_core_dma_worker_mem(1)}
{_kv_pair_core_dma_worker_mem(2)}
{_kv_pair_core_dma_worker_mem(3)}

    %memdma1 = aie.memtile_dma(%mem1) {{
{source_ring_dma}
      aie.end
    }}

    %memdma0 = aie.memtile_dma(%mem0) {{
{left_ring_dma}
      aie.end
    }}

    %memdma2 = aie.memtile_dma(%mem2) {{
{right_ring_dma}
      aie.end
    }}

    aie.runtime_sequence(%kv_cache: memref<4194304xbf16>) {{
{patches}
    }}
  }}
}}
"""


def build_static_kv_one_pair_two_stage_attention_bounded_contract_mlir(
    attend_seq_len: int,
    tile_size: int = 16,
) -> str:
    length_elements = history_tiles(attend_seq_len, tile_size) * 4096
    source_key_ring = _renamed_ring("source_key_ring", RINGS[0])
    source_value_ring = _renamed_ring("source_value_ring", RINGS[1])
    left_key_ring = _renamed_ring("left_key_ring", RINGS[0])
    left_value_ring = _renamed_ring("left_value_ring", RINGS[1])
    right_key_ring = _renamed_ring("right_key_ring", RINGS[0])
    right_value_ring = _renamed_ring("right_value_ring", RINGS[1])
    source_ring_dma = _kv_pair_source_fanout_ring_dma(
        source_key_ring,
        source_value_ring,
    )
    left_ring_dma = _kv_pair_ring_dma(left_key_ring, left_value_ring)
    right_ring_dma = _kv_pair_ring_dma(right_key_ring, right_value_ring)
    input_specs = (
        ("kv_k03_c1", 1, 0, PLANES[0].base_bytes // 2),
        ("kv_v03_c1", 1, 1, PLANES[1].base_bytes // 2),
    )
    input_allocations = "\n".join(
        _kv_runtime_input_allocation(name, column, channel)
        for name, column, channel, _offset in input_specs
    )
    input_tasks = "\n".join(
        _kv_runtime_input_task(name, offset, length_elements)
        for name, _column, _channel, offset in input_specs
    )
    input_frees = "\n".join(
        _kv_runtime_input_free(name) for name, _column, _channel, _offset in input_specs
    )
    output_tasks = "\n".join(_context_output_task(group) for group in range(4))
    output_awaits = "\n".join(_context_output_await(group) for group in range(4))
    output_allocations = "\n".join(_context_output_allocation(group) for group in range(4))
    return f"""module {{
  aie.device(npu2) {{
    %shim0 = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %shim2 = aie.tile(2, 0)
    %shim3 = aie.tile(3, 0)
    %mem0 = aie.tile(0, 1)
    %mem1 = aie.tile(1, 1)
    %mem2 = aie.tile(2, 1)
{_bounded_attention_buffers((0, 1, 2, 3))}

    // One-read two-stage bounded K/V-pair attention contract:
    //   mem1 reads k03/v03 once from shim column 1.
    //   mem1 forwards full K/V tiles to mem0 and mem2.
    //   mem0/mem2 split to four workers, then workers finalize and drain context.
{_bounded_attention_kernel_decls()}

{_ring_lock_decls(source_key_ring, "%mem1")}
{_ring_lock_decls(source_value_ring, "%mem1")}
{_ring_lock_decls(left_key_ring, "%mem0")}
{_ring_lock_decls(left_value_ring, "%mem0")}
{_ring_lock_decls(right_key_ring, "%mem2")}
{_ring_lock_decls(right_value_ring, "%mem2")}

{_ring_buffers(source_key_ring, "%mem1")}
{_ring_buffers(source_value_ring, "%mem1")}
{_ring_buffers(left_key_ring, "%mem0")}
{_ring_buffers(left_value_ring, "%mem0")}
{_ring_buffers(right_key_ring, "%mem2")}
{_ring_buffers(right_value_ring, "%mem2")}

    aie.flow(%shim1, DMA : 0, %mem1, DMA : 0)
    aie.flow(%shim1, DMA : 1, %mem1, DMA : 1)
    aie.flow(%mem1, DMA : 0, %mem0, DMA : 0)
    aie.flow(%mem1, DMA : 1, %mem2, DMA : 0)
    aie.flow(%mem1, DMA : 2, %mem0, DMA : 1)
    aie.flow(%mem1, DMA : 3, %mem2, DMA : 1)
    aie.flow(%mem0, DMA : 0, %worker_g0, DMA : 0)
    aie.flow(%mem0, DMA : 1, %worker_g1, DMA : 0)
    aie.flow(%mem0, DMA : 2, %worker_g0, DMA : 1)
    aie.flow(%mem0, DMA : 3, %worker_g1, DMA : 1)
    aie.flow(%mem2, DMA : 0, %worker_g2, DMA : 0)
    aie.flow(%mem2, DMA : 1, %worker_g3, DMA : 0)
    aie.flow(%mem2, DMA : 2, %worker_g2, DMA : 1)
    aie.flow(%mem2, DMA : 3, %worker_g3, DMA : 1)
    aie.flow(%worker_g0, DMA : 0, %shim0, DMA : 0)
    aie.flow(%worker_g1, DMA : 0, %shim1, DMA : 0)
    aie.flow(%worker_g2, DMA : 0, %shim2, DMA : 0)
    aie.flow(%worker_g3, DMA : 0, %shim3, DMA : 0)

{_bounded_attention_worker_core(0, attend_seq_len, tile_size)}
{_bounded_attention_worker_core(1, attend_seq_len, tile_size)}
{_bounded_attention_worker_core(2, attend_seq_len, tile_size)}
{_bounded_attention_worker_core(3, attend_seq_len, tile_size)}

{_bounded_attention_worker_mem(0)}
{_bounded_attention_worker_mem(1)}
{_bounded_attention_worker_mem(2)}
{_bounded_attention_worker_mem(3)}

    %memdma1 = aie.memtile_dma(%mem1) {{
{source_ring_dma}
      aie.end
    }}

    %memdma0 = aie.memtile_dma(%mem0) {{
{left_ring_dma}
      aie.end
    }}

    %memdma2 = aie.memtile_dma(%mem2) {{
{right_ring_dma}
      aie.end
    }}

    aie.runtime_sequence(%kv_cache: memref<{kv_plane_total_bf16_elements()}xbf16>, %context: memref<{len(WORKER_STREAMS) * Q_ELEMENTS_PER_GROUP}xbf16>) {{
{input_tasks}
{output_tasks}
{output_awaits}
{input_frees}
    }}

{input_allocations}
{output_allocations}
  }}
}}
"""


def build_static_kv_full_two_stage_attention_bounded_contract_mlir(
    attend_seq_len: int,
    tile_size: int = 16,
) -> str:
    length_dwords = history_tiles(attend_seq_len, tile_size) * 2048
    p0_source_key_ring = _renamed_ring("p0_source_key_ring", RINGS[0])
    p0_source_value_ring = _renamed_ring("p0_source_value_ring", RINGS[1])
    p0_left_key_ring = _renamed_ring("p0_left_key_ring", RINGS[0])
    p0_left_value_ring = _renamed_ring("p0_left_value_ring", RINGS[1])
    p0_right_key_ring = _renamed_ring("p0_right_key_ring", RINGS[0])
    p0_right_value_ring = _renamed_ring("p0_right_value_ring", RINGS[1])
    p1_source_key_ring = _renamed_ring("p1_source_key_ring", RINGS[0])
    p1_source_value_ring = _renamed_ring("p1_source_value_ring", RINGS[1])
    p1_left_key_ring = _renamed_ring("p1_left_key_ring", RINGS[0])
    p1_left_value_ring = _renamed_ring("p1_left_value_ring", RINGS[1])
    p1_right_key_ring = _renamed_ring("p1_right_key_ring", RINGS[0])
    p1_right_value_ring = _renamed_ring("p1_right_value_ring", RINGS[1])
    p0_source_ring_dma = _kv_pair_source_fanout_ring_dma(
        p0_source_key_ring,
        p0_source_value_ring,
    )
    p0_left_ring_dma = _kv_pair_ring_dma(p0_left_key_ring, p0_left_value_ring)
    p0_right_ring_dma = _kv_pair_ring_dma(p0_right_key_ring, p0_right_value_ring)
    p1_source_ring_dma = _kv_pair_source_fanout_ring_dma(
        p1_source_key_ring,
        p1_source_value_ring,
    )
    p1_left_ring_dma = _kv_pair_ring_dma(p1_left_key_ring, p1_left_value_ring)
    p1_right_ring_dma = _kv_pair_ring_dma(p1_right_key_ring, p1_right_value_ring)
    input_specs = (
        ("kv_k03_c1", 1, 0, 0, PLANES[0].base_bytes),
        ("kv_v03_c1", 1, 1, 1, PLANES[1].base_bytes),
        ("kv_k47_c5", 5, 0, 0, PLANES[2].base_bytes),
        ("kv_v47_c5", 5, 1, 1, PLANES[3].base_bytes),
    )
    input_tasks = "\n".join(
        _npu_input_sequence(name, column, channel, bd_id, offset, length_dwords)
        for name, column, channel, bd_id, offset in input_specs
    )
    output_tasks = "\n".join(
        _npu_context_sequence(group, bd_id=2 if group in (1, 5) else 0)
        for group in range(8)
    )
    output_awaits = "\n".join(_npu_sync_s2mm(group) for group in range(8))
    return f"""module {{
  aie.device(npu2) {{
    %shim0 = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %shim2 = aie.tile(2, 0)
    %shim3 = aie.tile(3, 0)
    %shim4 = aie.tile(4, 0)
    %shim5 = aie.tile(5, 0)
    %shim6 = aie.tile(6, 0)
    %shim7 = aie.tile(7, 0)
    %mem0 = aie.tile(0, 1)
    %mem1 = aie.tile(1, 1)
    %mem2 = aie.tile(2, 1)
    %mem4 = aie.tile(4, 1)
    %mem5 = aie.tile(5, 1)
    %mem6 = aie.tile(6, 1)
{_bounded_attention_buffers(tuple(range(8)))}

    // Full two-stage bounded K/V-pair attention contract:
    //   mem1 reads k03/v03 once, then fans out to mem0/mem2.
    //   mem5 reads k47/v47 once, then fans out to mem4/mem6.
    //   Second-stage memtiles split K/V halves to eight attention workers.
    //   Runtime uses explicit writebd/address_patch/push_queue/sync.
{_bounded_attention_kernel_decls()}

{_ring_lock_decls(p0_source_key_ring, "%mem1")}
{_ring_lock_decls(p0_source_value_ring, "%mem1")}
{_ring_lock_decls(p0_left_key_ring, "%mem0")}
{_ring_lock_decls(p0_left_value_ring, "%mem0")}
{_ring_lock_decls(p0_right_key_ring, "%mem2")}
{_ring_lock_decls(p0_right_value_ring, "%mem2")}
{_ring_lock_decls(p1_source_key_ring, "%mem5")}
{_ring_lock_decls(p1_source_value_ring, "%mem5")}
{_ring_lock_decls(p1_left_key_ring, "%mem4")}
{_ring_lock_decls(p1_left_value_ring, "%mem4")}
{_ring_lock_decls(p1_right_key_ring, "%mem6")}
{_ring_lock_decls(p1_right_value_ring, "%mem6")}

{_ring_buffers(p0_source_key_ring, "%mem1")}
{_ring_buffers(p0_source_value_ring, "%mem1")}
{_ring_buffers(p0_left_key_ring, "%mem0")}
{_ring_buffers(p0_left_value_ring, "%mem0")}
{_ring_buffers(p0_right_key_ring, "%mem2")}
{_ring_buffers(p0_right_value_ring, "%mem2")}
{_ring_buffers(p1_source_key_ring, "%mem5")}
{_ring_buffers(p1_source_value_ring, "%mem5")}
{_ring_buffers(p1_left_key_ring, "%mem4")}
{_ring_buffers(p1_left_value_ring, "%mem4")}
{_ring_buffers(p1_right_key_ring, "%mem6")}
{_ring_buffers(p1_right_value_ring, "%mem6")}

    aie.flow(%shim1, DMA : 0, %mem1, DMA : 0)
    aie.flow(%shim1, DMA : 1, %mem1, DMA : 1)
    aie.flow(%mem1, DMA : 0, %mem0, DMA : 0)
    aie.flow(%mem1, DMA : 1, %mem2, DMA : 0)
    aie.flow(%mem1, DMA : 2, %mem0, DMA : 1)
    aie.flow(%mem1, DMA : 3, %mem2, DMA : 1)
    aie.flow(%mem0, DMA : 0, %worker_g0, DMA : 0)
    aie.flow(%mem0, DMA : 1, %worker_g1, DMA : 0)
    aie.flow(%mem0, DMA : 2, %worker_g0, DMA : 1)
    aie.flow(%mem0, DMA : 3, %worker_g1, DMA : 1)
    aie.flow(%mem2, DMA : 0, %worker_g2, DMA : 0)
    aie.flow(%mem2, DMA : 1, %worker_g3, DMA : 0)
    aie.flow(%mem2, DMA : 2, %worker_g2, DMA : 1)
    aie.flow(%mem2, DMA : 3, %worker_g3, DMA : 1)
    aie.flow(%shim5, DMA : 0, %mem5, DMA : 0)
    aie.flow(%shim5, DMA : 1, %mem5, DMA : 1)
    aie.flow(%mem5, DMA : 0, %mem4, DMA : 0)
    aie.flow(%mem5, DMA : 1, %mem6, DMA : 0)
    aie.flow(%mem5, DMA : 2, %mem4, DMA : 1)
    aie.flow(%mem5, DMA : 3, %mem6, DMA : 1)
    aie.flow(%mem4, DMA : 0, %worker_g4, DMA : 0)
    aie.flow(%mem4, DMA : 1, %worker_g5, DMA : 0)
    aie.flow(%mem4, DMA : 2, %worker_g4, DMA : 1)
    aie.flow(%mem4, DMA : 3, %worker_g5, DMA : 1)
    aie.flow(%mem6, DMA : 0, %worker_g6, DMA : 0)
    aie.flow(%mem6, DMA : 1, %worker_g7, DMA : 0)
    aie.flow(%mem6, DMA : 2, %worker_g6, DMA : 1)
    aie.flow(%mem6, DMA : 3, %worker_g7, DMA : 1)
    aie.flow(%worker_g0, DMA : 0, %shim0, DMA : 0)
    aie.flow(%worker_g1, DMA : 0, %shim1, DMA : 0)
    aie.flow(%worker_g2, DMA : 0, %shim2, DMA : 0)
    aie.flow(%worker_g3, DMA : 0, %shim3, DMA : 0)
    aie.flow(%worker_g4, DMA : 0, %shim4, DMA : 0)
    aie.flow(%worker_g5, DMA : 0, %shim5, DMA : 0)
    aie.flow(%worker_g6, DMA : 0, %shim6, DMA : 0)
    aie.flow(%worker_g7, DMA : 0, %shim7, DMA : 0)

{_bounded_attention_worker_core(0, attend_seq_len, tile_size)}
{_bounded_attention_worker_core(1, attend_seq_len, tile_size)}
{_bounded_attention_worker_core(2, attend_seq_len, tile_size)}
{_bounded_attention_worker_core(3, attend_seq_len, tile_size)}
{_bounded_attention_worker_core(4, attend_seq_len, tile_size)}
{_bounded_attention_worker_core(5, attend_seq_len, tile_size)}
{_bounded_attention_worker_core(6, attend_seq_len, tile_size)}
{_bounded_attention_worker_core(7, attend_seq_len, tile_size)}

{_bounded_attention_worker_mem(0)}
{_bounded_attention_worker_mem(1)}
{_bounded_attention_worker_mem(2)}
{_bounded_attention_worker_mem(3)}
{_bounded_attention_worker_mem(4)}
{_bounded_attention_worker_mem(5)}
{_bounded_attention_worker_mem(6)}
{_bounded_attention_worker_mem(7)}

    %memdma1 = aie.memtile_dma(%mem1) {{
{p0_source_ring_dma}
      aie.end
    }}

    %memdma0 = aie.memtile_dma(%mem0) {{
{p0_left_ring_dma}
      aie.end
    }}

    %memdma2 = aie.memtile_dma(%mem2) {{
{p0_right_ring_dma}
      aie.end
    }}

    %memdma5 = aie.memtile_dma(%mem5) {{
{p1_source_ring_dma}
      aie.end
    }}

    %memdma4 = aie.memtile_dma(%mem4) {{
{p1_left_ring_dma}
      aie.end
    }}

    %memdma6 = aie.memtile_dma(%mem6) {{
{p1_right_ring_dma}
      aie.end
    }}

    aie.runtime_sequence(%kv_cache: memref<{kv_plane_total_bf16_elements()}xbf16>, %context: memref<{len(WORKER_STREAMS) * Q_ELEMENTS_PER_GROUP}xbf16>) {{
{input_tasks}
{output_tasks}
{output_awaits}
    }}
  }}
}}
"""


def check_static_kv_reader_contract_mlir(text: str) -> list[str]:
    messages: list[str] = []
    if "aie.memtile_dma(%mem)" not in text:
        raise ValueError("missing memtile DMA region")
    if text.count("aiex.npu.writebd") != len(PLANES):
        raise ValueError("expected exactly four npu.writebd ops")
    if text.count("aie.core(%worker_g") != len(WORKER_STREAMS):
        raise ValueError("expected one attention core per worker stream")
    if text.count("link_with") != 3:
        raise ValueError("expected three attention kernel declarations")
    if text.count("func.call @llama_chunked_attention_init_f32") != len(WORKER_STREAMS):
        raise ValueError("expected one attention init call per worker stream")
    if text.count("func.call @qwen_plane_group_attention_update_bf16") != len(WORKER_STREAMS):
        raise ValueError("expected one attention update call per worker stream")
    if text.count("func.call @llama_chunked_attention_finalize_bf16") != len(WORKER_STREAMS):
        raise ValueError("expected one attention finalize call per worker stream")
    if text.count(f"stack_size = {ATTENTION_STACK_SIZE} : i32") != len(WORKER_STREAMS):
        raise ValueError("expected attention stack size on every core")
    expected_fifo_count = len(Q_ROUTES) + len(KV_ROUTES) + len(OUT_ROUTES)
    if text.count("// fifo_route qwen_layer_") != expected_fifo_count:
        raise ValueError("expected q/kv/out FIFO route contract for every worker")
    if "Do not lower the KV routes to high-level aie.objectfifo here" not in text:
        raise ValueError("missing static BD route warning")
    for memref in (
        f"memref<{Q_CURRENT_ELEMENTS_PER_GROUP}xbf16>",
        f"memref<{PLANE_GROUP_PAIR_CHUNK_ELEMENTS}xbf16>",
        f"memref<{STATE_ELEMENTS}xf32>",
        f"memref<{Q_ELEMENTS_PER_GROUP}xf32>",
        f"memref<{Q_ELEMENTS_PER_GROUP}xbf16>",
    ):
        if memref not in text:
            raise ValueError(f"missing attention ABI memref {memref}")

    for ring in RINGS:
        for lock_id in ring.mlir_locks:
            if f"aie.lock(%mem, {lock_id})" not in text:
                raise ValueError(f"missing {ring.name} MLIR lock {lock_id}")
        comment = (
            f"// {ring.name}: fastflow_reference_locks="
            f"{ring.fastflow_locks[0]},{ring.fastflow_locks[1]},{ring.fastflow_locks[2]}"
        )
        if comment not in text:
            raise ValueError(f"missing {ring.name} FastFlow lock annotation")

    rows = re.findall(
        r"memref<(\d+)xbf16>, 0, (\d+)\).*bd_id = (\d+) : i32, next_bd_id = (\d+) : i32",
        text,
    )
    found_lengths = {int(bd_id): int(length) for _memref_len, length, bd_id, _next_bd in rows}
    if found_lengths != EXPECTED_BD_LENGTHS:
        raise ValueError(f"unexpected BD length table: {found_lengths}")

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
    found_pairs = {int(bd_id): int(next_bd) for _memref_len, _length, bd_id, next_bd in rows}
    if found_pairs != expected_pairs:
        raise ValueError(f"unexpected next-BD table: {found_pairs}")

    for plane in PLANES:
        pattern = (
            rf"// {plane.name}:.*?bd_id = (\d+) : i32, "
            rf"buffer_length = (\d+) : i32, buffer_offset = (\d+) : i32"
        )
        match = re.search(pattern, text, flags=re.S)
        if match is None:
            raise ValueError(f"missing writebd fields for {plane.name}")
        found_bd, found_len, found_offset = map(int, match.groups())
        if found_bd != plane.shim_bd_id:
            raise ValueError(f"{plane.name} bd_id {found_bd} != {plane.shim_bd_id}")
        if found_offset != plane.base_bytes:
            raise ValueError(f"{plane.name} offset {found_offset} != {plane.base_bytes}")
        if found_len % 0x1000 != 0:
            raise ValueError(f"{plane.name} length should be 16-token tile aligned")

    if "use_next_bd = 0 : i32" not in text:
        raise ValueError("dynamic shim descriptor should not chain by default")

    for stream in WORKER_STREAMS:
        q_col, q_row = stream.q_split_tile
        p_col, p_row = stream.plane_split_tile
        w_col, w_row = stream.worker_tile
        expected = (
            f"worker_g{stream.group}: "
            f"planes={stream.key_plane}/{stream.value_plane} "
            f"pair={stream.plane_pair} group_in_plane={stream.group_in_plane} "
            f"q_split=c{q_col}r{q_row} plane_split=c{p_col}r{p_row} "
            f"worker=c{w_col}r{w_row}"
        )
        if expected not in text:
            raise ValueError(f"missing worker stream contract: {expected}")

    for route in (*Q_ROUTES, *KV_ROUTES, *OUT_ROUTES):
        producer = _tile_value_name(route.producer_tile)
        consumer = _tile_value_name(route.consumer_tile)
        declaration = (
            f"fifo_route {route.name}: {producer} -> {consumer}, "
            f"elements={route.elements}, depth={route.depth}"
        )
        if declaration not in text:
            raise ValueError(f"missing FIFO route declaration: {declaration}")

    messages.append(
        "ring_a mlir_locks=4->5->6->4 fastflow_ref=64->65->66->64 bd=0/1,2/3,24/25"
    )
    messages.append(
        "ring_b mlir_locks=7->8->9->7 fastflow_ref=67->68->69->67 bd=26/27,4/5,28/29"
    )
    messages.append("planes=k03,v03,k47,v47; length is ceil(L/16)*0x1000 dwords")
    messages.append("workers=g0..g3 consume k03/v03; g4..g7 consume k47/v47")
    messages.append(
        "attention workers call init/update/finalize; FIFO route contract is explicit but kept as BD/lock work"
    )
    return messages


def check_static_kv_core_dma_contract_mlir(text: str) -> list[str]:
    messages: list[str] = []
    if "core-DMA contract: k03 plane -> row1 ring_a -> worker_g0/g1 half buffers" not in text:
        raise ValueError("missing core-DMA contract marker")
    if text.count("aie.mem(%worker_g") != 2:
        raise ValueError("expected two worker memory DMA regions")
    if text.count("aie.core(%worker_g") != 2:
        raise ValueError("expected two worker cores")
    if text.count("aie.dma_start(S2MM, 0") != 3:
        raise ValueError("expected one memtile load and two worker receive S2MM DMAs")
    if text.count("aiex.npu.writebd") != 1:
        raise ValueError("expected one k03 runtime writebd")
    if "bd_id = 0 : i32, next_bd_id = 1 : i32" not in text:
        raise ValueError("missing ping BD")
    if "bd_id = 1 : i32, next_bd_id = 0 : i32" not in text:
        raise ValueError("missing pong BD")
    for group in (0, 1):
        for lock_name in ("empty", "full"):
            if f"%worker_g{group}_{lock_name} = aie.lock(%worker_g{group}" not in text:
                raise ValueError(f"missing worker_g{group}_{lock_name} lock")
        if f"aie.flow(%mem, DMA : {group}, %worker_g{group}, DMA : 0)" not in text:
            raise ValueError(f"missing memtile-to-worker flow for worker_g{group}")
        if f"memref<2048xbf16>, 0, 2048) {{bd_id = 0" not in text:
            raise ValueError("missing worker receive ping length")

    messages.append("core_dma=ring_a half0/half1 -> worker_g0/g1 local ping-pong buffers")
    messages.append("runtime_writebd=k03 only; length is ceil(L/16)*0x1000 dwords")
    return messages


def check_static_kv_pair_core_dma_contract_mlir(text: str) -> list[str]:
    messages: list[str] = []
    if "K/V-pair core-DMA contract" not in text:
        raise ValueError("missing K/V-pair core-DMA contract marker")
    if text.count("aie.mem(%worker_g") != 2:
        raise ValueError("expected two worker memory DMA regions")
    if text.count("aie.core(%worker_g") != 2:
        raise ValueError("expected two worker cores")
    if text.count("aiex.npu.writebd") != 2:
        raise ValueError("expected k03 and v03 runtime writebd descriptors")
    for ring in (RINGS[0], RINGS[1]):
        if f"%{ring.name}_load = aie.dma_start(S2MM, {ring.load_channel}" not in text:
            raise ValueError(f"missing {ring.name} load DMA")
        if f"%{ring.name}_half0 = aie.dma_start(MM2S, {ring.half0_channel}" not in text:
            raise ValueError(f"missing {ring.name} half0 DMA")
        if f"%{ring.name}_half1 = aie.dma_start(MM2S, {ring.half1_channel}" not in text:
            raise ValueError(f"missing {ring.name} half1 DMA")
    for plane in (PLANES[0], PLANES[1]):
        if f"// {plane.name}:" not in text:
            raise ValueError(f"missing {plane.name} runtime writebd comment")
    expected_flows = (
        "aie.flow(%mem, DMA : 0, %worker_g0, DMA : 0)",
        "aie.flow(%mem, DMA : 1, %worker_g1, DMA : 0)",
        "aie.flow(%mem, DMA : 2, %worker_g0, DMA : 1)",
        "aie.flow(%mem, DMA : 3, %worker_g1, DMA : 1)",
    )
    for flow in expected_flows:
        if flow not in text:
            raise ValueError(f"missing flow: {flow}")
    for group in (0, 1):
        for role in ("k", "v"):
            for lock_name in ("empty", "full"):
                declaration = f"%worker_g{group}_{role}_{lock_name} = aie.lock(%worker_g{group}"
                if declaration not in text:
                    raise ValueError(f"missing worker_g{group}_{role}_{lock_name} lock")
            if f"%worker_g{group}_{role}_rx = aie.dma_start(S2MM" not in text:
                raise ValueError(f"missing worker_g{group}_{role} receive DMA")
        if f"aie.use_lock(%worker_g{group}_k_full, AcquireGreaterEqual, 1)" not in text:
            raise ValueError(f"missing worker_g{group} K full acquire")
        if f"aie.use_lock(%worker_g{group}_v_full, AcquireGreaterEqual, 1)" not in text:
            raise ValueError(f"missing worker_g{group} V full acquire")

    messages.append("kv_pair=ring_a(k03) + ring_b(v03) -> worker_g0/g1 K/V ping-pong buffers")
    messages.append("worker input DMA channels: K uses DMA0, V uses DMA1")
    messages.append("runtime_writebd=k03,v03; each length is ceil(L/16)*0x1000 dwords")
    return messages


def check_static_kv_pair_attention_ingress_contract_mlir(text: str) -> list[str]:
    messages = check_static_kv_pair_core_dma_contract_mlir(text)
    if "K/V-pair attention ingress contract" not in text:
        raise ValueError("missing K/V-pair attention ingress contract marker")
    if text.count("link_with") != 2:
        raise ValueError("expected split attention init/update kernel declarations")
    if text.count("func.call @llama_chunked_attention_init_f32") != 2:
        raise ValueError("expected one attention init call per worker")
    if text.count("func.call @qwen_plane_group_attention_update_split_bf16") != 4:
        raise ValueError("expected ping and pong split attention update calls per worker")
    if text.count(f"stack_size = {ATTENTION_STACK_SIZE} : i32") != 2:
        raise ValueError("expected attention stack size on both worker cores")
    for group in (0, 1):
        for memref in (
            f"%q_current_g{group}",
            f"%state_g{group}",
            f"%acc_g{group}",
            f"%worker_g{group}_k_ping",
            f"%worker_g{group}_k_pong",
            f"%worker_g{group}_v_ping",
            f"%worker_g{group}_v_pong",
        ):
            if memref not in text:
                raise ValueError(f"missing worker_g{group} attention buffer {memref}")
        ping_call = (
            f"func.call @qwen_plane_group_attention_update_split_bf16"
            f"(%q_current_g{group}, %worker_g{group}_k_ping, %worker_g{group}_v_ping"
        )
        pong_call = (
            f"func.call @qwen_plane_group_attention_update_split_bf16"
            f"(%q_current_g{group}, %worker_g{group}_k_pong, %worker_g{group}_v_pong"
        )
        if ping_call not in text:
            raise ValueError(f"missing worker_g{group} ping split attention call")
        if pong_call not in text:
            raise ValueError(f"missing worker_g{group} pong split attention call")

    messages.append("attention_ingress=worker no-op loop replaced by split K/V update ABI")
    messages.append("split_update=ping and pong K/V buffers call qwen_plane_group_attention_update_split_bf16")
    return messages


def check_static_kv_dual_pair_attention_ingress_contract_mlir(text: str) -> list[str]:
    messages: list[str] = []
    if "Dual K/V-pair attention ingress contract" not in text:
        raise ValueError("missing dual K/V-pair attention ingress marker")
    if text.count("aie.memtile_dma(%mem") != 2:
        raise ValueError("expected two row1 memtile DMA regions")
    if text.count("aie.mem(%worker_g") != 4:
        raise ValueError("expected four worker memory DMA regions")
    if text.count("aie.core(%worker_g") != 4:
        raise ValueError("expected four attention worker cores")
    if text.count("aiex.npu.writebd") != 4:
        raise ValueError("expected four runtime writebd descriptors")
    if text.count("link_with") != 2:
        raise ValueError("expected split attention init/update kernel declarations")
    if text.count("func.call @llama_chunked_attention_init_f32") != 4:
        raise ValueError("expected one attention init call per worker")
    if text.count("func.call @qwen_plane_group_attention_update_split_bf16") != 8:
        raise ValueError("expected ping and pong split attention update calls per worker")
    if text.count(f"stack_size = {ATTENTION_STACK_SIZE} : i32") != 4:
        raise ValueError("expected attention stack size on four worker cores")

    expected_flows = (
        "aie.flow(%shim0, DMA : 0, %mem0, DMA : 0)",
        "aie.flow(%shim0, DMA : 1, %mem0, DMA : 1)",
        "aie.flow(%shim4, DMA : 0, %mem4, DMA : 0)",
        "aie.flow(%shim4, DMA : 1, %mem4, DMA : 1)",
        "aie.flow(%mem0, DMA : 0, %worker_g0, DMA : 0)",
        "aie.flow(%mem0, DMA : 1, %worker_g1, DMA : 0)",
        "aie.flow(%mem0, DMA : 2, %worker_g0, DMA : 1)",
        "aie.flow(%mem0, DMA : 3, %worker_g1, DMA : 1)",
        "aie.flow(%mem4, DMA : 0, %worker_g4, DMA : 0)",
        "aie.flow(%mem4, DMA : 1, %worker_g5, DMA : 0)",
        "aie.flow(%mem4, DMA : 2, %worker_g4, DMA : 1)",
        "aie.flow(%mem4, DMA : 3, %worker_g5, DMA : 1)",
    )
    for flow in expected_flows:
        if flow not in text:
            raise ValueError(f"missing flow: {flow}")

    expected_planes = {
        "k03": (PLANES[0].shim_bd_id, PLANES[0].base_bytes, 0),
        "v03": (PLANES[1].shim_bd_id, PLANES[1].base_bytes, 0),
        "k47": (PLANES[2].shim_bd_id, PLANES[2].base_bytes, 4),
        "v47": (PLANES[3].shim_bd_id, PLANES[3].base_bytes, 4),
    }
    for plane_name, (bd_id, offset, column) in expected_planes.items():
        pattern = (
            rf"// {plane_name}:.*?bd_id = {bd_id} : i32, "
            rf"buffer_length = (\d+) : i32, buffer_offset = {offset} : i32, "
            rf".*?column = {column} : i32"
        )
        match = re.search(pattern, text, flags=re.S)
        if match is None:
            raise ValueError(f"missing {plane_name} writebd fields")
        if int(match.group(1)) % KV_TILE_LENGTH_DWORDS != 0:
            raise ValueError(f"{plane_name} length should be 16-token tile aligned")

    for ring_name in (
        "left_key_ring",
        "left_value_ring",
        "right_key_ring",
        "right_value_ring",
    ):
        if f"%{ring_name}_load = aie.dma_start(S2MM" not in text:
            raise ValueError(f"missing {ring_name} load DMA")
        for suffix in ("empty", "loaded", "split"):
            if f"%{ring_name}_{suffix} = aie.lock(" not in text:
                raise ValueError(f"missing {ring_name}_{suffix} lock")

    for group in (0, 1, 4, 5):
        for role in ("k", "v"):
            for lock_name in ("empty", "full"):
                declaration = f"%worker_g{group}_{role}_{lock_name} = aie.lock(%worker_g{group}"
                if declaration not in text:
                    raise ValueError(f"missing worker_g{group}_{role}_{lock_name} lock")
            if f"%worker_g{group}_{role}_rx = aie.dma_start(S2MM" not in text:
                raise ValueError(f"missing worker_g{group}_{role} receive DMA")
        ping_call = (
            f"func.call @qwen_plane_group_attention_update_split_bf16"
            f"(%q_current_g{group}, %worker_g{group}_k_ping, %worker_g{group}_v_ping"
        )
        pong_call = (
            f"func.call @qwen_plane_group_attention_update_split_bf16"
            f"(%q_current_g{group}, %worker_g{group}_k_pong, %worker_g{group}_v_pong"
        )
        if ping_call not in text:
            raise ValueError(f"missing worker_g{group} ping split attention call")
        if pong_call not in text:
            raise ValueError(f"missing worker_g{group} pong split attention call")

    messages.append("dual_pair=mem0(k03/v03)->worker_g0/g1 and mem4(k47/v47)->worker_g4/g5")
    messages.append("runtime_writebd=4 descriptors; k03/v03 column 0, k47/v47 column 4")
    messages.append("attention_ingress=4 workers call split K/V update on local ping-pong buffers")
    return messages


def check_static_kv_quad_pair_attention_ingress_contract_mlir(text: str) -> list[str]:
    messages: list[str] = []
    if "Quad K/V-pair attention ingress contract" not in text:
        raise ValueError("missing quad K/V-pair attention ingress marker")
    if text.count("aie.memtile_dma(%mem") != 4:
        raise ValueError("expected four row1 memtile DMA regions")
    if text.count("aie.mem(%worker_g") != 8:
        raise ValueError("expected eight worker memory DMA regions")
    if text.count("aie.core(%worker_g") != 8:
        raise ValueError("expected eight attention worker cores")
    if text.count("aiex.npu.writebd") != 8:
        raise ValueError("expected eight runtime writebd descriptors")
    if text.count("link_with") != 2:
        raise ValueError("expected split attention init/update kernel declarations")
    if text.count("func.call @llama_chunked_attention_init_f32") != 8:
        raise ValueError("expected one attention init call per worker")
    if text.count("func.call @qwen_plane_group_attention_update_split_bf16") != 16:
        raise ValueError("expected ping and pong split attention update calls per worker")
    if text.count(f"stack_size = {ATTENTION_STACK_SIZE} : i32") != 8:
        raise ValueError("expected attention stack size on eight worker cores")

    for column in (0, 2, 4, 6):
        if f"%shim{column} = aie.tile({column}, 0)" not in text:
            raise ValueError(f"missing shim{column} tile")
        if f"%mem{column} = aie.tile({column}, 1)" not in text:
            raise ValueError(f"missing mem{column} tile")
        for dma_channel in (0, 1):
            flow = f"aie.flow(%shim{column}, DMA : {dma_channel}, %mem{column}, DMA : {dma_channel})"
            if flow not in text:
                raise ValueError(f"missing flow: {flow}")

    for mem_column, group0, group1 in ((0, 0, 1), (2, 2, 3), (4, 4, 5), (6, 6, 7)):
        expected_flows = (
            f"aie.flow(%mem{mem_column}, DMA : 0, %worker_g{group0}, DMA : 0)",
            f"aie.flow(%mem{mem_column}, DMA : 1, %worker_g{group1}, DMA : 0)",
            f"aie.flow(%mem{mem_column}, DMA : 2, %worker_g{group0}, DMA : 1)",
            f"aie.flow(%mem{mem_column}, DMA : 3, %worker_g{group1}, DMA : 1)",
        )
        for flow in expected_flows:
            if flow not in text:
                raise ValueError(f"missing flow: {flow}")

    expected_patches = (
        ("k03", PLANES[0].shim_bd_id, PLANES[0].base_bytes, 0),
        ("v03", PLANES[1].shim_bd_id, PLANES[1].base_bytes, 0),
        ("k03", PLANES[0].shim_bd_id, PLANES[0].base_bytes, 2),
        ("v03", PLANES[1].shim_bd_id, PLANES[1].base_bytes, 2),
        ("k47", PLANES[2].shim_bd_id, PLANES[2].base_bytes, 4),
        ("v47", PLANES[3].shim_bd_id, PLANES[3].base_bytes, 4),
        ("k47", PLANES[2].shim_bd_id, PLANES[2].base_bytes, 6),
        ("v47", PLANES[3].shim_bd_id, PLANES[3].base_bytes, 6),
    )
    for plane_name, bd_id, offset, column in expected_patches:
        pattern = (
            rf"// {plane_name}:.*?\n"
            rf"\s+aiex\.npu\.writebd \{{(?:(?!\n\s+// ).)*?"
            rf"bd_id = {bd_id} : i32, buffer_length = (\d+) : i32, "
            rf"buffer_offset = {offset} : i32, (?:(?!\n\s+// ).)*?"
            rf"column = {column} : i32"
        )
        match = re.search(pattern, text, flags=re.S)
        if match is None:
            raise ValueError(f"missing {plane_name} column {column} writebd fields")
        if int(match.group(1)) % KV_TILE_LENGTH_DWORDS != 0:
            raise ValueError(f"{plane_name} column {column} length should be tile aligned")

    for prefix in ("p0", "p1", "p2", "p3"):
        for role in ("key", "value"):
            ring_name = f"{prefix}_{role}_ring"
            if f"%{ring_name}_load = aie.dma_start(S2MM" not in text:
                raise ValueError(f"missing {ring_name} load DMA")
            for suffix in ("empty", "loaded", "split"):
                if f"%{ring_name}_{suffix} = aie.lock(" not in text:
                    raise ValueError(f"missing {ring_name}_{suffix} lock")

    for group in range(8):
        for role in ("k", "v"):
            for lock_name in ("empty", "full"):
                declaration = f"%worker_g{group}_{role}_{lock_name} = aie.lock(%worker_g{group}"
                if declaration not in text:
                    raise ValueError(f"missing worker_g{group}_{role}_{lock_name} lock")
            if f"%worker_g{group}_{role}_rx = aie.dma_start(S2MM" not in text:
                raise ValueError(f"missing worker_g{group}_{role} receive DMA")
        ping_call = (
            f"func.call @qwen_plane_group_attention_update_split_bf16"
            f"(%q_current_g{group}, %worker_g{group}_k_ping, %worker_g{group}_v_ping"
        )
        pong_call = (
            f"func.call @qwen_plane_group_attention_update_split_bf16"
            f"(%q_current_g{group}, %worker_g{group}_k_pong, %worker_g{group}_v_pong"
        )
        if ping_call not in text:
            raise ValueError(f"missing worker_g{group} ping split attention call")
        if pong_call not in text:
            raise ValueError(f"missing worker_g{group} pong split attention call")

    messages.append("quad_pair=mem0/mem2 read k03/v03; mem4/mem6 read k47/v47")
    messages.append("runtime_writebd=8 descriptors; each plane pair is duplicated across two shim columns")
    messages.append("attention_ingress=8 workers call split K/V update on local ping-pong buffers")
    return messages


def check_static_kv_quad_pair_attention_bounded_contract_mlir(text: str) -> list[str]:
    messages: list[str] = []
    if "Quad K/V-pair bounded attention contract" not in text:
        raise ValueError("missing quad bounded attention marker")
    if text.count("aie.memtile_dma(%mem") != 4:
        raise ValueError("expected four row1 memtile DMA regions")
    if text.count("aie.mem(%worker_g") != 8:
        raise ValueError("expected eight worker memory DMA regions")
    if text.count("aie.core(%worker_g") != 8:
        raise ValueError("expected eight attention worker cores")
    if text.count("aiex.npu.writebd") != 0:
        raise ValueError("bounded numeric contract should use runtime input tasks, not writebd")
    if text.count("aiex.dma_configure_task_for @kv_") != 8:
        raise ValueError("expected eight KV runtime input tasks")
    if text.count("aie.shim_dma_allocation @kv_") != 8:
        raise ValueError("expected eight KV input shim allocations")
    if text.count("aiex.dma_configure_task_for @context_g") != 8:
        raise ValueError("expected eight context output runtime tasks")
    if text.count("aie.shim_dma_allocation @context_g") != 8:
        raise ValueError("expected eight context output shim allocations")
    if text.count("func.call @qwen_zero_q_current_bf16") != 8:
        raise ValueError("expected deterministic q_current init on every worker")
    if text.count("func.call @llama_chunked_attention_init_f32") != 8:
        raise ValueError("expected one attention init call per worker")
    if text.count("func.call @llama_chunked_attention_finalize_bf16") != 8:
        raise ValueError("expected one attention finalize call per worker")
    if text.count("func.call @qwen_plane_group_attention_update_split_bf16") < 8:
        raise ValueError("expected bounded split attention updates")
    if text.count(f"stack_size = {ATTENTION_STACK_SIZE} : i32") != 8:
        raise ValueError("expected attention stack size on eight worker cores")

    for group in range(8):
        if f"%worker_g{group}_out_empty = aie.lock(%worker_g{group}" not in text:
            raise ValueError(f"missing worker_g{group} output empty lock")
        if f"%worker_g{group}_out_full = aie.lock(%worker_g{group}" not in text:
            raise ValueError(f"missing worker_g{group} output full lock")
        if f"%worker_g{group}_out_tx = aie.dma_start(MM2S, 0" not in text:
            raise ValueError(f"missing worker_g{group} context transmit DMA")
        if f"aie.flow(%worker_g{group}, DMA : 0, %shim{group}, DMA : 0)" not in text:
            raise ValueError(f"missing worker_g{group} to shim{group} context flow")
        if f"aiex.dma_await_task(%context_task_g{group})" not in text:
            raise ValueError(f"missing context_task_g{group} await")
        if f"memref<{len(WORKER_STREAMS) * Q_ELEMENTS_PER_GROUP}xbf16>, {group * Q_ELEMENTS_PER_GROUP}, {Q_ELEMENTS_PER_GROUP}" not in text:
            raise ValueError(f"missing context offset for group {group}")

    for name in (
        "kv_k03_c0",
        "kv_v03_c0",
        "kv_k03_c2",
        "kv_v03_c2",
        "kv_k47_c4",
        "kv_v47_c4",
        "kv_k47_c6",
        "kv_v47_c6",
    ):
        if f"%{name}_task = aiex.dma_configure_task_for @{name}_alloc" not in text:
            raise ValueError(f"missing {name} input task")
        if f"aiex.dma_free_task(%{name}_task)" not in text:
            raise ValueError(f"missing {name} input free")

    messages.append("bounded_attention=finite KV tile updates, finalize_bf16, host-visible context egress")
    messages.append("q_current=deterministic local zero init; no third worker input DMA")
    messages.append("kv_input=8 coarse runtime MM2S tasks feed the existing duplicate-read row1 rings")
    messages.append("context_egress=8 worker MM2S streams drained by 8 shim S2MM runtime tasks")
    return messages


def _check_one_pair_two_stage_topology(text: str) -> list[str]:
    messages: list[str] = []
    expected_flows = (
        "aie.flow(%shim1, DMA : 0, %mem1, DMA : 0)",
        "aie.flow(%shim1, DMA : 1, %mem1, DMA : 1)",
        "aie.flow(%mem1, DMA : 0, %mem0, DMA : 0)",
        "aie.flow(%mem1, DMA : 1, %mem2, DMA : 0)",
        "aie.flow(%mem1, DMA : 2, %mem0, DMA : 1)",
        "aie.flow(%mem1, DMA : 3, %mem2, DMA : 1)",
        "aie.flow(%mem0, DMA : 0, %worker_g0, DMA : 0)",
        "aie.flow(%mem0, DMA : 1, %worker_g1, DMA : 0)",
        "aie.flow(%mem0, DMA : 2, %worker_g0, DMA : 1)",
        "aie.flow(%mem0, DMA : 3, %worker_g1, DMA : 1)",
        "aie.flow(%mem2, DMA : 0, %worker_g2, DMA : 0)",
        "aie.flow(%mem2, DMA : 1, %worker_g3, DMA : 0)",
        "aie.flow(%mem2, DMA : 2, %worker_g2, DMA : 1)",
        "aie.flow(%mem2, DMA : 3, %worker_g3, DMA : 1)",
    )
    for flow in expected_flows:
        if flow not in text:
            raise ValueError(f"missing flow: {flow}")

    for fragment in (
        "%source_key_ring_to_left = aie.dma_start(MM2S, 0",
        "%source_key_ring_to_right = aie.dma_start(MM2S, 1",
        "%source_value_ring_to_left = aie.dma_start(MM2S, 2",
        "%source_value_ring_to_right = aie.dma_start(MM2S, 3",
    ):
        if fragment not in text:
            raise ValueError(f"missing source fanout fragment: {fragment}")

    messages.append("one_read_source=mem1 reads k03/v03 once from shim column 1")
    messages.append("two_stage_fanout=mem1 full-tile streams -> mem0/mem2 split rings")
    messages.append("workers=g0/g1 served by mem0; g2/g3 served by mem2")
    return messages


def check_static_kv_one_pair_two_stage_attention_ingress_contract_mlir(
    text: str,
) -> list[str]:
    messages: list[str] = []
    if "One-read two-stage K/V-pair attention ingress contract" not in text:
        raise ValueError("missing one-read two-stage ingress marker")
    if text.count("aie.memtile_dma(%mem") != 3:
        raise ValueError("expected source memtile plus two split memtiles")
    if text.count("aie.mem(%worker_g") != 4:
        raise ValueError("expected four worker memory DMA regions")
    if text.count("aie.core(%worker_g") != 4:
        raise ValueError("expected four attention worker cores")
    if text.count("aiex.npu.writebd") != 2:
        raise ValueError("expected one K and one V runtime descriptor")
    if text.count("link_with") != 2:
        raise ValueError("expected split attention init/update kernel declarations")
    if text.count("func.call @llama_chunked_attention_init_f32") != 4:
        raise ValueError("expected one attention init call per worker")
    if text.count("func.call @qwen_plane_group_attention_update_split_bf16") != 8:
        raise ValueError("expected ping and pong split attention update calls per worker")

    for plane, column in (("k03", 1), ("v03", 1)):
        if f"// {plane}:" not in text:
            raise ValueError(f"missing {plane} runtime descriptor")
        if f"column = {column} : i32" not in text:
            raise ValueError(f"missing {plane} column {column} descriptor")

    messages.extend(_check_one_pair_two_stage_topology(text))
    return messages


def check_static_kv_one_pair_two_stage_attention_bounded_contract_mlir(
    text: str,
) -> list[str]:
    messages: list[str] = []
    if "One-read two-stage bounded K/V-pair attention contract" not in text:
        raise ValueError("missing one-read two-stage bounded marker")
    if text.count("aie.memtile_dma(%mem") != 3:
        raise ValueError("expected source memtile plus two split memtiles")
    if text.count("aie.mem(%worker_g") != 4:
        raise ValueError("expected four worker memory DMA regions")
    if text.count("aie.core(%worker_g") != 4:
        raise ValueError("expected four attention worker cores")
    if text.count("link_with") != 4:
        raise ValueError("expected bounded attention kernel declarations")
    if text.count("func.call @llama_chunked_attention_init_f32") != 4:
        raise ValueError("expected one attention init call per worker")
    if text.count("aiex.npu.writebd") != 0:
        raise ValueError("bounded numeric contract should use runtime input tasks, not writebd")
    if text.count("aiex.dma_configure_task_for @kv_") != 2:
        raise ValueError("expected two KV runtime input tasks")
    if text.count("aie.shim_dma_allocation @kv_") != 2:
        raise ValueError("expected two KV input shim allocations")
    if text.count("aiex.dma_configure_task_for @context_g") != 4:
        raise ValueError("expected four context output runtime tasks")
    if text.count("aie.shim_dma_allocation @context_g") != 4:
        raise ValueError("expected four context output shim allocations")
    if text.count("func.call @qwen_zero_q_current_bf16") != 4:
        raise ValueError("expected deterministic q_current init on four workers")
    if text.count("func.call @llama_chunked_attention_finalize_bf16") != 4:
        raise ValueError("expected one attention finalize call per worker")
    if text.count("func.call @qwen_plane_group_attention_update_split_bf16") < 4:
        raise ValueError("expected bounded split attention updates")

    for group in range(4):
        if f"%worker_g{group}_out_tx = aie.dma_start(MM2S, 0" not in text:
            raise ValueError(f"missing worker_g{group} context transmit DMA")
        if f"aie.flow(%worker_g{group}, DMA : 0, %shim{group}, DMA : 0)" not in text:
            raise ValueError(f"missing worker_g{group} to shim{group} context flow")
        if f"aiex.dma_await_task(%context_task_g{group})" not in text:
            raise ValueError(f"missing context_task_g{group} await")

    for name in ("kv_k03_c1", "kv_v03_c1"):
        if f"%{name}_task = aiex.dma_configure_task_for @{name}_alloc" not in text:
            raise ValueError(f"missing {name} input task")
        if f"aiex.dma_free_task(%{name}_task)" not in text:
            raise ValueError(f"missing {name} input free")

    messages.extend(_check_one_pair_two_stage_topology(text))
    messages.append("bounded_attention=one-read two-stage finite loop, finalize, context drain")
    messages.append("kv_input=2 coarse runtime MM2S tasks feed source mem1")
    messages.append("context_egress=4 worker MM2S streams drained by shim S2MM tasks")
    return messages


def check_static_kv_full_two_stage_attention_bounded_contract_mlir(
    text: str,
) -> list[str]:
    messages: list[str] = []
    if "Full two-stage bounded K/V-pair attention contract" not in text:
        raise ValueError("missing full two-stage bounded marker")
    if text.count("aie.memtile_dma(%mem") != 6:
        raise ValueError("expected two source memtiles plus four split memtiles")
    if text.count("aie.mem(%worker_g") != 8:
        raise ValueError("expected eight worker memory DMA regions")
    if text.count("aie.core(%worker_g") != 8:
        raise ValueError("expected eight attention worker cores")
    if text.count("link_with") != 4:
        raise ValueError("expected bounded attention kernel declarations")
    if text.count("func.call @qwen_zero_q_current_bf16") != 8:
        raise ValueError("expected deterministic q_current init on eight workers")
    if text.count("func.call @llama_chunked_attention_init_f32") != 8:
        raise ValueError("expected one attention init call per worker")
    if text.count("func.call @llama_chunked_attention_finalize_bf16") != 8:
        raise ValueError("expected one attention finalize call per worker")
    if text.count("func.call @qwen_plane_group_attention_update_split_bf16") < 8:
        raise ValueError("expected bounded split attention updates")

    expected_flows = (
        "aie.flow(%shim1, DMA : 0, %mem1, DMA : 0)",
        "aie.flow(%shim1, DMA : 1, %mem1, DMA : 1)",
        "aie.flow(%mem1, DMA : 0, %mem0, DMA : 0)",
        "aie.flow(%mem1, DMA : 1, %mem2, DMA : 0)",
        "aie.flow(%mem1, DMA : 2, %mem0, DMA : 1)",
        "aie.flow(%mem1, DMA : 3, %mem2, DMA : 1)",
        "aie.flow(%shim5, DMA : 0, %mem5, DMA : 0)",
        "aie.flow(%shim5, DMA : 1, %mem5, DMA : 1)",
        "aie.flow(%mem5, DMA : 0, %mem4, DMA : 0)",
        "aie.flow(%mem5, DMA : 1, %mem6, DMA : 0)",
        "aie.flow(%mem5, DMA : 2, %mem4, DMA : 1)",
        "aie.flow(%mem5, DMA : 3, %mem6, DMA : 1)",
    )
    for flow in expected_flows:
        if flow not in text:
            raise ValueError(f"missing flow: {flow}")

    for group in range(8):
        if f"%worker_g{group}_out_tx = aie.dma_start(MM2S, 0" not in text:
            raise ValueError(f"missing worker_g{group} context transmit DMA")
        if f"aie.flow(%worker_g{group}, DMA : 0, %shim{group}, DMA : 0)" not in text:
            raise ValueError(f"missing worker_g{group} to shim{group} context flow")
        if f"// context_g{group}: shim{group} S2MM0" not in text:
            raise ValueError(f"missing context_g{group} low-level output descriptor")
        if _npu_sync_s2mm(group).strip() not in text:
            raise ValueError(f"missing context_g{group} S2MM sync")

    if text.count("aiex.npu.writebd") != 12:
        raise ValueError("expected four KV plus eight context low-level writebd ops")
    if text.count("aiex.npu.address_patch") != 12:
        raise ValueError("expected one address patch per low-level writebd")
    if text.count("aiex.npu.push_queue") != 12:
        raise ValueError("expected one push_queue per low-level writebd")
    if text.count("aiex.npu.sync") != 8:
        raise ValueError("expected one S2MM sync per context output")
    if text.count("aiex.dma_configure_task_for") != 0:
        raise ValueError("full two-stage low-level contract should not use DMA tasks")
    if text.count("aie.shim_dma_allocation") != 0:
        raise ValueError("full two-stage low-level contract should not use shim allocations")

    for name in ("kv_k03_c1", "kv_v03_c1", "kv_k47_c5", "kv_v47_c5"):
        if f"// {name}: arg0+" not in text:
            raise ValueError(f"missing {name} low-level input descriptor")

    for ring_name in ("p0_source_key_ring", "p0_source_value_ring", "p1_source_key_ring", "p1_source_value_ring"):
        for suffix in ("to_left", "to_right"):
            if f"%{ring_name}_{suffix} = aie.dma_start(MM2S" not in text:
                raise ValueError(f"missing {ring_name}_{suffix} fanout DMA")

    messages.append("full_two_stage=mem1 reads k03/v03 once; mem5 reads k47/v47 once")
    messages.append("split_fanout=source memtiles forward full tiles to four split memtiles")
    messages.append("bounded_attention=8 workers finite loop, finalize, context drain")
    messages.append("kv_input=4 low-level writebd/address_patch/push_queue descriptors")
    messages.append("context_egress=8 low-level S2MM descriptors with explicit sync")
    return messages
