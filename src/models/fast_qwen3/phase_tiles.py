#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tile placement contract for the Fast Qwen3 fused layer phases."""

from __future__ import annotations

from aie.iron.device import Tile


PATCH_ROWS = 64
PATCH_CHUNKS = 2
CHUNK_ROWS = PATCH_ROWS // PATCH_CHUNKS
PROJECTION_BASE_COL = 2
PROJECTION_BASE_ROW = 2
EDGE_COLS = (0, 1, 6, 7)


def projection_tile(patch_idx: int, chunk_idx: int) -> Tile:
    return Tile(
        PROJECTION_BASE_COL + patch_idx // 2,
        PROJECTION_BASE_ROW + (patch_idx % 2) * PATCH_CHUNKS + chunk_idx,
    )


def projection_shim_tile(patch_idx: int) -> Tile:
    return Tile(PROJECTION_BASE_COL + patch_idx // 2, 0)


def projection_mem_tile(patch_idx: int) -> Tile:
    return Tile(PROJECTION_BASE_COL + patch_idx // 2, 1)


def residual_tile(patch_idx: int) -> Tile:
    return Tile(EDGE_COLS[patch_idx // 2], PROJECTION_BASE_ROW + patch_idx % 2)


def residual_output_shim_tile(patch_idx: int) -> Tile:
    return Tile(EDGE_COLS[patch_idx // 2], 0)
