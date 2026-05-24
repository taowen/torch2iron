# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar, Dict

import numpy as np
from ml_dtypes import bfloat16

import aie.utils as aie_utils
from iron.common import (
    AIERuntimeArgSpec,
    DesignGenerator,
    KernelObjectArtifact,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
)
from iron.common.context import AIEContext


@dataclass
class W4A16GEMM(MLIROperator):
    """AIE W4A16 matrix-matrix multiply over compressed int4 tiles.

    The B/weight input uses the column-stream ``gemm_w4_weight`` layout
    produced by ``models.quantized_qwen3.packed_format.make_gemm_w4_mmul_tile``:
    ``(num_aie_columns, N // (tile_n * num_aie_columns), K // tile_k,
    tile_n // 8, n_block_bytes)``. Each B sub-tile stores packed biased int4
    values followed by a 64-lane bf16 scale vector in the ``s x t`` order
    consumed by ``aie::mmul<4,8,8>``.
    """

    M: int
    K: int
    N: int
    num_aie_columns: int = 8
    num_aie_rows: int = 4
    tile_m: int = 8
    tile_k: int = 128
    tile_n: int = 64
    group_size: int = 128
    context: AIEContext | None = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "num_aie_rows": "row",
        "tile_m": "tm",
        "tile_k": "tk",
        "tile_n": "tn",
        "group_size": "g",
    }

    def __post_init__(self):
        if self.tile_k != self.group_size:
            raise ValueError("W4A16GEMM currently requires tile_k == group_size")
        if self.M % (self.tile_m * self.num_aie_rows) != 0:
            raise ValueError("M must be divisible by tile_m * num_aie_rows")
        if self.K % self.tile_k != 0:
            raise ValueError("K must be a multiple of tile_k")
        if self.N % (self.tile_n * self.num_aie_columns) != 0:
            raise ValueError("N must be divisible by tile_n * num_aie_columns")
        if self.tile_m != 4 and self.tile_m % 8 != 0:
            raise ValueError(
                "tile_m must be 4 or a multiple of 8 for aie::mmul<4,8,8>"
            )
        if self.tile_k % 8 != 0:
            raise ValueError("tile_k must be a multiple of 8 for aie::mmul<4,8,8>")
        if self.tile_n % 16 != 0:
            raise ValueError(
                "tile_n must be a multiple of 16 for 2x2 aie::mmul expansion"
            )
        MLIROperator.__init__(self, context=self.context)

    @property
    def k_tiles(self) -> int:
        return self.K // self.tile_k

    @property
    def n_tiles(self) -> int:
        return self.N // self.tile_n

    @property
    def n_tile_groups(self) -> int:
        return self.n_tiles // self.num_aie_columns

    @property
    def n_blocks(self) -> int:
        return self.tile_n // 8

    @property
    def k_blocks(self) -> int:
        return self.tile_k // 8

    @property
    def q_vector_bytes(self) -> int:
        return 8 * 8 // 2

    @property
    def scale_vector_bytes(self) -> int:
        return 8 * 8 * np.dtype(bfloat16).itemsize

    @property
    def n_block_bytes(self) -> int:
        return self.k_blocks * self.q_vector_bytes + self.scale_vector_bytes

    def _kernel_object_name(self) -> str:
        return (
            "w4a16_gemm_w4tile_mmul_"
            f"{self.M}m_{self.K}k_{self.N}n_"
            f"{self.tile_k}tk_{self.tile_n}tn_{self.group_size}g_"
            f"{self.tile_m}tm.o"
        )

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "my_w4a16_gemm",
                (
                    aie_utils.get_current_device(),
                    self.num_aie_columns,
                    self.num_aie_rows,
                    self.M,
                    self.K,
                    self.N,
                    self.tile_m,
                    self.tile_k,
                    self.tile_n,
                    self.group_size,
                ),
                {
                    "verbose": mlir_verbose,
                    "kernel_object": self._kernel_object_name(),
                },
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact(
                self._kernel_object_name(),
                dependencies=[SourceArtifact(self.operator_dir / "kernel.cc")],
                extra_flags=[
                    f"-DDIM_M={self.tile_m}",
                    f"-DDIM_K={self.K}",
                    f"-DDIM_N={self.N}",
                    f"-DTILE_M={self.tile_m}",
                    f"-DTILE_K={self.tile_k}",
                    f"-DTILE_N={self.tile_n}",
                    f"-DGROUP_SIZE={self.group_size}",
                ],
            ),
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.M, self.K), dtype=bfloat16),
            AIERuntimeArgSpec(
                "in",
                (
                    self.num_aie_columns,
                    self.n_tile_groups,
                    self.k_tiles,
                    self.n_blocks,
                    self.n_block_bytes,
                ),
                dtype=np.uint8,
            ),
            AIERuntimeArgSpec("out", (self.M, self.N), dtype=bfloat16),
        ]


@dataclass
class W4A16PairedKGroupGEMM(W4A16GEMM):
    """Paired W4A16 GEMM that accumulates multiple K tiles per worker call."""

    k_group: int = 2

    _name_aliases: ClassVar[Dict[str, str]] = {
        **W4A16GEMM._name_aliases,
        "k_group": "kg",
    }

    def __post_init__(self):
        super().__post_init__()
        if self.k_tiles % self.k_group != 0:
            raise ValueError("k_tiles must be divisible by k_group")

    def _kernel_object_name(self) -> str:
        return (
            "w4a16_paired_k_group_gemm_w4tile_mmul_"
            f"{self.M}m_{self.K}k_{self.N}n_"
            f"{self.tile_k}tk_{self.tile_n}tn_{self.group_size}g_"
            f"{self.tile_m}tm_{self.k_group}kg.o"
        )

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "my_w4a16_paired_k_group_gemm",
                (
                    aie_utils.get_current_device(),
                    self.num_aie_columns,
                    self.num_aie_rows,
                    self.M,
                    self.K,
                    self.N,
                    self.tile_m,
                    self.tile_k,
                    self.tile_n,
                    self.group_size,
                    self.k_group,
                ),
                {
                    "verbose": mlir_verbose,
                    "kernel_object": self._kernel_object_name(),
                },
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact(
                self._kernel_object_name(),
                dependencies=[SourceArtifact(self.operator_dir / "paired_kernel.cc")],
                extra_flags=[
                    f"-DDIM_M={self.tile_m}",
                    f"-DDIM_K={self.K}",
                    f"-DDIM_N={self.N}",
                    f"-DTILE_M={self.tile_m}",
                    f"-DTILE_K={self.tile_k}",
                    f"-DTILE_N={self.tile_n}",
                    f"-DGROUP_SIZE={self.group_size}",
                    f"-DK_GROUP={self.k_group}",
                ],
            ),
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.M, self.K), dtype=bfloat16),
            AIERuntimeArgSpec(
                "in",
                (
                    self.num_aie_columns,
                    self.n_tile_groups,
                    self.k_tiles,
                    2,
                    self.n_blocks,
                    self.n_block_bytes,
                ),
                dtype=np.uint8,
            ),
            AIERuntimeArgSpec("out", (self.M, self.N), dtype=bfloat16),
            AIERuntimeArgSpec("out", (self.M, self.N), dtype=bfloat16),
        ]


@dataclass
class W4A16KGroupGEMM(W4A16GEMM):
    """W4A16 GEMM that accumulates multiple K tiles inside one worker call."""

    k_group: int = 2

    _name_aliases: ClassVar[Dict[str, str]] = {
        **W4A16GEMM._name_aliases,
        "k_group": "kg",
    }

    def __post_init__(self):
        super().__post_init__()
        if self.k_tiles % self.k_group != 0:
            raise ValueError("k_tiles must be divisible by k_group")

    def _kernel_object_name(self) -> str:
        return (
            "w4a16_k_group_gemm_w4tile_mmul_"
            f"{self.M}m_{self.K}k_{self.N}n_"
            f"{self.tile_k}tk_{self.tile_n}tn_{self.group_size}g_"
            f"{self.tile_m}tm_{self.k_group}kg.o"
        )

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "my_w4a16_k_group_gemm",
                (
                    aie_utils.get_current_device(),
                    self.num_aie_columns,
                    self.num_aie_rows,
                    self.M,
                    self.K,
                    self.N,
                    self.tile_m,
                    self.tile_k,
                    self.tile_n,
                    self.group_size,
                    self.k_group,
                ),
                {
                    "verbose": mlir_verbose,
                    "kernel_object": self._kernel_object_name(),
                },
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact(
                self._kernel_object_name(),
                dependencies=[SourceArtifact(self.operator_dir / "kernel.cc")],
                extra_flags=[
                    f"-DDIM_M={self.tile_m}",
                    f"-DDIM_K={self.K}",
                    f"-DDIM_N={self.N}",
                    f"-DTILE_M={self.tile_m}",
                    f"-DTILE_K={self.tile_k}",
                    f"-DTILE_N={self.tile_n}",
                    f"-DGROUP_SIZE={self.group_size}",
                    f"-DK_GROUP={self.k_group}",
                ],
            ),
        ]


@dataclass
class W4A16NShardGEMM(W4A16GEMM):
    """W4A16 GEMM that uses core rows as N-dimension shards."""

    def __post_init__(self):
        if self.tile_k != self.group_size:
            raise ValueError("W4A16NShardGEMM currently requires tile_k == group_size")
        if self.M % self.tile_m != 0:
            raise ValueError("M must be divisible by tile_m")
        if self.K % self.tile_k != 0:
            raise ValueError("K must be a multiple of tile_k")
        if self.N % (self.tile_n * self.num_aie_columns * self.num_aie_rows) != 0:
            raise ValueError(
                "N must be divisible by tile_n * num_aie_columns * num_aie_rows"
            )
        if self.tile_m != 4 and self.tile_m % 8 != 0:
            raise ValueError(
                "tile_m must be 4 or a multiple of 8 for aie::mmul<4,8,8>"
            )
        if self.tile_k % 8 != 0:
            raise ValueError("tile_k must be a multiple of 8 for aie::mmul<4,8,8>")
        if self.tile_n % 16 != 0:
            raise ValueError(
                "tile_n must be a multiple of 16 for 2x2 aie::mmul expansion"
            )
        MLIROperator.__init__(self, context=self.context)

    @property
    def n_tile_groups(self) -> int:
        return self.n_tiles // self.num_aie_columns

    def _kernel_object_name(self) -> str:
        return (
            "w4a16_n_shard_gemm_w4tile_mmul_"
            f"{self.M}m_{self.K}k_{self.N}n_"
            f"{self.tile_k}tk_{self.tile_n}tn_{self.group_size}g_"
            f"{self.tile_m}tm_{self.num_aie_rows}r.o"
        )

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "my_w4a16_n_shard_gemm",
                (
                    aie_utils.get_current_device(),
                    self.num_aie_columns,
                    self.num_aie_rows,
                    self.M,
                    self.K,
                    self.N,
                    self.tile_m,
                    self.tile_k,
                    self.tile_n,
                    self.group_size,
                ),
                {
                    "verbose": mlir_verbose,
                    "kernel_object": self._kernel_object_name(),
                },
            ),
        )
