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


@dataclass
class W4A16GEMM(MLIROperator):
    """AIE prefill matrix-matrix multiply over offline dequantized W4A16 tiles.

    The B/weight input uses the column-stream ``gemm_weight`` layout produced by
    ``models.quantized_qwen3.packed_format.make_gemm_bf16_tile``:
    ``(num_aie_columns, N // (tile_n * num_aie_columns), K // tile_k,
    tile_n, tile_k)``. Decode still uses the compact W4 qparam layout; this
    prefill-only format spends disk space to remove qparam unpack/dequant from
    the GEMM inner loop.
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
    kernel_vector_size: int = field(default=32, repr=False)
    context: object = field(default=None, repr=False)

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
        if self.kernel_vector_size != 32:
            raise ValueError("W4A16GEMM currently supports kernel_vector_size=32 only")
        if self.tile_k % self.kernel_vector_size != 0:
            raise ValueError("tile_k must be a multiple of kernel_vector_size")
        MLIROperator.__init__(self, context=self.context)

    @property
    def qparam_row_bytes(self) -> int:
        return self.tile_k

    @property
    def k_tiles(self) -> int:
        return self.K // self.tile_k

    @property
    def n_tiles(self) -> int:
        return self.N // self.tile_n

    @property
    def n_tile_groups(self) -> int:
        return self.n_tiles // self.num_aie_columns

    def _kernel_object_name(self) -> str:
        return (
            f"w4a16_gemm_bf16tile_{self.M}m_{self.K}k_{self.N}n_"
            f"{self.tile_k}tk_{self.tile_n}tn_{self.group_size}g_"
            f"{self.kernel_vector_size}vs.o"
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
                    f"-DVEC_SIZE={self.kernel_vector_size}",
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
                    self.tile_n,
                    self.tile_k,
                ),
                dtype=bfloat16,
            ),
            AIERuntimeArgSpec("out", (self.M, self.N), dtype=bfloat16),
        ]
