# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar, Dict

import numpy as np
from ml_dtypes import bfloat16

from iron.common import (
    AIERuntimeArgSpec,
    DesignGenerator,
    KernelObjectArtifact,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
)
import aie.utils as aie_utils


@dataclass
class W4A16GEMV(MLIROperator):
    """AIE W4A16 matrix-vector multiply with fused group dequantization."""

    M: int
    K: int
    num_aie_columns: int = 1
    tile_size_input: int = 2
    tile_size_output: int | None = None
    num_batches: int = 1
    shared_qparam: bool = False
    group_size: int = 128
    kernel_vector_size: int = field(default=32, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "tile_size_input": "tsi",
        "tile_size_output": "tso",
        "num_batches": "batch",
        "shared_qparam": "sharedqp",
        "group_size": "g",
    }

    def __post_init__(self):
        if self.tile_size_output is None:
            self.tile_size_output = self.tile_size_input
        if self.K % 2 != 0:
            raise ValueError("K must be even for packed int4 weights")
        if self.K % self.group_size != 0:
            raise ValueError("K must be a multiple of group_size")
        if self.kernel_vector_size != 32:
            raise ValueError("W4A16GEMV currently supports kernel_vector_size=32 only")
        if self.group_size % self.kernel_vector_size != 0:
            raise ValueError("group_size must be a multiple of kernel_vector_size")
        if self.K % self.kernel_vector_size != 0:
            raise ValueError("K must be a multiple of kernel_vector_size")
        if not (
            self.tile_size_output % self.tile_size_input == 0
            and self.tile_size_output >= self.tile_size_input
        ):
            raise ValueError("tile_size_output must be a multiple of tile_size_input")
        MLIROperator.__init__(self, context=self.context)

    @property
    def num_groups(self) -> int:
        return self.K // self.group_size

    @property
    def qparam_row_bytes(self) -> int:
        return self.K // 2 + self.num_groups * np.dtype(bfloat16).itemsize

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "my_w4a16_matvec",
                (
                    aie_utils.get_current_device(),
                    self.num_aie_columns,
                    self.M,
                    self.K,
                    self.group_size,
                    self.tile_size_input,
                    self.tile_size_output,
                    self.num_batches,
                    self.shared_qparam,
                ),
                {
                    "verbose": mlir_verbose,
                    "kernel_object": (
                        f"w4a16_gemv_qparam_{self.K}k_{self.group_size}g_"
                        f"{self.kernel_vector_size}vs.o"
                    ),
                },
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact(
                f"w4a16_gemv_qparam_{self.K}k_{self.group_size}g_{self.kernel_vector_size}vs.o",
                dependencies=[SourceArtifact(self.operator_dir / "kernel.cc")],
                extra_flags=[
                    f"-DDIM_K={self.K}",
                    f"-DGROUP_SIZE={self.group_size}",
                    f"-DVEC_SIZE={self.kernel_vector_size}",
                ],
            ),
        ]

    def get_arg_spec(self):
        batch_dim = (self.num_batches,) if self.num_batches > 1 else ()
        qparam_dim = () if self.shared_qparam else batch_dim
        return [
            AIERuntimeArgSpec(
                "in",
                qparam_dim + (self.M, self.qparam_row_bytes),
                dtype=np.uint8,
            ),
            AIERuntimeArgSpec("in", batch_dim + (self.K,), dtype=bfloat16),
            AIERuntimeArgSpec("out", batch_dim + (self.M,), dtype=bfloat16),
        ]
