# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar

import aie.utils as aie_utils

from iron.common import (
    AIERuntimeArgSpec,
    DesignGenerator,
    KernelObjectArtifact,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
)
from iron.common.utils import get_shim_dma_limit


@dataclass
class SiLUMul(MLIROperator):
    """Fused ``silu(left) * right`` elementwise operator."""

    size: int
    tile_size: int
    num_aie_columns: int = 8
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
    }

    def __post_init__(self):
        if self.size % (self.num_aie_columns * self.tile_size) != 0:
            raise ValueError(
                f"size ({self.size}) must be a multiple of "
                f"num_aie_columns * tile_size ({self.num_aie_columns * self.tile_size})"
            )
        dev = aie_utils.get_current_device()
        shim_dma_limit = get_shim_dma_limit(dev)
        if self.num_aie_columns * 2 > shim_dma_limit:
            raise ValueError(
                f"num_aie_columns ({self.num_aie_columns}) exceeds ShimDMA limit "
                f"of {shim_dma_limit // 2} columns for this device"
            )
        super().__init__(context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir.parent / "binary_elementwise_design.py",
                "binary_elementwise_design",
                (
                    aie_utils.get_current_device(),
                    self.size,
                    self.num_aie_columns,
                    self.tile_size,
                    0,
                    "silu_mul_bf16_vector",
                    "silu_mul.o",
                ),
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact(
                "silu_mul.o",
                dependencies=[SourceArtifact(self.operator_dir / "kernel.cc")],
            )
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.size,)),
            AIERuntimeArgSpec("in", (self.size,)),
            AIERuntimeArgSpec("out", (self.size,)),
        ]
