#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar

import aie.utils as aie_utils
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
from iron.common.context import AIEContext

from models.fast_qwen3.q4nx_layout import (
    Q4NX_CHUNK_BYTES,
    Q4NX_IN_CHUNK,
    Q4NX_PATCH_OUT_ROWS,
)


@dataclass
class Q4NXFusedLinearProjection(MLIROperator):
    """Raw bf16 vector times packed Q4NX linear output patches.

    This is the attention-output and residual-path projection building block.
    Qwen3-0.6B o_proj uses K=2048.  The full 512-wide block maps to
    c2..c5/r2..r5: every 64-row output patch is split into two 32-row Q4NX
    chunks, each worker consumes full K for its 32 output rows, and no
    cross-tile K reduction is needed.
    """

    in_features: int = 2048
    output_patches: int = 8
    context: AIEContext | None = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "in_features": "k",
        "output_patches": "op",
    }

    def __post_init__(self) -> None:
        if self.in_features % Q4NX_IN_CHUNK != 0:
            raise ValueError(f"in_features must be divisible by {Q4NX_IN_CHUNK}")
        if not 1 <= self.output_patches <= 8:
            raise ValueError("output_patches must be in [1, 8]")
        if self.k_chunks > 8:
            raise ValueError(
                "Q4NXFusedLinearProjection currently supports at most 2048 K"
            )
        MLIROperator.__init__(self, context=self.context)

    @property
    def k_chunks(self) -> int:
        return self.in_features // Q4NX_IN_CHUNK

    @property
    def patch_bytes_per_k_chunk(self) -> int:
        return Q4NX_CHUNK_BYTES

    @property
    def weight_stream_bytes(self) -> int:
        return (
            self.output_patches
            * 2
            * self.k_chunks
            * self.patch_bytes_per_k_chunk
        )

    @property
    def output_elements(self) -> int:
        return self.output_patches * Q4NX_PATCH_OUT_ROWS

    def _kernel_object_name(self) -> str:
        return (
            "q4nx_fused_linear_projection_"
            f"{self.in_features}k_op{self.output_patches}.o"
        )

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "q4nx_fused_linear_projection",
                (
                    aie_utils.get_current_device(),
                    self.in_features,
                    self.output_patches,
                ),
                {
                    "verbose": mlir_verbose,
                    "kernel_object": self._kernel_object_name(),
                },
            ),
        )

    def get_kernel_artifacts(self):
        kernel_source = (
            self.operator_dir.parent / "q4nx_fused_qkv_projection" / "kernel.cc"
        )
        return [
            KernelObjectArtifact(
                self._kernel_object_name(),
                dependencies=[SourceArtifact(kernel_source)],
                extra_flags=[
                    f"-DQ4NX_FULL_IN_FEATURES={self.in_features}",
                    f"-DQ4NX_OUT_ROWS={Q4NX_PATCH_OUT_ROWS}",
                    f"-DQ4NX_CHUNK_BYTES={Q4NX_CHUNK_BYTES}",
                ],
            )
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.in_features,), dtype=bfloat16),
            AIERuntimeArgSpec("in", (self.weight_stream_bytes,), dtype=np.uint8),
            AIERuntimeArgSpec(
                "out",
                (self.output_patches, Q4NX_PATCH_OUT_ROWS),
                dtype=bfloat16,
            ),
        ]
