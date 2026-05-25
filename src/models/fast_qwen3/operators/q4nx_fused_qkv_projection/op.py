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
    linear_patch_bytes,
)


@dataclass
class Q4NXFusedQKVProjection(MLIROperator):
    """Fused RMSNorm + Q/K/V projection over Q4NX output patches.

    This is the minimum layer-engine building block: a norm worker computes the
    normed hidden once, then forwards it to one worker per 64-row output patch.
    Q/K/V weight streams contain only packed Q4NX data; RMSNorm weight is not
    repeated per patch.
    """

    in_features: int = Q4NX_IN_CHUNK
    out_rows: int = Q4NX_PATCH_OUT_ROWS
    output_patches: int = 8
    rms_norm_epsilon: float = 1e-6
    context: AIEContext | None = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "in_features": "k",
        "out_rows": "orows",
        "output_patches": "opatches",
        "rms_norm_epsilon": "eps",
    }

    def __post_init__(self) -> None:
        if self.in_features % Q4NX_IN_CHUNK != 0:
            raise ValueError(f"in_features must be divisible by {Q4NX_IN_CHUNK}")
        if self.out_rows != Q4NX_PATCH_OUT_ROWS:
            raise ValueError(f"out_rows must be {Q4NX_PATCH_OUT_ROWS}")
        if not 1 <= self.output_patches <= 8:
            raise ValueError("output_patches must be in [1, 8]")
        if self.rms_norm_epsilon <= 0:
            raise ValueError("rms_norm_epsilon must be positive")
        MLIROperator.__init__(self, context=self.context)

    @property
    def patch_bytes(self) -> int:
        return linear_patch_bytes(self.in_features)

    @property
    def k_chunk_patch_bytes(self) -> int:
        return 2 * Q4NX_CHUNK_BYTES

    @property
    def qkv_k_chunk_bytes(self) -> int:
        return 3 * self.k_chunk_patch_bytes

    @property
    def qkv_patch_stream_bytes(self) -> int:
        return self.output_patches * 3 * self.patch_bytes

    @property
    def k_chunks(self) -> int:
        return self.in_features // Q4NX_IN_CHUNK

    def _kernel_object_name(self) -> str:
        eps_tag = f"{self.rms_norm_epsilon:.0e}".replace("-", "m")
        return (
            "q4nx_fused_rms_qkv_projection_"
            f"{self.in_features}k_{self.out_rows}orows_"
            f"{self.output_patches}opatches_eps{eps_tag}.o"
        )

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "q4nx_fused_qkv_projection",
                (
                    aie_utils.get_current_device(),
                    self.in_features,
                    self.out_rows,
                    self.output_patches,
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
                    f"-DQ4NX_FULL_IN_FEATURES={self.in_features}",
                    f"-DQ4NX_OUT_ROWS={self.out_rows}",
                    f"-DQ4NX_CHUNK_BYTES={Q4NX_CHUNK_BYTES}",
                    f"-DRMS_NORM_EPSILON={self.rms_norm_epsilon}f",
                ],
            )
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.in_features,), dtype=bfloat16),
            AIERuntimeArgSpec("in", (self.in_features,), dtype=bfloat16),
            AIERuntimeArgSpec("in", (self.qkv_patch_stream_bytes,), dtype=np.uint8),
            AIERuntimeArgSpec(
                "out",
                (self.output_patches * 3 * self.out_rows,),
                dtype=bfloat16,
            ),
        ]
