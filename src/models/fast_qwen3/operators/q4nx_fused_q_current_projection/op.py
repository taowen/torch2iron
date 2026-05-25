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
class Q4NXFusedQCurrentProjection(MLIROperator):
    """Fused RMSNorm + group-local Q/K/V projection into ``q_current``.

    One operator instance builds one or more KV groups of ``q_current``:
    ``[Q heads, current K, current V]``.  For each Qwen3-0.6B KV group this is
    exactly eight 64-row output patches: four Q patches, two K patches, and two
    V patches.
    """

    in_features: int = Q4NX_IN_CHUNK
    num_kv_groups: int = 1
    group_index: int = 0
    q_heads_per_group: int = 2
    head_dim: int = 128
    rms_norm_epsilon: float = 1e-6
    context: AIEContext | None = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "in_features": "k",
        "num_kv_groups": "kvg",
        "group_index": "gidx",
        "q_heads_per_group": "qhpg",
        "head_dim": "hd",
        "rms_norm_epsilon": "eps",
    }

    def __post_init__(self) -> None:
        if self.in_features % Q4NX_IN_CHUNK != 0:
            raise ValueError(f"in_features must be divisible by {Q4NX_IN_CHUNK}")
        if self.num_kv_groups <= 0:
            raise ValueError("num_kv_groups must be positive")
        if self.group_index < 0:
            raise ValueError("group_index must be non-negative")
        if self.q_heads_per_group <= 0:
            raise ValueError("q_heads_per_group must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.head_dim % Q4NX_PATCH_OUT_ROWS != 0:
            raise ValueError(f"head_dim must be divisible by {Q4NX_PATCH_OUT_ROWS}")
        if self.q_current_patches > 8:
            raise ValueError("q_current projection currently supports at most 8 patches")
        if self.rms_norm_epsilon <= 0:
            raise ValueError("rms_norm_epsilon must be positive")
        MLIROperator.__init__(self, context=self.context)

    @property
    def k_chunks(self) -> int:
        return self.in_features // Q4NX_IN_CHUNK

    @property
    def patch_bytes_per_k_chunk(self) -> int:
        return 2 * Q4NX_CHUNK_BYTES

    @property
    def head_chunks(self) -> int:
        return self.head_dim // Q4NX_PATCH_OUT_ROWS

    @property
    def q_patches_per_group(self) -> int:
        return self.q_heads_per_group * self.head_chunks

    @property
    def q_current_patches(self) -> int:
        return self.q_patches_per_group + 2 * self.head_chunks

    @property
    def q_elements_per_group(self) -> int:
        return self.q_heads_per_group * self.head_dim

    @property
    def q_current_elements(self) -> int:
        return self.num_kv_groups * self.q_current_elements_per_group

    @property
    def q_current_elements_per_group(self) -> int:
        return self.q_elements_per_group + 2 * self.head_dim

    @property
    def weight_stream_bytes(self) -> int:
        return (
            self.num_kv_groups
            * self.q_current_patches
            * self.k_chunks
            * self.patch_bytes_per_k_chunk
        )

    def _kernel_object_name(self) -> str:
        eps_tag = f"{self.rms_norm_epsilon:.0e}".replace("-", "m")
        return (
            "q4nx_fused_rms_q_current_projection_"
            f"{self.in_features}k_kvg{self.num_kv_groups}_g{self.group_index}_hd{self.head_dim}_"
            f"qh{self.q_heads_per_group}_eps{eps_tag}.o"
        )

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "q4nx_fused_q_current_projection",
                (
                    aie_utils.get_current_device(),
                    self.in_features,
                    self.num_kv_groups,
                    self.group_index,
                    self.q_heads_per_group,
                    self.head_dim,
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
                    f"-DRMS_NORM_EPSILON={self.rms_norm_epsilon}f",
                ],
            )
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.in_features,), dtype=bfloat16),
            AIERuntimeArgSpec("in", (self.in_features,), dtype=bfloat16),
            AIERuntimeArgSpec("in", (self.weight_stream_bytes,), dtype=np.uint8),
            AIERuntimeArgSpec("out", (self.q_current_elements,), dtype=bfloat16),
        ]
