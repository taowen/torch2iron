#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import aie.utils as aie_utils

from iron.common import DesignGenerator, PythonGeneratedMLIRArtifact

from models.fast_qwen3.operators.q4nx_fused_q_current_projection import (
    Q4NXFusedQCurrentProjectionPlaneWrite,
)


@dataclass
class Qwen3LayerFusedMLIROperator(Q4NXFusedQCurrentProjectionPlaneWrite):
    """Qwen3 layer-local fused operator.

    The initial implementation covers the first decode layer slice:
    RMSNorm, grouped Q/K/V online-Q4 projection into ``q_current``, and current
    K/V persistence into the FastFlowLM-style four-plane cache.  Later slices
    should extend this layer-local design instead of stacking more child
    operator runlist entries.
    """

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "qwen3_layer_fused",
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
                    "packet_seq_len": self.packet_seq_len,
                    "current_slot": self.current_slot,
                },
            ),
        )
