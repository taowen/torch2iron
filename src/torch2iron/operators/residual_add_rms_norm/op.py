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


@dataclass
class ResidualAddRMSNorm(MLIROperator):
    """Residual add followed by weighted RMSNorm over full hidden rows."""

    size: int
    num_aie_columns: int
    tile_size: int
    epsilon: float = 1e-5
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "tile_size": "ts",
    }

    def __post_init__(self):
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.size % (self.num_aie_columns * self.tile_size) != 0:
            raise ValueError(
                "size must be divisible by num_aie_columns * tile_size"
            )
        MLIROperator.__init__(self, context=self.context)

    @property
    def _epsilon_tag(self):
        return f"eps_{self.epsilon:.0e}".replace("-", "m")

    @property
    def _kernel_object(self):
        return f"residual_add_rms_norm_{self._epsilon_tag}.o"

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "residual_add_rms_norm",
                (
                    aie_utils.get_current_device(),
                    self.size,
                    self.num_aie_columns,
                    self.tile_size,
                    0,
                ),
                {"kernel_object": self._kernel_object},
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact(
                self._kernel_object,
                dependencies=[SourceArtifact(self.operator_dir / "kernel.cc")],
                extra_flags=[f"-DRMS_NORM_EPSILON={self.epsilon}f"],
            )
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.size,)),
            AIERuntimeArgSpec("in", (self.size,)),
            AIERuntimeArgSpec("in", (self.tile_size,)),
            AIERuntimeArgSpec("out", (self.size,)),
            AIERuntimeArgSpec("out", (self.size,)),
        ]
