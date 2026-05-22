# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from models.exported_llama3.generated.decode_layout import DECODE_WEIGHT_SPECS


def iter_llama_decode_weight_specs(config):
    generated_layers = {
        spec["layer"] for spec in DECODE_WEIGHT_SPECS if spec["layer"] is not None
    }
    if len(generated_layers) != config.n_layers:
        raise ValueError(
            f"generated decode weight layout expects {len(generated_layers)} layers, "
            f"got {config.n_layers}"
        )
    for spec in DECODE_WEIGHT_SPECS:
        yield dict(spec)
