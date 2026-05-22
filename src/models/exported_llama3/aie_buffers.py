#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class AIELlamaBuffers:
    """Compatibility shell for the old unfused prefill runtime.

    Fused prefill owns its runtime buffers through ``FusedFullELFCallable`` now:
    the chunk transformer and the final lm_head each allocate their own input,
    output, scratch, and external weight buffers.  Keeping this small object
    lets ``LlamaNpuRunner`` preserve its public shape while avoiding the old
    full-sequence activation and seq-major KV allocations.
    """

    def __init__(self, config, prompt_len, aie_ops):
        pass
