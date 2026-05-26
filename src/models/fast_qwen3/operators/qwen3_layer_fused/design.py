#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from models.fast_qwen3.operators.q4nx_fused_q_current_projection.design import (
    q4nx_fused_q_current_projection,
)


def qwen3_layer_fused(
    dev,
    in_features,
    num_kv_groups,
    group_index,
    q_heads_per_group,
    head_dim,
    trace_size=0,
    trace_ddr_id=7,
    func_prefix="",
    kernel_object="q4nx_fused_q_current_projection.o",
    verbose=False,
    packet_seq_len=128,
    current_slot=0,
):
    """First Qwen3 fused-layer slice: q_current projection plus KV-plane write."""

    return q4nx_fused_q_current_projection(
        dev,
        in_features,
        num_kv_groups,
        group_index,
        q_heads_per_group,
        head_dim,
        trace_size=trace_size,
        trace_ddr_id=trace_ddr_id,
        func_prefix=func_prefix,
        kernel_object=kernel_object,
        verbose=verbose,
        write_kv_plane=True,
        packet_seq_len=packet_seq_len,
        current_slot=current_slot,
    )
