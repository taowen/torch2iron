#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

from iron.common.context import AIEContext
from torch2iron.fusion import FusedFullELFCallable
from torch2iron.operators import (
    ElementwiseAdd,
    ElementwiseMul,
    GEMM,
    RMSNorm,
    RoPE,
    SiLU,
)

from models.exported_llama3.generated.decode_fused import build_decode_fused_op
from models.exported_llama3.llama_packed_weights import (
    load_llama_packed_segment,
    validate_llama_packed_weight_artifact,
)
from models.exported_llama3.llama_weight_layout import iter_llama_decode_weight_specs


class AIEPrefillOperations:
    pass


class AIEDecodeOperations:
    pass


class AIELlamaOperators:
    def __init__(self, config, prompt_len):
        build_suffix = f"seq{prompt_len}"
        self.context = AIEContext(build_dir=Path("build") / build_suffix)
        self.context.build_dir.mkdir(parents=True, exist_ok=True)

        self.prefill = AIEPrefillOperations()
        self.decode = AIEDecodeOperations()

        self._build_prefill_ops(config, prompt_len)
        self._build_decode_ops(config, prompt_len, build_suffix)

    def _build_prefill_ops(self, config, prompt_len):
        self.prefill.rms_norm = (
            RMSNorm(
                size=prompt_len * config.emb_dim,
                num_aie_columns=8,
                num_channels=1,
                tile_size=config.emb_dim,
                weighted=True,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.residual_add = (
            ElementwiseAdd(
                size=prompt_len * config.emb_dim,
                tile_size=config.emb_dim,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        min_N = 64 * 8 * 4  # tile_n * num_aie_columns * partition_N
        config.padded_vocab_size = (config.vocab_size + min_N - 1) // min_N * min_N
        config.vocab_partitions = 4
        self.prefill.gemv_out_head_compilable = GEMM(
            M=prompt_len,
            K=config.emb_dim,
            N=config.padded_vocab_size // config.vocab_partitions,
            num_aie_columns=8,
            tile_m=64,
            tile_k=64,
            tile_n=64,
            b_col_maj=True,
            separate_c_tiles=True,
            context=self.context,
        ).compile()
        self.prefill.out_head = self.prefill.gemv_out_head_compilable.get_callable()

        self.prefill.ffn_up_gate = (
            GEMM(
                M=prompt_len,
                K=config.emb_dim,
                N=config.hidden_dim,
                num_aie_columns=8,
                tile_m=64,
                tile_k=64,
                tile_n=64,
                b_col_maj=False,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.ffn_down = (
            GEMM(
                M=prompt_len,
                K=config.hidden_dim,
                N=config.emb_dim,
                num_aie_columns=8,
                tile_m=64,
                tile_k=64,
                tile_n=64,
                b_col_maj=False,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.ffn_silu = (
            SiLU(
                size=prompt_len * config.hidden_dim,
                tile_size=config.hidden_dim,
                num_aie_columns=8,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.eltwise_mul_ffn = (
            ElementwiseMul(
                size=prompt_len * config.hidden_dim,
                tile_size=config.hidden_dim,
                num_aie_columns=8,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.attn_scale = (
            ElementwiseMul(
                size=config.n_heads * prompt_len * prompt_len,
                tile_size=prompt_len,
                num_aie_columns=8,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.rope_queries = (
            RoPE(
                rows=prompt_len * config.n_heads,
                cols=config.head_dim,
                angle_rows=prompt_len,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.rope_keys = (
            RoPE(
                rows=prompt_len * config.n_kv_groups,
                cols=config.head_dim,
                angle_rows=prompt_len,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.attn_query = (
            GEMM(
                M=prompt_len,
                K=config.emb_dim,
                N=config.n_heads * config.head_dim,
                num_aie_columns=8,
                tile_m=64,
                tile_k=64,
                tile_n=64,
                b_col_maj=False,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.attn_key = (
            GEMM(
                M=prompt_len,
                K=config.emb_dim,
                N=config.n_kv_groups * config.head_dim,
                num_aie_columns=8,
                tile_m=64,
                tile_k=64,
                tile_n=64,
                b_col_maj=False,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.attn_value = (
            GEMM(
                M=prompt_len,
                K=config.emb_dim,
                N=config.n_kv_groups * config.head_dim,
                num_aie_columns=8,
                tile_m=64,
                tile_k=64,
                tile_n=64,
                b_col_maj=False,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

        self.prefill.attn_scores = (
            GEMM(
                M=prompt_len,
                K=config.head_dim,
                N=prompt_len,
                num_aie_columns=8,
                tile_m=64,
                tile_k=64,
                tile_n=64,
                b_col_maj=False,
                context=self.context,
            )
            .compile()
            .get_callable()
        )

    def _build_decode_ops(self, config, prompt_len, build_suffix):
        self.decode.fused_op, self.decode.current_cache_slot = build_decode_fused_op(
            config, prompt_len, build_suffix
        )
        self.decode.fused = FusedFullELFCallable(self.decode.fused_op)

        load_decode_weight_buffers(config, self.decode.fused)
        self.decode.fused.input_buffer.to("npu")
        self.decode.fused.weight_buffer.to("npu")
        self.decode.fused.lm_head_buffer.to("npu")
        self.decode.fused.scratch_buffer.to("npu")
        self.decode.fused.output_buffer.to("npu")


def load_decode_weight_buffers(config, fused):
    packed_dir = getattr(config, "packed_weights_dir", None)
    require_packed = getattr(config, "require_packed_weights", False)
    manifest = None
    if packed_dir is not None and Path(packed_dir).exists():
        manifest = validate_llama_packed_weight_artifact(config, packed_dir)
        logging.info("Loading decode weights from packed artifact: %s", packed_dir)
    elif require_packed:
        raise FileNotFoundError(f"packed weights required but not found at {packed_dir}")
    else:
        logging.info("Loading decode weights from safetensors tensors")

    for spec in iter_llama_decode_weight_specs(config):
        view = fused.get_buffer(spec["name"]).torch_view()
        if manifest is None:
            view[:] = config.weights[spec["source"]].flatten()
        else:
            view[:] = load_llama_packed_segment(
                packed_dir, manifest, spec["name"]
            ).flatten()
