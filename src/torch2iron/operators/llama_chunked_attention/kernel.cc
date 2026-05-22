// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef LLAMA_VEC_SIZE
#define LLAMA_VEC_SIZE 32
#endif

#ifndef LLAMA_ATTN_SCALE
#define LLAMA_ATTN_SCALE 0.125f
#endif

#define LOG2E_F 1.4426950408889634f

static inline float exp_approx(float x)
{
    aie::vector<float, 8> input = aie::broadcast<float, 8>(x * LOG2E_F);
    aie::vector<bfloat16, 8> output = aie::exp2<bfloat16>(input);
    return static_cast<float>(output.get(0));
}

extern "C" {

void llama_chunked_attention_init_f32(float *__restrict state,
                                      float *__restrict acc,
                                      int32_t q_heads,
                                      int32_t head_dim)
{
    event0();

    for (int32_t q_head = 0; q_head < q_heads; q_head++) {
        state[q_head * 2] = -__builtin_inff();
        state[q_head * 2 + 1] = 0.0f;
        float *__restrict q_acc = acc + q_head * head_dim;
        for (int32_t dim = 0; dim < head_dim; dim++) {
            q_acc[dim] = 0.0f;
        }
    }

    event1();
}

void llama_chunked_attention_update_packed_bf16(
    const bfloat16 *__restrict q_full,
    const bfloat16 *__restrict packed_chunk,
    float *__restrict state,
    float *__restrict acc,
    int32_t q_heads,
    int32_t chunk_size,
    int32_t head_dim)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    constexpr int vec_len = LLAMA_VEC_SIZE;
    const bfloat16 *__restrict k = packed_chunk;
    const bfloat16 *__restrict v = packed_chunk + chunk_size * head_dim;
    const bfloat16 *__restrict mask = packed_chunk + 2 * chunk_size * head_dim;

    for (int32_t q_head = 0; q_head < q_heads; q_head++) {
        const bfloat16 *__restrict q = q_full + q_head * head_dim;
        float *__restrict q_state = state + q_head * 2;
        float *__restrict q_acc = acc + q_head * head_dim;
        float chunk_max = -__builtin_inff();
        bool has_valid = false;

        for (int32_t row = 0; row < chunk_size; row++) {
            if (static_cast<float>(mask[row]) <= 0.5f) {
                continue;
            }

            aie::accum<accfloat, vec_len> dot = aie::zeros<accfloat, vec_len>();
            const bfloat16 *__restrict k_row = k + row * head_dim;
            for (int32_t dim = 0; dim < head_dim; dim += vec_len) {
                aie::vector<bfloat16, vec_len> q_vec =
                    aie::load_v<vec_len>(q + dim);
                aie::vector<bfloat16, vec_len> k_vec =
                    aie::load_v<vec_len>(k_row + dim);
                dot = aie::mac(dot, q_vec, k_vec);
            }

            float score =
                aie::reduce_add(dot.template to_vector<float>()) * LLAMA_ATTN_SCALE;
            if (!has_valid || score > chunk_max) {
                chunk_max = score;
                has_valid = true;
            }
        }

        if (!has_valid) {
            continue;
        }

        float old_max = q_state[0];
        float old_sum = q_state[1];
        float new_max = old_max > chunk_max ? old_max : chunk_max;
        float correction = old_sum > 0.0f ? exp_approx(old_max - new_max) : 0.0f;
        float chunk_sum = 0.0f;

        for (int32_t dim = 0; dim < head_dim; dim++) {
            q_acc[dim] *= correction;
        }

        for (int32_t row = 0; row < chunk_size; row++) {
            if (static_cast<float>(mask[row]) <= 0.5f) {
                continue;
            }

            aie::accum<accfloat, vec_len> dot = aie::zeros<accfloat, vec_len>();
            const bfloat16 *__restrict k_row = k + row * head_dim;
            for (int32_t dim = 0; dim < head_dim; dim += vec_len) {
                aie::vector<bfloat16, vec_len> q_vec =
                    aie::load_v<vec_len>(q + dim);
                aie::vector<bfloat16, vec_len> k_vec =
                    aie::load_v<vec_len>(k_row + dim);
                dot = aie::mac(dot, q_vec, k_vec);
            }

            float score =
                aie::reduce_add(dot.template to_vector<float>()) * LLAMA_ATTN_SCALE;
            float weight = exp_approx(score - new_max);
            chunk_sum += weight;

            const bfloat16 *__restrict v_row = v + row * head_dim;
            for (int32_t dim = 0; dim < head_dim; dim++) {
                q_acc[dim] += weight * static_cast<float>(v_row[dim]);
            }
        }

        q_state[0] = new_max;
        q_state[1] = old_sum * correction + chunk_sum;
    }

    event1();
}

void llama_chunked_attention_finalize_bf16(
    const float *__restrict state,
    const float *__restrict acc,
    bfloat16 *__restrict out_full,
    int32_t q_heads,
    int32_t head_dim)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    for (int32_t q_head = 0; q_head < q_heads; q_head++) {
        const float *__restrict q_state = state + q_head * 2;
        const float *__restrict q_acc = acc + q_head * head_dim;
        bfloat16 *__restrict out = out_full + q_head * head_dim;
        float denom = q_state[1];
        if (denom <= 0.0f) {
            for (int32_t dim = 0; dim < head_dim; dim++) {
                out[dim] = static_cast<bfloat16>(0.0f);
            }
        } else {
            float inv = 1.0f / denom;
            for (int32_t dim = 0; dim < head_dim; dim++) {
                out[dim] = static_cast<bfloat16>(q_acc[dim] * inv);
            }
        }
    }

    event1();
}

} // extern "C"
