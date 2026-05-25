// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef Q4NX_OUT_ROWS
#define Q4NX_OUT_ROWS 64
#endif

#ifndef Q4NX_FULL_IN_FEATURES
#define Q4NX_FULL_IN_FEATURES 256
#endif

#ifndef Q4NX_CHUNK_BYTES
#define Q4NX_CHUNK_BYTES 5120
#endif

#ifndef RMS_NORM_EPSILON
#define RMS_NORM_EPSILON 1e-6f
#endif

#define Q4NX_GROUP_SIZE 32
#define Q4NX_OUT_CHUNK 32
#define Q4NX_IN_CHUNK 256
#define Q4NX_GROUPS_PER_ROW (Q4NX_IN_CHUNK / Q4NX_GROUP_SIZE)
#define Q4NX_SCALE_BYTES (Q4NX_OUT_CHUNK * Q4NX_GROUPS_PER_ROW * sizeof(bfloat16))
#define Q4NX_ZERO_BYTES Q4NX_SCALE_BYTES
#define Q4NX_INT4_OFFSET (Q4NX_SCALE_BYTES + Q4NX_ZERO_BYTES)
#define Q4NX_INT4_ROW_BYTES (Q4NX_IN_CHUNK / 2)
#define Q4NX_PATCH_BYTES (2 * Q4NX_CHUNK_BYTES)

static_assert(Q4NX_OUT_ROWS == 2 * Q4NX_OUT_CHUNK);
static_assert(Q4NX_CHUNK_BYTES == 5120);
static_assert(Q4NX_FULL_IN_FEATURES % Q4NX_IN_CHUNK == 0);

static inline float rms_inv_scale(const bfloat16 *__restrict hidden)
{
    constexpr int vec_len = 16;
    aie::vector<float, vec_len> sum_vec = aie::zeros<float, vec_len>();
    for (int32_t dim = 0; dim < Q4NX_FULL_IN_FEATURES; dim += vec_len)
        chess_prepare_for_pipelining chess_loop_range(1, )
    {
        aie::vector<bfloat16, vec_len> x = aie::load_v<vec_len>(hidden + dim);
        sum_vec = aie::add(sum_vec, aie::mul_square(x).template to_vector<float>());
    }
    float sum_sq = aie::reduce_add(sum_vec);
    return aie::invsqrt(
        sum_sq / static_cast<float>(Q4NX_FULL_IN_FEATURES) + RMS_NORM_EPSILON);
}

static inline void weighted_rms_norm_full(
    const bfloat16 *__restrict hidden,
    const bfloat16 *__restrict norm_weight,
    bfloat16 *__restrict normed_hidden)
{
    constexpr int vec_len = 16;
    float inv_rms = rms_inv_scale(hidden);
    aie::vector<bfloat16, vec_len> inv_vec =
        aie::broadcast<bfloat16, vec_len>(static_cast<bfloat16>(inv_rms));
    for (int32_t dim = 0; dim < Q4NX_FULL_IN_FEATURES; dim += vec_len)
        chess_prepare_for_pipelining chess_loop_range(1, )
    {
        aie::vector<bfloat16, vec_len> x = aie::load_v<vec_len>(hidden + dim);
        aie::vector<bfloat16, vec_len> w =
            aie::load_v<vec_len>(norm_weight + dim);
        x = aie::mul(x, inv_vec).template to_vector<bfloat16>();
        x = aie::mul(x, w).template to_vector<bfloat16>();
        aie::store_v(normed_hidden + dim, x);
    }
}

static inline void q4nx_project_patch(
    int32_t reset_accumulator,
    int32_t k_offset,
    const bfloat16 *__restrict normed_hidden,
    const uint8_t *__restrict patch,
    bfloat16 *__restrict out)
{
    constexpr int vec_len = Q4NX_GROUP_SIZE;
    for (int32_t out_row = 0; out_row < Q4NX_OUT_ROWS; out_row += 2)
        chess_prepare_for_pipelining chess_loop_range(1, )
    {
        int32_t chunk_idx = out_row / Q4NX_OUT_CHUNK;
        int32_t row_in_chunk = out_row - chunk_idx * Q4NX_OUT_CHUNK;
        const uint8_t *__restrict chunk = patch + chunk_idx * Q4NX_CHUNK_BYTES;
        const bfloat16 *__restrict scales0 =
            reinterpret_cast<const bfloat16 *>(chunk)
            + row_in_chunk * Q4NX_GROUPS_PER_ROW;
        const bfloat16 *__restrict zeros0 =
            reinterpret_cast<const bfloat16 *>(chunk + Q4NX_SCALE_BYTES)
            + row_in_chunk * Q4NX_GROUPS_PER_ROW;
        const uint4 *__restrict qrow0 =
            reinterpret_cast<const uint4 *>(
                chunk + Q4NX_INT4_OFFSET + row_in_chunk * Q4NX_INT4_ROW_BYTES);

        const bfloat16 *__restrict scales1 = scales0 + Q4NX_GROUPS_PER_ROW;
        const bfloat16 *__restrict zeros1 = zeros0 + Q4NX_GROUPS_PER_ROW;
        const uint4 *__restrict qrow1 =
            reinterpret_cast<const uint4 *>(
                chunk + Q4NX_INT4_OFFSET
                + (row_in_chunk + 1) * Q4NX_INT4_ROW_BYTES);

        aie::accum<accfloat, vec_len> acc0 = aie::zeros<accfloat, vec_len>();
        aie::accum<accfloat, vec_len> acc1 = aie::zeros<accfloat, vec_len>();

        for (int32_t group = 0; group < Q4NX_GROUPS_PER_ROW; group++)
            chess_flatten_loop
        {
            int32_t k = group * Q4NX_GROUP_SIZE;
            aie::vector<bfloat16, vec_len> x_vec =
                aie::load_v<vec_len>(normed_hidden + k_offset + k);

            aie::vector<uint4, vec_len> q4_0 =
                aie::load_v<vec_len>(qrow0 + k / 2);
            aie::vector<uint8, vec_len> q8_0 = aie::unpack(q4_0);
            aie::vector<uint16, vec_len> q16_0 = aie::unpack(q8_0);
            aie::vector<bfloat16, vec_len> q_bf16_0 =
                aie::to_float<bfloat16>(q16_0, 0);
            aie::vector<bfloat16, vec_len> zero_vec0 =
                aie::broadcast<bfloat16, vec_len>(zeros0[group]);
            aie::vector<bfloat16, vec_len> scale_vec0 =
                aie::broadcast<bfloat16, vec_len>(scales0[group]);
            q_bf16_0 = aie::sub(q_bf16_0, zero_vec0);
            aie::vector<bfloat16, vec_len> q_scaled0 =
                aie::mul(q_bf16_0, scale_vec0).template to_vector<bfloat16>();
            acc0 = aie::mac(acc0, x_vec, q_scaled0);

            aie::vector<uint4, vec_len> q4_1 =
                aie::load_v<vec_len>(qrow1 + k / 2);
            aie::vector<uint8, vec_len> q8_1 = aie::unpack(q4_1);
            aie::vector<uint16, vec_len> q16_1 = aie::unpack(q8_1);
            aie::vector<bfloat16, vec_len> q_bf16_1 =
                aie::to_float<bfloat16>(q16_1, 0);
            aie::vector<bfloat16, vec_len> zero_vec1 =
                aie::broadcast<bfloat16, vec_len>(zeros1[group]);
            aie::vector<bfloat16, vec_len> scale_vec1 =
                aie::broadcast<bfloat16, vec_len>(scales1[group]);
            q_bf16_1 = aie::sub(q_bf16_1, zero_vec1);
            aie::vector<bfloat16, vec_len> q_scaled1 =
                aie::mul(q_bf16_1, scale_vec1).template to_vector<bfloat16>();
            acc1 = aie::mac(acc1, x_vec, q_scaled1);
        }

        float sum0 = aie::reduce_add(acc0.template to_vector<float>());
        float sum1 = aie::reduce_add(acc1.template to_vector<float>());
        if (!reset_accumulator) {
            sum0 += static_cast<float>(out[out_row]);
            sum1 += static_cast<float>(out[out_row + 1]);
        }
        out[out_row] = static_cast<bfloat16>(sum0);
        out[out_row + 1] = static_cast<bfloat16>(sum1);
    }
}

extern "C" {

void q4nx_up_gate_rms_norm_full(
    const bfloat16 *__restrict hidden,
    const bfloat16 *__restrict norm_weight,
    bfloat16 *__restrict normed_hidden)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    weighted_rms_norm_full(hidden, norm_weight, normed_hidden);
    event1();
}

void q4nx_fused_up_gate_projection_patch(
    int32_t reset_accumulator,
    int32_t k_offset,
    const bfloat16 *__restrict normed_hidden,
    const uint8_t *__restrict weight_chunk,
    bfloat16 *__restrict up_gate_out)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    q4nx_project_patch(
        reset_accumulator,
        k_offset,
        normed_hidden,
        weight_chunk,
        up_gate_out);
    q4nx_project_patch(
        reset_accumulator,
        k_offset,
        normed_hidden,
        weight_chunk + Q4NX_PATCH_BYTES,
        up_gate_out + Q4NX_OUT_ROWS);
    event1();
}

} // extern "C"
