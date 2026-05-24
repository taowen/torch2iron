// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef RMS_NORM_EPSILON
#define RMS_NORM_EPSILON 1e-5f
#endif

static inline void residual_add_copy(
    const bfloat16 *__restrict residual,
    const bfloat16 *__restrict update,
    bfloat16 *__restrict summed,
    bfloat16 *__restrict norm_input,
    int32_t dims)
{
    constexpr int vec_len = 16;

    for (int32_t dim = 0; dim < dims; dim += vec_len) {
        aie::vector<bfloat16, vec_len> residual_vec =
            aie::load_v<vec_len>(residual + dim);
        aie::vector<bfloat16, vec_len> update_vec =
            aie::load_v<vec_len>(update + dim);
        aie::vector<bfloat16, vec_len> sum_bf16 =
            aie::add(residual_vec, update_vec);
        aie::store_v(summed + dim, sum_bf16);
        aie::store_v(norm_input + dim, sum_bf16);
    }
}

static inline void weighted_rms_norm(
    const bfloat16 *__restrict input,
    const bfloat16 *__restrict weight,
    bfloat16 *__restrict norm,
    int32_t dims)
{
    constexpr int vec_len = 16;
    aie::vector<float, vec_len> sum_vec = aie::zeros<float, vec_len>();

    for (int32_t dim = 0; dim < dims; dim += vec_len) {
        aie::vector<bfloat16, vec_len> input_vec =
            aie::load_v<vec_len>(input + dim);
        sum_vec = aie::add(
            sum_vec,
            aie::mul_square(input_vec).template to_vector<float>());
    }

    float sum_sq = aie::reduce_add(sum_vec);
    float inv_rms = aie::invsqrt(
        sum_sq / static_cast<float>(dims) + RMS_NORM_EPSILON);
    aie::vector<bfloat16, vec_len> inv_rms_vec =
        aie::broadcast<bfloat16, vec_len>(static_cast<bfloat16>(inv_rms));

    for (int32_t dim = 0; dim < dims; dim += vec_len) {
        aie::vector<bfloat16, vec_len> input_vec =
            aie::load_v<vec_len>(input + dim);
        aie::vector<bfloat16, vec_len> weight_vec =
            aie::load_v<vec_len>(weight + dim);
        aie::vector<bfloat16, vec_len> normalized =
            aie::mul(input_vec, inv_rms_vec).template to_vector<bfloat16>();
        aie::store_v(
            norm + dim,
            aie::mul(normalized, weight_vec).template to_vector<bfloat16>());
    }
}

extern "C" {

void residual_add_bf16_vector(
    const bfloat16 *__restrict residual,
    const bfloat16 *__restrict update,
    bfloat16 *__restrict summed,
    bfloat16 *__restrict norm_input,
    int32_t dims)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    residual_add_copy(residual, update, summed, norm_input, dims);
    event1();
}

void weighted_rms_norm_bf16_vector(
    const bfloat16 *__restrict input,
    const bfloat16 *__restrict weight,
    bfloat16 *__restrict norm,
    int32_t dims)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    weighted_rms_norm(input, weight, norm, dims);
    event1();
}

} // extern "C"
