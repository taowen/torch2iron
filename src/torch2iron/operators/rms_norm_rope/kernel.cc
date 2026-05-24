// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef RMS_NORM_EPSILON
#define RMS_NORM_EPSILON 1e-5f
#endif

static inline void weighted_rms_norm_row(
    const bfloat16 *__restrict input,
    const bfloat16 *__restrict weight,
    bfloat16 *__restrict output,
    int32_t dims)
{
    constexpr int vec_len = 16;
    aie::vector<float, vec_len> sum_vec = aie::zeros<float, vec_len>();

    for (int32_t dim = 0; dim < dims; dim += vec_len) {
        aie::vector<bfloat16, vec_len> x = aie::load_v<vec_len>(input + dim);
        sum_vec = aie::add(sum_vec, aie::mul_square(x).template to_vector<float>());
    }

    float sum_sq = aie::reduce_add(sum_vec);
    float inv_rms = aie::invsqrt(sum_sq / static_cast<float>(dims) + RMS_NORM_EPSILON);
    aie::vector<bfloat16, vec_len> inv_rms_vec =
        aie::broadcast<bfloat16, vec_len>(static_cast<bfloat16>(inv_rms));

    for (int32_t dim = 0; dim < dims; dim += vec_len) {
        aie::vector<bfloat16, vec_len> x = aie::load_v<vec_len>(input + dim);
        aie::vector<bfloat16, vec_len> w = aie::load_v<vec_len>(weight + dim);
        aie::vector<bfloat16, vec_len> norm =
            aie::mul(x, inv_rms_vec).template to_vector<bfloat16>();
        aie::store_v(output + dim, aie::mul(norm, w).template to_vector<bfloat16>());
    }
}

static inline void rope_two_halves(
    const bfloat16 *__restrict input,
    const bfloat16 *__restrict angles,
    bfloat16 *__restrict output,
    int32_t dims)
{
    constexpr int vec_len = 16;
    int32_t half_dims = dims / 2;

    for (int32_t dim = 0, angle_dim = 0; dim < half_dims; dim += vec_len, angle_dim += 2 * vec_len) {
        aie::vector<bfloat16, vec_len> x1 = aie::load_v<vec_len>(input + dim);
        aie::vector<bfloat16, vec_len> x2 = aie::load_v<vec_len>(input + dim + half_dims);
        aie::vector<bfloat16, 2 * vec_len> angle =
            aie::load_v<2 * vec_len>(angles + angle_dim);

        aie::vector<bfloat16, vec_len> cos_val = aie::filter_even(angle, 1);
        aie::vector<bfloat16, vec_len> sin_val = aie::filter_odd(angle, 1);

        aie::vector<bfloat16, vec_len> out_first =
            aie::sub(aie::mul(x1, cos_val), aie::mul(x2, sin_val))
                .template to_vector<bfloat16>();
        aie::vector<bfloat16, vec_len> out_second =
            aie::add(aie::mul(x2, cos_val), aie::mul(x1, sin_val))
                .template to_vector<bfloat16>();

        aie::store_v(output + dim, out_first);
        aie::store_v(output + dim + half_dims, out_second);
    }
}

extern "C" {

void weighted_rms_norm_row_bf16(const bfloat16 *__restrict input,
                                const bfloat16 *__restrict weight,
                                bfloat16 *__restrict output,
                                int32_t dims)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    weighted_rms_norm_row(input, weight, output, dims);
    event1();
}

void rope_row_bf16(const bfloat16 *__restrict input,
                  const bfloat16 *__restrict angles,
                  bfloat16 *__restrict output,
                  int32_t dims)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    rope_two_halves(input, angles, output, dims);
    event1();
}

} // extern "C"
