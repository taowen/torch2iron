// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

void silu_mul_bf16_vector(bfloat16 *restrict left,
                          bfloat16 *restrict right,
                          bfloat16 *restrict output,
                          int32_t size)
{
    event0();

    auto left_it = aie::begin_restrict_vector<16>(left);
    auto right_it = aie::begin_restrict_vector<16>(right);
    auto output_it = aie::begin_restrict_vector<16>(output);

    aie::vector<bfloat16, 16> register_0_5 =
        aie::broadcast<bfloat16, 16>(0.5f);
    aie::vector<bfloat16, 16> register_1 =
        aie::broadcast<bfloat16, 16>(1.0f);

    for (int32_t i = 0; i < size; i += 16) {
        aie::vector<bfloat16, 16> x = *left_it++;
        aie::vector<bfloat16, 16> y = *right_it++;

        auto half_x = aie::mul(x, register_0_5);
        auto tanh_half_x = aie::tanh<bfloat16>(half_x.to_vector<float>());
        auto sigmoid_times_two = aie::add(tanh_half_x, register_1);
        auto sigmoid = aie::mul(sigmoid_times_two, register_0_5)
                           .template to_vector<bfloat16>();
        auto silu_x = aie::mul(x, sigmoid).template to_vector<bfloat16>();
        auto result = aie::mul(silu_x, y);

        *output_it++ = result.to_vector<bfloat16>();
    }

    event1();
}

} // extern "C"
