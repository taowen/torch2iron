// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define NOCPP

#include <stdint.h>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

#ifndef VEC_SIZE
#define VEC_SIZE 32
#endif

#ifndef GROUP_SIZE
#define GROUP_SIZE 128
#endif

#ifndef DIM_K
#define DIM_K 1024
#endif

#define NUM_GROUPS (DIM_K / GROUP_SIZE)

extern "C" {

void w4a16_matvec_bf16(uint32_t m,
                       uint32_t row_offset,
                       const uint4 *__restrict qparam,
                       const bfloat16 *__restrict x,
                       bfloat16 *__restrict y)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    constexpr int vec_len = VEC_SIZE;
    constexpr int blocks_per_group = GROUP_SIZE / vec_len;
    constexpr int qweight_row_bytes = DIM_K / 2;
    constexpr int qparam_row_bytes = qweight_row_bytes + NUM_GROUPS * sizeof(bfloat16);
    const bfloat16 bias = static_cast<bfloat16>(8.0f);
    aie::vector<bfloat16, vec_len> bias_vec = aie::broadcast<bfloat16, vec_len>(bias);

    // Process two output rows together so each activation vector load is reused.
    for (uint32_t row = 0; row < m; row += 2) {
        const uint4 *__restrict q_row0 = qparam + row * qparam_row_bytes;
        const bfloat16 *__restrict scale_row0 =
            reinterpret_cast<const bfloat16 *>(q_row0 + qweight_row_bytes);

        if (row + 1 < m) {
            const uint4 *__restrict q_row1 = q_row0 + qparam_row_bytes;
            const bfloat16 *__restrict scale_row1 =
                reinterpret_cast<const bfloat16 *>(q_row1 + qweight_row_bytes);
            aie::accum<accfloat, vec_len> acc0 = aie::zeros<accfloat, vec_len>();
            aie::accum<accfloat, vec_len> acc1 = aie::zeros<accfloat, vec_len>();

            for (int group = 0; group < NUM_GROUPS; group++) {
                aie::vector<bfloat16, vec_len> scale_vec0 =
                    aie::broadcast<bfloat16, vec_len>(scale_row0[group]);
                aie::vector<bfloat16, vec_len> scale_vec1 =
                    aie::broadcast<bfloat16, vec_len>(scale_row1[group]);

                for (int block = 0; block < blocks_per_group; block++)
                    chess_prepare_for_pipelining chess_loop_range(blocks_per_group, )
                {
                    int k = group * GROUP_SIZE + block * vec_len;
                    aie::vector<bfloat16, vec_len> x_vec =
                        aie::load_v<vec_len>(x + k);

                    aie::vector<uint4, vec_len> q4_0 =
                        aie::load_v<vec_len>(q_row0 + k / 2);
                    aie::vector<uint8, vec_len> q8_0 = aie::unpack(q4_0);
                    aie::vector<uint16, vec_len> q16_0 = aie::unpack(q8_0);
                    aie::vector<bfloat16, vec_len> q_bf16_0 =
                        aie::to_float<bfloat16>(q16_0, 0);
                    q_bf16_0 = aie::sub(q_bf16_0, bias_vec);
                    aie::vector<bfloat16, vec_len> q_scaled0 =
                        aie::mul(q_bf16_0, scale_vec0).template to_vector<bfloat16>();
                    acc0 = aie::mac(acc0, x_vec, q_scaled0);

                    aie::vector<uint4, vec_len> q4_1 =
                        aie::load_v<vec_len>(q_row1 + k / 2);
                    aie::vector<uint8, vec_len> q8_1 = aie::unpack(q4_1);
                    aie::vector<uint16, vec_len> q16_1 = aie::unpack(q8_1);
                    aie::vector<bfloat16, vec_len> q_bf16_1 =
                        aie::to_float<bfloat16>(q16_1, 0);
                    q_bf16_1 = aie::sub(q_bf16_1, bias_vec);
                    aie::vector<bfloat16, vec_len> q_scaled1 =
                        aie::mul(q_bf16_1, scale_vec1).template to_vector<bfloat16>();
                    acc1 = aie::mac(acc1, x_vec, q_scaled1);
                }
            }

            float sum0 = aie::reduce_add(acc0.template to_vector<float>());
            float sum1 = aie::reduce_add(acc1.template to_vector<float>());
            y[row_offset + row] = static_cast<bfloat16>(sum0);
            y[row_offset + row + 1] = static_cast<bfloat16>(sum1);
        } else {
            aie::accum<accfloat, vec_len> acc = aie::zeros<accfloat, vec_len>();

            for (int group = 0; group < NUM_GROUPS; group++) {
                bfloat16 scale = scale_row0[group];
                aie::vector<bfloat16, vec_len> scale_vec =
                    aie::broadcast<bfloat16, vec_len>(scale);

                for (int block = 0; block < blocks_per_group; block++)
                    chess_prepare_for_pipelining chess_loop_range(blocks_per_group, )
                {
                    int k = group * GROUP_SIZE + block * vec_len;
                    aie::vector<uint4, vec_len> q4 =
                        aie::load_v<vec_len>(q_row0 + k / 2);
                    aie::vector<uint8, vec_len> q8 = aie::unpack(q4);
                    aie::vector<uint16, vec_len> q16 = aie::unpack(q8);
                    aie::vector<bfloat16, vec_len> q_bf16 =
                        aie::to_float<bfloat16>(q16, 0);
                    q_bf16 = aie::sub(q_bf16, bias_vec);
                    aie::vector<bfloat16, vec_len> q_scaled =
                        aie::mul(q_bf16, scale_vec).template to_vector<bfloat16>();

                    aie::vector<bfloat16, vec_len> x_vec =
                        aie::load_v<vec_len>(x + k);
                    acc = aie::mac(acc, x_vec, q_scaled);
                }
            }

            float sum = aie::reduce_add(acc.template to_vector<float>());
            y[row_offset + row] = static_cast<bfloat16>(sum);
        }
    }

    event1();
}

} // extern "C"
