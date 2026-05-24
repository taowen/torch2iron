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

#ifndef TILE_K
#define TILE_K 128
#endif

#ifndef TILE_N
#define TILE_N 64
#endif

#ifndef DIM_M
#define DIM_M 32
#endif

extern "C" {

void w4a16_gemm_accum_bf16(int reset_accumulator,
                           const bfloat16 *__restrict a,
                           const bfloat16 *__restrict weight,
                           bfloat16 *__restrict c)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    constexpr int vec_len = VEC_SIZE;
    constexpr int blocks = TILE_K / vec_len;
    static_assert(TILE_K % vec_len == 0);
    static_assert(DIM_M % 2 == 0);

    for (int n = 0; n < TILE_N; n++) {
        const bfloat16 *__restrict w_row = weight + n * TILE_K;

        for (int m = 0; m < DIM_M; m += 2) {
            const bfloat16 *__restrict a0 = a + m * TILE_K;
            const bfloat16 *__restrict a1 = a0 + TILE_K;
            aie::accum<accfloat, vec_len> acc0 =
                aie::zeros<accfloat, vec_len>();
            aie::accum<accfloat, vec_len> acc1 =
                aie::zeros<accfloat, vec_len>();

            for (int block = 0; block < blocks; block++)
                chess_prepare_for_pipelining chess_loop_range(blocks, )
            {
                int k = block * vec_len;
                aie::vector<bfloat16, vec_len> w_vec =
                    aie::load_v<vec_len>(w_row + k);
                aie::vector<bfloat16, vec_len> a0_vec =
                    aie::load_v<vec_len>(a0 + k);
                aie::vector<bfloat16, vec_len> a1_vec =
                    aie::load_v<vec_len>(a1 + k);
                acc0 = aie::mac(acc0, a0_vec, w_vec);
                acc1 = aie::mac(acc1, a1_vec, w_vec);
            }

            float sum0 = aie::reduce_add(acc0.template to_vector<float>());
            float sum1 = aie::reduce_add(acc1.template to_vector<float>());
            if (!reset_accumulator) {
                sum0 += static_cast<float>(c[m * TILE_N + n]);
                sum1 += static_cast<float>(c[(m + 1) * TILE_N + n]);
            }
            c[m * TILE_N + n] = static_cast<bfloat16>(sum0);
            c[(m + 1) * TILE_N + n] = static_cast<bfloat16>(sum1);
        }
    }

    event1();
}

} // extern "C"
