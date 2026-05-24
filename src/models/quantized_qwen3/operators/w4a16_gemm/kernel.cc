// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define NOCPP

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

#ifndef TILE_K
#define TILE_K 128
#endif

#ifndef TILE_N
#define TILE_N 64
#endif

#ifndef TILE_M
#define TILE_M 8
#endif

#define MMUL_R 4
#define MMUL_S 8
#define MMUL_T 8

using MMUL = aie::mmul<MMUL_R, MMUL_S, MMUL_T, bfloat16, bfloat16, accauto>;

static inline aie::vector<bfloat16, MMUL::size_C>
load_or_zero(bool reset_accumulator, const bfloat16 *__restrict ptr)
{
    if (reset_accumulator) {
        return aie::zeros<bfloat16, MMUL::size_C>();
    }
    return aie::load_v<MMUL::size_C>(ptr);
}

static inline void gemm_accum_mmul(int reset_accumulator,
                                   const bfloat16 *__restrict a,
                                   const bfloat16 *__restrict weight,
                                   bfloat16 *__restrict c)
{
    constexpr int row_blocks = TILE_M / MMUL_R;
    constexpr int k_blocks = TILE_K / MMUL_S;
    constexpr int n_blocks = TILE_N / MMUL_T;

    static_assert(TILE_M == MMUL_R || TILE_M % (2 * MMUL_R) == 0);
    static_assert(TILE_K % MMUL_S == 0);
    static_assert(TILE_N % (2 * MMUL_T) == 0);

    if constexpr (row_blocks == 1) {
        bfloat16 *__restrict c0 = c;

        for (int n_block = 0; n_block < n_blocks; n_block += 2)
            chess_flatten_loop
        {
            const bfloat16 *__restrict a0 = a;
            const bfloat16 *__restrict b0 = weight + n_block * MMUL::size_B;
            const bfloat16 *__restrict b1 = b0 + MMUL::size_B;

            MMUL c00(load_or_zero(reset_accumulator, c0));
            MMUL c01(load_or_zero(reset_accumulator, c0 + MMUL::size_C));

            for (int k_block = 0; k_block < k_blocks; k_block++)
                chess_flatten_loop
            {
                aie::vector<bfloat16, MMUL::size_A> a_vec0 =
                    aie::load_v<MMUL::size_A>(a0);
                aie::vector<bfloat16, MMUL::size_B> b_vec0 =
                    aie::load_v<MMUL::size_B>(b0);
                aie::vector<bfloat16, MMUL::size_B> b_vec1 =
                    aie::load_v<MMUL::size_B>(b1);

                c00.mac(a_vec0, b_vec0);
                c01.mac(a_vec0, b_vec1);

                a0 += MMUL::size_A;
                b0 += n_blocks * MMUL::size_B;
                b1 += n_blocks * MMUL::size_B;
            }

            aie::store_v(c0, c00.template to_vector<bfloat16>());
            aie::store_v(c0 + MMUL::size_C, c01.template to_vector<bfloat16>());
            c0 += 2 * MMUL::size_C;
        }
        return;
    }

    for (int m_block = 0; m_block < row_blocks; m_block += 2)
        chess_prepare_for_pipelining chess_loop_range(1, )
    {
        bfloat16 *__restrict c0 = c + (m_block * n_blocks) * MMUL::size_C;
        bfloat16 *__restrict c1 = c + ((m_block + 1) * n_blocks) * MMUL::size_C;

        for (int n_block = 0; n_block < n_blocks; n_block += 2)
            chess_flatten_loop
        {
            const bfloat16 *__restrict a0 =
                a + (m_block * k_blocks) * MMUL::size_A;
            const bfloat16 *__restrict a1 =
                a + ((m_block + 1) * k_blocks) * MMUL::size_A;
            const bfloat16 *__restrict b0 = weight + n_block * MMUL::size_B;
            const bfloat16 *__restrict b1 = b0 + MMUL::size_B;

            MMUL c00(load_or_zero(reset_accumulator, c0));
            MMUL c01(load_or_zero(reset_accumulator, c0 + MMUL::size_C));
            MMUL c10(load_or_zero(reset_accumulator, c1));
            MMUL c11(load_or_zero(reset_accumulator, c1 + MMUL::size_C));

            for (int k_block = 0; k_block < k_blocks; k_block++)
                chess_flatten_loop
            {
                aie::vector<bfloat16, MMUL::size_A> a_vec0 =
                    aie::load_v<MMUL::size_A>(a0);
                aie::vector<bfloat16, MMUL::size_A> a_vec1 =
                    aie::load_v<MMUL::size_A>(a1);
                aie::vector<bfloat16, MMUL::size_B> b_vec0 =
                    aie::load_v<MMUL::size_B>(b0);
                aie::vector<bfloat16, MMUL::size_B> b_vec1 =
                    aie::load_v<MMUL::size_B>(b1);

                c00.mac(a_vec0, b_vec0);
                c01.mac(a_vec0, b_vec1);
                c10.mac(a_vec1, b_vec0);
                c11.mac(a_vec1, b_vec1);

                a0 += MMUL::size_A;
                a1 += MMUL::size_A;
                b0 += n_blocks * MMUL::size_B;
                b1 += n_blocks * MMUL::size_B;
            }

            aie::store_v(c0, c00.template to_vector<bfloat16>());
            aie::store_v(c0 + MMUL::size_C, c01.template to_vector<bfloat16>());
            aie::store_v(c1, c10.template to_vector<bfloat16>());
            aie::store_v(c1 + MMUL::size_C, c11.template to_vector<bfloat16>());
            c0 += 2 * MMUL::size_C;
            c1 += 2 * MMUL::size_C;
        }
    }
}

extern "C" {

void w4a16_gemm_accum_bf16(int reset_accumulator,
                           const bfloat16 *__restrict a,
                           const bfloat16 *__restrict weight,
                           bfloat16 *__restrict c)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    gemm_accum_mmul(reset_accumulator, a, weight, c);
    event1();
}

} // extern "C"
