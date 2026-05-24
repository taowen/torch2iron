// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define NOCPP

#include <stdint.h>

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

#ifndef K_GROUP
#define K_GROUP 1
#endif

#define MMUL_R 4
#define MMUL_S 8
#define MMUL_T 8

using MMUL = aie::mmul<MMUL_R, MMUL_S, MMUL_T, bfloat16, bfloat16, accauto>;

#define K_BLOCKS (TILE_K / MMUL_S)
#define N_BLOCKS (TILE_N / MMUL_T)
#define Q4_VECTOR_BYTES (MMUL_S * MMUL_T / 2)
#define SCALE_VECTOR_BYTES (MMUL_S * MMUL_T * sizeof(bfloat16))
#define N_BLOCK_BYTES (K_BLOCKS * Q4_VECTOR_BYTES + SCALE_VECTOR_BYTES)

static inline aie::vector<bfloat16, MMUL::size_C>
load_or_zero(bool reset_accumulator, const bfloat16 *__restrict ptr)
{
    if (reset_accumulator) {
        return aie::zeros<bfloat16, MMUL::size_C>();
    }
    return aie::load_v<MMUL::size_C>(ptr);
}

static inline aie::vector<bfloat16, MMUL::size_B>
dequant_b_vec(const uint4 *__restrict q_ptr,
              const aie::vector<bfloat16, MMUL::size_B> &scale_vec)
{
    const bfloat16 bias = static_cast<bfloat16>(8.0f);
    aie::vector<bfloat16, MMUL::size_B> bias_vec =
        aie::broadcast<bfloat16, MMUL::size_B>(bias);

    aie::vector<uint4, MMUL::size_B> q4 =
        aie::load_v<MMUL::size_B>(q_ptr);
    aie::vector<uint8, MMUL::size_B> q8 = aie::unpack(q4);
    aie::vector<uint16, MMUL::size_B> q16 = aie::unpack(q8);
    aie::vector<bfloat16, MMUL::size_B> q_bf16 =
        aie::to_float<bfloat16>(q16, 0);
    q_bf16 = aie::sub(q_bf16, bias_vec);
    return aie::mul(q_bf16, scale_vec).template to_vector<bfloat16>();
}

static inline const uint4 *__restrict q4_block_ptr(
    const uint4 *__restrict weight,
    int n_block,
    int k_block)
{
    return weight + n_block * N_BLOCK_BYTES + k_block * Q4_VECTOR_BYTES;
}

static inline const bfloat16 *__restrict scale_block_ptr(
    const uint4 *__restrict weight,
    int n_block)
{
    return reinterpret_cast<const bfloat16 *>(
        weight + n_block * N_BLOCK_BYTES + K_BLOCKS * Q4_VECTOR_BYTES);
}

static inline void gemm_accum_mmul(int reset_accumulator,
                                   const bfloat16 *__restrict a,
                                   const uint4 *__restrict weight,
                                   bfloat16 *__restrict c)
{
    constexpr int row_blocks = TILE_M / MMUL_R;
    constexpr int k_blocks = K_BLOCKS;
    constexpr int n_blocks = N_BLOCKS;

    static_assert(TILE_M == MMUL_R || TILE_M % (2 * MMUL_R) == 0);
    static_assert(TILE_K % MMUL_S == 0);
    static_assert(TILE_N % (2 * MMUL_T) == 0);

    for (int m_block = 0; m_block < row_blocks; m_block++)
        chess_prepare_for_pipelining chess_loop_range(1, )
    {
        bfloat16 *__restrict c0 = c + (m_block * n_blocks) * MMUL::size_C;

        for (int n_block = 0; n_block < n_blocks; n_block += 2)
            chess_flatten_loop
        {
            const bfloat16 *__restrict a0 =
                a + (m_block * k_blocks) * MMUL::size_A;
            const uint4 *__restrict b0 = q4_block_ptr(weight, n_block, 0);
            const uint4 *__restrict b1 = q4_block_ptr(weight, n_block + 1, 0);
            aie::vector<bfloat16, MMUL::size_B> scale0 =
                aie::load_v<MMUL::size_B>(scale_block_ptr(weight, n_block));
            aie::vector<bfloat16, MMUL::size_B> scale1 =
                aie::load_v<MMUL::size_B>(scale_block_ptr(weight, n_block + 1));

            MMUL c00(load_or_zero(reset_accumulator, c0));
            MMUL c01(load_or_zero(reset_accumulator, c0 + MMUL::size_C));

            for (int k_block = 0; k_block < k_blocks; k_block++)
                chess_flatten_loop
            {
                aie::vector<bfloat16, MMUL::size_A> a_vec0 =
                    aie::load_v<MMUL::size_A>(a0);
                aie::vector<bfloat16, MMUL::size_B> b_vec0 =
                    dequant_b_vec(b0, scale0);
                aie::vector<bfloat16, MMUL::size_B> b_vec1 =
                    dequant_b_vec(b1, scale1);

                c00.mac(a_vec0, b_vec0);
                c01.mac(a_vec0, b_vec1);

                a0 += MMUL::size_A;
                b0 += Q4_VECTOR_BYTES;
                b1 += Q4_VECTOR_BYTES;
            }

            aie::store_v(c0, c00.template to_vector<bfloat16>());
            aie::store_v(c0 + MMUL::size_C, c01.template to_vector<bfloat16>());
            c0 += 2 * MMUL::size_C;
        }
    }
}

static inline void gemm_accum_mmul_kgroup(int reset_accumulator,
                                          const bfloat16 *__restrict a_group,
                                          const uint4 *__restrict weight_group,
                                          bfloat16 *__restrict c)
{
    constexpr int row_blocks = TILE_M / MMUL_R;
    constexpr int k_blocks = K_BLOCKS;
    constexpr int n_blocks = N_BLOCKS;

    static_assert(K_GROUP >= 1);
    static_assert(TILE_M == MMUL_R || TILE_M % (2 * MMUL_R) == 0);
    static_assert(TILE_K % MMUL_S == 0);
    static_assert(TILE_N % (2 * MMUL_T) == 0);

    for (int m_block = 0; m_block < row_blocks; m_block++)
        chess_prepare_for_pipelining chess_loop_range(1, )
    {
        bfloat16 *__restrict c0 = c + (m_block * n_blocks) * MMUL::size_C;

        for (int n_block = 0; n_block < n_blocks; n_block += 2)
            chess_flatten_loop
        {
            MMUL c00(load_or_zero(reset_accumulator, c0));
            MMUL c01(load_or_zero(reset_accumulator, c0 + MMUL::size_C));

            for (int k_group = 0; k_group < K_GROUP; k_group++)
                chess_flatten_loop
            {
                const bfloat16 *__restrict a0 =
                    a_group
                    + k_group * TILE_M * TILE_K
                    + (m_block * k_blocks) * MMUL::size_A;
                const uint4 *__restrict weight =
                    weight_group + k_group * N_BLOCKS * N_BLOCK_BYTES;
                const uint4 *__restrict b0 = q4_block_ptr(weight, n_block, 0);
                const uint4 *__restrict b1 = q4_block_ptr(weight, n_block + 1, 0);
                aie::vector<bfloat16, MMUL::size_B> scale0 =
                    aie::load_v<MMUL::size_B>(scale_block_ptr(weight, n_block));
                aie::vector<bfloat16, MMUL::size_B> scale1 =
                    aie::load_v<MMUL::size_B>(scale_block_ptr(weight, n_block + 1));

                for (int k_block = 0; k_block < k_blocks; k_block++)
                    chess_flatten_loop
                {
                    aie::vector<bfloat16, MMUL::size_A> a_vec0 =
                        aie::load_v<MMUL::size_A>(a0);
                    aie::vector<bfloat16, MMUL::size_B> b_vec0 =
                        dequant_b_vec(b0, scale0);
                    aie::vector<bfloat16, MMUL::size_B> b_vec1 =
                        dequant_b_vec(b1, scale1);

                    c00.mac(a_vec0, b_vec0);
                    c01.mac(a_vec0, b_vec1);

                    a0 += MMUL::size_A;
                    b0 += Q4_VECTOR_BYTES;
                    b1 += Q4_VECTOR_BYTES;
                }
            }

            aie::store_v(c0, c00.template to_vector<bfloat16>());
            aie::store_v(c0 + MMUL::size_C, c01.template to_vector<bfloat16>());
            c0 += 2 * MMUL::size_C;
        }
    }
}

extern "C" {

void w4a16_gemm_accum_w4(int reset_accumulator,
                         const bfloat16 *__restrict a,
                         const uint4 *__restrict weight,
                         bfloat16 *__restrict c)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    gemm_accum_mmul(reset_accumulator, a, weight, c);
    event1();
}

void w4a16_gemm_accum_w4_kgroup(int reset_accumulator,
                                const bfloat16 *__restrict a,
                                const uint4 *__restrict weight,
                                bfloat16 *__restrict c)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    gemm_accum_mmul_kgroup(reset_accumulator, a, weight, c);
    event1();
}

} // extern "C"
