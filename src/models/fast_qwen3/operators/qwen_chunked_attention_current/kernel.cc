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

#ifndef LLAMA_Q_HEADS_PER_GROUP
#define LLAMA_Q_HEADS_PER_GROUP 2
#endif

#define LLAMA_HEAD_BLOCKS (LLAMA_HEAD_DIM / LLAMA_VEC_SIZE)
#define LOG2E_F 1.4426950408889634f

static inline float exp_approx(float x)
{
    aie::vector<float, 8> input = aie::broadcast<float, 8>(x * LOG2E_F);
    aie::vector<bfloat16, 8> output = aie::exp2<bfloat16>(input);
    return static_cast<float>(output.get(0));
}

static inline void scale_acc_inplace(float *__restrict acc,
                                     float scale,
                                     int32_t head_dim)
{
    constexpr int vec_len = LLAMA_VEC_SIZE;
    aie::vector<float, vec_len> scale_vec =
        aie::broadcast<float, vec_len>(scale);

    for (int32_t dim = 0; dim < head_dim; dim += vec_len) {
        aie::accum<accfloat, vec_len> acc_vec;
        acc_vec.from_vector(aie::load_v<vec_len>(acc + dim), 0);
        acc_vec = aie::mul(acc_vec.to_vector<float>(), scale_vec);
        aie::store_v(acc + dim, acc_vec.to_vector<float>());
    }
}

static inline void accumulate_value_inplace(float *__restrict acc,
                                            const bfloat16 *__restrict value,
                                            float weight,
                                            int32_t head_dim)
{
    constexpr int vec_len = LLAMA_VEC_SIZE;
    aie::vector<float, vec_len> weight_vec =
        aie::broadcast<float, vec_len>(weight);

    for (int32_t dim = 0; dim < head_dim; dim += vec_len) {
        aie::accum<accfloat, vec_len> acc_vec;
        acc_vec.from_vector(aie::load_v<vec_len>(acc + dim), 0);

        aie::accum<accfloat, vec_len> value_vec;
        value_vec.from_vector(aie::load_v<vec_len>(value + dim), 0);

        aie::accum<accfloat, vec_len> weighted_value =
            aie::mul(value_vec.to_vector<float>(), weight_vec);
        acc_vec = aie::add(acc_vec, weighted_value.to_vector<float>());
        aie::store_v(acc + dim, acc_vec.to_vector<float>());
    }
}

static inline void zero_bf16_vector(bfloat16 *__restrict out,
                                    int32_t head_dim)
{
    constexpr int vec_len = LLAMA_VEC_SIZE;
    aie::vector<bfloat16, vec_len> zero_vec =
        aie::zeros<bfloat16, vec_len>();

    for (int32_t dim = 0; dim < head_dim; dim += vec_len) {
        aie::store_v(out + dim, zero_vec);
    }
}

static inline void zero_float_vector(float *__restrict out,
                                     int32_t head_dim)
{
    constexpr int vec_len = LLAMA_VEC_SIZE;
    aie::vector<float, vec_len> zero_vec =
        aie::zeros<float, vec_len>();

    for (int32_t dim = 0; dim < head_dim; dim += vec_len) {
        aie::store_v(out + dim, zero_vec);
    }
}

static inline void normalize_acc_to_bf16(const float *__restrict acc,
                                         bfloat16 *__restrict out,
                                         float inv,
                                         int32_t head_dim)
{
    constexpr int vec_len = LLAMA_VEC_SIZE;
    aie::vector<float, vec_len> inv_vec =
        aie::broadcast<float, vec_len>(inv);

    for (int32_t dim = 0; dim < head_dim; dim += vec_len) {
        aie::accum<accfloat, vec_len> acc_vec;
        acc_vec.from_vector(aie::load_v<vec_len>(acc + dim), 0);
        acc_vec = aie::mul(acc_vec.to_vector<float>(), inv_vec);
        aie::store_v(out + dim, acc_vec.to_vector<bfloat16>());
    }
}

extern "C" {

void llama_chunked_attention_init_f32(float *__restrict state,
                                      float *__restrict acc,
                                      int32_t q_heads,
                                      int32_t head_dim)
{
    event0();
    constexpr int q_heads_const = LLAMA_Q_HEADS_PER_GROUP;
    constexpr int head_dim_const = LLAMA_HEAD_DIM;
    (void)q_heads;
    (void)head_dim;

    for (int32_t q_head = 0; q_head < q_heads_const; q_head++) {
        state[q_head * 2] = -__builtin_inff();
        state[q_head * 2 + 1] = 0.0f;
        float *__restrict q_acc = acc + q_head * head_dim_const;
        zero_float_vector(q_acc, head_dim_const);
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
    constexpr int q_heads_const = LLAMA_Q_HEADS_PER_GROUP;
    constexpr int chunk_size_const = LLAMA_CHUNK_SIZE;
    constexpr int head_dim_const = LLAMA_HEAD_DIM;
    (void)q_heads;
    (void)chunk_size;
    (void)head_dim;

    const bfloat16 *__restrict k = packed_chunk;
    const bfloat16 *__restrict v = packed_chunk + chunk_size_const * head_dim_const;
    const bfloat16 *__restrict mask =
        packed_chunk + 2 * chunk_size_const * head_dim_const;

    for (int32_t q_head = 0; q_head < q_heads_const; q_head++) {
        const bfloat16 *__restrict q = q_full + q_head * head_dim_const;
        float *__restrict q_state = state + q_head * 2;
        float *__restrict q_acc = acc + q_head * head_dim_const;
        float scores[LLAMA_CHUNK_SIZE];
        float chunk_max = -__builtin_inff();
        bool has_valid = false;
        aie::vector<bfloat16, vec_len> q_vecs[LLAMA_HEAD_BLOCKS];

        for (int32_t block = 0; block < LLAMA_HEAD_BLOCKS; block++)
            chess_flatten_loop
        {
            q_vecs[block] = aie::load_v<vec_len>(q + block * vec_len);
        }

        for (int32_t row = 0; row < chunk_size_const; row++) {
            if (static_cast<float>(mask[row]) <= 0.5f) {
                scores[row] = -__builtin_inff();
                continue;
            }

            aie::accum<accfloat, vec_len> dot = aie::zeros<accfloat, vec_len>();
            const bfloat16 *__restrict k_row = k + row * head_dim_const;
            for (int32_t block = 0; block < LLAMA_HEAD_BLOCKS; block++)
                chess_flatten_loop
            {
                aie::vector<bfloat16, vec_len> k_vec =
                    aie::load_v<vec_len>(k_row + block * vec_len);
                dot = aie::mac(dot, q_vecs[block], k_vec);
            }

            float score =
                aie::reduce_add(dot.template to_vector<float>()) * LLAMA_ATTN_SCALE;
            scores[row] = score;
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

        scale_acc_inplace(q_acc, correction, head_dim_const);

        for (int32_t row = 0; row < chunk_size_const; row++) {
            if (static_cast<float>(mask[row]) <= 0.5f) {
                continue;
            }

            float score = scores[row];
            float weight = exp_approx(score - new_max);
            chunk_sum += weight;

            const bfloat16 *__restrict v_row = v + row * head_dim_const;
            accumulate_value_inplace(q_acc, v_row, weight, head_dim_const);
        }

        q_state[0] = new_max;
        q_state[1] = old_sum * correction + chunk_sum;
    }

    event1();
}

void qwen_chunked_attention_current_direct_bf16(
    const bfloat16 *__restrict q_current,
    const bfloat16 *__restrict packed_chunk,
    float *__restrict state,
    float *__restrict acc,
    int32_t current_row,
    int32_t q_heads,
    int32_t chunk_size,
    int32_t head_dim)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    constexpr int vec_len = LLAMA_VEC_SIZE;
    constexpr int q_heads_const = LLAMA_Q_HEADS_PER_GROUP;
    constexpr int chunk_size_const = LLAMA_CHUNK_SIZE;
    constexpr int head_dim_const = LLAMA_HEAD_DIM;
    (void)q_heads;
    (void)chunk_size;
    (void)head_dim;

    const bfloat16 *__restrict current_key =
        q_current + LLAMA_Q_HEADS_PER_GROUP * head_dim_const;
    const bfloat16 *__restrict current_value = current_key + head_dim_const;
    const bfloat16 *__restrict k = packed_chunk;
    const bfloat16 *__restrict v = packed_chunk + chunk_size_const * head_dim_const;
    const bfloat16 *__restrict mask =
        packed_chunk + 2 * chunk_size_const * head_dim_const;

    for (int32_t q_head = 0; q_head < q_heads_const; q_head++) {
        const bfloat16 *__restrict q = q_current + q_head * head_dim_const;
        float *__restrict q_state = state + q_head * 2;
        float *__restrict q_acc = acc + q_head * head_dim_const;
        float scores[LLAMA_CHUNK_SIZE];
        float chunk_max = -__builtin_inff();
        bool has_valid = false;
        aie::vector<bfloat16, vec_len> q_vecs[LLAMA_HEAD_BLOCKS];

        for (int32_t block = 0; block < LLAMA_HEAD_BLOCKS; block++)
            chess_flatten_loop
        {
            q_vecs[block] = aie::load_v<vec_len>(q + block * vec_len);
        }

        for (int32_t row = 0; row < chunk_size_const; row++) {
            bool row_is_current = row == current_row;
            if (!row_is_current && static_cast<float>(mask[row]) <= 0.5f) {
                scores[row] = -__builtin_inff();
                continue;
            }

            aie::accum<accfloat, vec_len> dot = aie::zeros<accfloat, vec_len>();
            const bfloat16 *__restrict k_row =
                row_is_current ? current_key : k + row * head_dim_const;
            for (int32_t block = 0; block < LLAMA_HEAD_BLOCKS; block++)
                chess_flatten_loop
            {
                aie::vector<bfloat16, vec_len> k_vec =
                    aie::load_v<vec_len>(k_row + block * vec_len);
                dot = aie::mac(dot, q_vecs[block], k_vec);
            }

            float score =
                aie::reduce_add(dot.template to_vector<float>()) * LLAMA_ATTN_SCALE;
            scores[row] = score;
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

        scale_acc_inplace(q_acc, correction, head_dim_const);

        for (int32_t row = 0; row < chunk_size_const; row++) {
            bool row_is_current = row == current_row;
            if (!row_is_current && static_cast<float>(mask[row]) <= 0.5f) {
                continue;
            }

            float score = scores[row];
            float weight = exp_approx(score - new_max);
            chunk_sum += weight;

            const bfloat16 *__restrict v_row =
                row_is_current ? current_value : v + row * head_dim_const;
            accumulate_value_inplace(q_acc, v_row, weight, head_dim_const);
        }

        q_state[0] = new_max;
        q_state[1] = old_sum * correction + chunk_sum;
    }

    event1();
}

void qwen_plane_attention_update_bf16(
    const bfloat16 *__restrict q_current,
    const bfloat16 *__restrict plane_pair_chunk,
    float *__restrict state,
    float *__restrict acc,
    int32_t group_in_plane,
    int32_t current_row,
    int32_t valid_rows,
    int32_t q_heads,
    int32_t chunk_size,
    int32_t head_dim)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    constexpr int vec_len = LLAMA_VEC_SIZE;
    constexpr int q_heads_const = LLAMA_Q_HEADS_PER_GROUP;
    constexpr int chunk_size_const = LLAMA_CHUNK_SIZE;
    constexpr int head_dim_const = LLAMA_HEAD_DIM;
    constexpr int plane_group_count = 4;
    constexpr int plane_row_stride = plane_group_count * head_dim_const;
    (void)q_heads;
    (void)chunk_size;
    (void)head_dim;

    const bfloat16 *__restrict current_key =
        q_current + LLAMA_Q_HEADS_PER_GROUP * head_dim_const;
    const bfloat16 *__restrict current_value = current_key + head_dim_const;
    const bfloat16 *__restrict k_plane_chunk = plane_pair_chunk;
    const bfloat16 *__restrict v_plane_chunk =
        plane_pair_chunk + chunk_size_const * plane_row_stride;

    for (int32_t q_head = 0; q_head < q_heads_const; q_head++) {
        const bfloat16 *__restrict q = q_current + q_head * head_dim_const;
        float *__restrict q_state = state + q_head * 2;
        float *__restrict q_acc = acc + q_head * head_dim_const;
        float scores[LLAMA_CHUNK_SIZE];
        float chunk_max = -__builtin_inff();
        bool has_valid = false;
        aie::vector<bfloat16, vec_len> q_vecs[LLAMA_HEAD_BLOCKS];

        for (int32_t block = 0; block < LLAMA_HEAD_BLOCKS; block++)
            chess_flatten_loop
        {
            q_vecs[block] = aie::load_v<vec_len>(q + block * vec_len);
        }

        for (int32_t row = 0; row < chunk_size_const; row++) {
            if (row >= valid_rows) {
                scores[row] = -__builtin_inff();
                continue;
            }
            bool row_is_current = row == current_row;
            aie::accum<accfloat, vec_len> dot = aie::zeros<accfloat, vec_len>();
            const bfloat16 *__restrict k_row =
                row_is_current
                    ? current_key
                    : k_plane_chunk + row * plane_row_stride
                          + group_in_plane * head_dim_const;
            for (int32_t block = 0; block < LLAMA_HEAD_BLOCKS; block++)
                chess_flatten_loop
            {
                aie::vector<bfloat16, vec_len> k_vec =
                    aie::load_v<vec_len>(k_row + block * vec_len);
                dot = aie::mac(dot, q_vecs[block], k_vec);
            }

            float score =
                aie::reduce_add(dot.template to_vector<float>()) * LLAMA_ATTN_SCALE;
            scores[row] = score;
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

        scale_acc_inplace(q_acc, correction, head_dim_const);

        for (int32_t row = 0; row < chunk_size_const; row++) {
            if (row >= valid_rows) {
                continue;
            }
            bool row_is_current = row == current_row;
            float score = scores[row];
            float weight = exp_approx(score - new_max);
            chunk_sum += weight;

            const bfloat16 *__restrict v_row =
                row_is_current
                    ? current_value
                    : v_plane_chunk + row * plane_row_stride
                          + group_in_plane * head_dim_const;
            accumulate_value_inplace(q_acc, v_row, weight, head_dim_const);
        }

        q_state[0] = new_max;
        q_state[1] = old_sum * correction + chunk_sum;
    }

    event1();
}

void qwen_plane_group_attention_update_bf16(
    const bfloat16 *__restrict q_current,
    const bfloat16 *__restrict plane_group_pair_chunk,
    float *__restrict state,
    float *__restrict acc,
    int32_t current_row,
    int32_t valid_rows,
    int32_t q_heads,
    int32_t chunk_size,
    int32_t head_dim)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    constexpr int vec_len = LLAMA_VEC_SIZE;
    constexpr int q_heads_const = LLAMA_Q_HEADS_PER_GROUP;
    constexpr int chunk_size_const = LLAMA_CHUNK_SIZE;
    constexpr int head_dim_const = LLAMA_HEAD_DIM;
    (void)q_heads;
    (void)chunk_size;
    (void)head_dim;

    const bfloat16 *__restrict current_key =
        q_current + LLAMA_Q_HEADS_PER_GROUP * head_dim_const;
    const bfloat16 *__restrict current_value = current_key + head_dim_const;
    const bfloat16 *__restrict k_plane_chunk = plane_group_pair_chunk;
    const bfloat16 *__restrict v_plane_chunk =
        plane_group_pair_chunk + chunk_size_const * head_dim_const;

    for (int32_t q_head = 0; q_head < q_heads_const; q_head++) {
        const bfloat16 *__restrict q = q_current + q_head * head_dim_const;
        float *__restrict q_state = state + q_head * 2;
        float *__restrict q_acc = acc + q_head * head_dim_const;
        float scores[LLAMA_CHUNK_SIZE];
        float chunk_max = -__builtin_inff();
        bool has_valid = false;
        aie::vector<bfloat16, vec_len> q_vecs[LLAMA_HEAD_BLOCKS];

        for (int32_t block = 0; block < LLAMA_HEAD_BLOCKS; block++)
            chess_flatten_loop
        {
            q_vecs[block] = aie::load_v<vec_len>(q + block * vec_len);
        }

        for (int32_t row = 0; row < chunk_size_const; row++) {
            if (row >= valid_rows) {
                scores[row] = -__builtin_inff();
                continue;
            }
            bool row_is_current = row == current_row;
            aie::accum<accfloat, vec_len> dot = aie::zeros<accfloat, vec_len>();
            const bfloat16 *__restrict k_row =
                row_is_current ? current_key
                               : k_plane_chunk + row * head_dim_const;
            for (int32_t block = 0; block < LLAMA_HEAD_BLOCKS; block++)
                chess_flatten_loop
            {
                aie::vector<bfloat16, vec_len> k_vec =
                    aie::load_v<vec_len>(k_row + block * vec_len);
                dot = aie::mac(dot, q_vecs[block], k_vec);
            }

            float score =
                aie::reduce_add(dot.template to_vector<float>()) * LLAMA_ATTN_SCALE;
            scores[row] = score;
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

        scale_acc_inplace(q_acc, correction, head_dim_const);

        for (int32_t row = 0; row < chunk_size_const; row++) {
            if (row >= valid_rows) {
                continue;
            }
            bool row_is_current = row == current_row;
            float score = scores[row];
            float weight = exp_approx(score - new_max);
            chunk_sum += weight;

            const bfloat16 *__restrict v_row =
                row_is_current ? current_value
                               : v_plane_chunk + row * head_dim_const;
            accumulate_value_inplace(q_acc, v_row, weight, head_dim_const);
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
    constexpr int q_heads_const = LLAMA_Q_HEADS_PER_GROUP;
    constexpr int head_dim_const = LLAMA_HEAD_DIM;
    (void)q_heads;
    (void)head_dim;

    for (int32_t q_head = 0; q_head < q_heads_const; q_head++) {
        const float *__restrict q_state = state + q_head * 2;
        const float *__restrict q_acc = acc + q_head * head_dim_const;
        bfloat16 *__restrict out = out_full + q_head * head_dim_const;
        float denom = q_state[1];
        if (denom <= 0.0f) {
            zero_bf16_vector(out, head_dim_const);
        } else {
            float inv = 1.0f / denom;
            normalize_acc_to_bf16(q_acc, out, inv, head_dim_const);
        }
    }

    event1();
}

} // extern "C"
