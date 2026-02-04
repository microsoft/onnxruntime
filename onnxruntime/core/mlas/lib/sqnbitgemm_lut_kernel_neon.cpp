/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_lut_kernel_neon.cpp

Abstract:

    This module implements ARM64 NEON kernel functions for LUT-based quantized
    n-bit integer matrix multiplication.

    It provides optimized ARM NEON implementations for lookup table generation,
    GEMM computation, and related operations on quantized weight and activation
    matrices.

    Inspired by T-MAC implementation in llama.cpp (https://github.com/microsoft/T-MAC)

--*/

#if defined(MLAS_TARGET_ARM64)

#include <arm_neon.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>
#include <type_traits>

#include "mlasi.h"
#include "qlutgemm.h"
#include "qnbitgemm.h"
#include "sqnbitgemm_q8_block.h"

// Conditional pragma unroll for compiler compatibility
#if defined(__clang__)
#define PRAGMA_UNROLL _Pragma("unroll")
#else
#define PRAGMA_UNROLL
#endif

//
// Template classes for accumulation - adapted from llama.cpp tbl.cpp
//

// Fast aggregation using halving add (vrhaddq_s8)
// Used when ActK is a power of 2 for faster accumulation
template <int N>
struct SignedHalvingAdder {
    SignedHalvingAdder<N / 2> adder;
    int8x16_t lhs;

    inline void push(int8x16_t v, int k)
    {
        if (k < N / 2) {
            adder.push(v, k);
            if (k == N / 2 - 1) {
                lhs = adder.get();
            }
        } else {
            adder.push(v, k - N / 2);
            if (k == N - 1) {
                lhs = vrhaddq_s8(lhs, adder.get());
            }
        }
    }

    inline int8x16_t get()
    {
        return lhs;
    }

    inline int16x8_t get_low()
    {
        return vmovl_s8(vget_low_s8(lhs));
    }

    inline int16x8_t get_high()
    {
        return vmovl_high_s8(lhs);
    }
};

template <>
struct SignedHalvingAdder<2> {
    int8x16_t lhs;

    inline void push(int8x16_t v, int k)
    {
        if (k == 0) {
            lhs = v;
        } else {
            lhs = vrhaddq_s8(lhs, v);
        }
    }

    inline int8x16_t get()
    {
        return lhs;
    }

    inline int16x8_t get_low()
    {
        return vmovl_s8(vget_low_s8(lhs));
    }

    inline int16x8_t get_high()
    {
        return vmovl_high_s8(lhs);
    }
};

// Widening adder for accuracy (no fast aggregation)
// Used when precision is more important than speed
template <int N>
struct SignedWideningAdder {
    static_assert(N > 0, "N parameter exists for API compatibility with SignedHalvingAdder");
    int16x8_t lhs_low = vdupq_n_s16(0);
    int16x8_t lhs_high = vdupq_n_s16(0);

    inline void push(int8x16_t v, int k)
    {
        if (k == 0) {
            lhs_low = vmovl_s8(vget_low_s8(v));
            lhs_high = vmovl_high_s8(v);
        } else {
            lhs_low = vaddq_s16(lhs_low, vmovl_s8(vget_low_s8(v)));
            lhs_high = vaddq_s16(lhs_high, vmovl_high_s8(v));
        }
    }

    inline int16x8_t get_low()
    {
        return lhs_low;
    }

    inline int16x8_t get_high()
    {
        return lhs_high;
    }
};

template <bool FastAggregation, int ActK>
using SignedAdder = typename std::conditional<FastAggregation, SignedHalvingAdder<ActK>, SignedWideningAdder<ActK>>::type;

// Template for computing log2 at compile time
template <int K>
struct mylog2 {
    enum {
        value = 1 + mylog2<K / 2>::value
    };
};

template <>
struct mylog2<0> {
    enum {
        value = -1
    };
};

// Template for computing bias scale at compile time
template <int bits>
constexpr int
get_bias_scale()
{
    // The bias scale will be added to the first bit
    // 15 = (1/2 + 1 + 2 + 4) / (1/2)
    // 7 = (1/2 + 1 + 2) / (1/2)
    // 3 = (1/2 + 1) / (1/2)
    // 1 = (1/2) / (1/2)
    return 3;  // For 2-bit quantization
}

//
// Partial max computation for LUT scale calculation
//
static inline void
partial_max_g4_int8_k8_neon(float* lut_scales, const float* b)
{
    // Process 8 groups of 4 floats each (strided by 4)
    float32x4_t max_abs = vdupq_n_f32(0.0f);

    for (int i = 0; i < 8; i++) {
        // Load 4 consecutive floats from position i*4
        float32x4_t vals = vld1q_f32(b + i * 4);
        float32x4_t abs_vals = vabsq_f32(vals);
        max_abs = vmaxq_f32(max_abs, abs_vals);
    }

    // Horizontal max across the vector
    float max_val = vmaxvq_f32(max_abs);
    float scales = max_val / 127.0f;
    *lut_scales = std::max(*lut_scales, scales);
}

//
// LUT construction for int8 quantized activations
// Builds 16-entry lookup tables for groups of 4 activation values
//
static inline void
lut_ctor_g4_int8_impl_neon(
    int32_t act_k,
    int8_t* qlut,
    const float* b,
    float* lut_scales,
    float* lut_biases
)
{
    float32x4_t vec_lut[16];
    float biases = 0.0f;
    float scales = *lut_scales;
    float t_scales = scales ? 1.0f / scales : 0.0f;

    for (int k = 0; k < act_k / 32; ++k) {
        // Load 4 groups of 8 floats (strided pattern)
        // ORT uses contiguous float layout, so we load and rearrange
        float32x4_t vec_b0, vec_b1, vec_b2, vec_b3;

        // Load first 4 elements from each group of 4
        // Pattern: b[k*32 + i*4 + j] where i=0..7, j=0..3
        // We need vec_b0 = {b[0], b[4], b[8], b[12], b[16], b[20], b[24], b[28]} etc.
        // For NEON with float32, we work with 4 elements at a time

        // Simplified: process 4 lanes at a time
        for (int lane = 0; lane < 2; lane++) {
            const float* base = b + k * 32 + lane * 16;

            // Load 4 values with stride 4
            float b0_vals[4] = {base[0], base[4], base[8], base[12]};
            float b1_vals[4] = {base[1], base[5], base[9], base[13]};
            float b2_vals[4] = {base[2], base[6], base[10], base[14]};
            float b3_vals[4] = {base[3], base[7], base[11], base[15]};

            vec_b0 = vld1q_f32(b0_vals);
            vec_b1 = vld1q_f32(b1_vals);
            vec_b2 = vld1q_f32(b2_vals);
            vec_b3 = vld1q_f32(b3_vals);

            // Build 16-entry LUT: each entry is ±b0 ±b1 ±b2 ±b3
            PRAGMA_UNROLL
            for (int g = 1; g < 16; g += 2) {
                vec_lut[g] = vec_b0;
                if (g & 0b0010) {
                    vec_lut[g] = vaddq_f32(vec_lut[g], vec_b1);
                } else {
                    vec_lut[g] = vsubq_f32(vec_lut[g], vec_b1);
                }
                if (g & 0b0100) {
                    vec_lut[g] = vaddq_f32(vec_lut[g], vec_b2);
                } else {
                    vec_lut[g] = vsubq_f32(vec_lut[g], vec_b2);
                }
                if (g & 0b1000) {
                    vec_lut[g] = vaddq_f32(vec_lut[g], vec_b3);
                } else {
                    vec_lut[g] = vsubq_f32(vec_lut[g], vec_b3);
                }
            }

            // Symmetric: vec_lut[g] = -vec_lut[15 - g]
            PRAGMA_UNROLL
            for (int g = 0; g < 16; g += 2) {
                vec_lut[g] = vnegq_f32(vec_lut[15 - g]);
            }

            // Accumulate bias
            biases += vaddvq_f32(vec_lut[0]);

            // Scale and quantize
            PRAGMA_UNROLL
            for (int g = 0; g < 16; ++g) {
                vec_lut[g] = vmulq_n_f32(vec_lut[g], t_scales);
            }

            // Convert to int8 and store
            int8_t* qlut_dst = qlut + k * 128 + lane * 64;  // 8 * 16 / 2 = 64

            PRAGMA_UNROLL
            for (int g = 0; g < 16; ++g) {
                // Round and convert to int32
                int32x4_t i32 = vcvtnq_s32_f32(vec_lut[g]);
                // Narrow to int16
                int16x4_t i16 = vqmovn_s32(i32);
                // Narrow to int8
                int8x8_t i8 = vqmovn_s16(vcombine_s16(i16, i16));

                // Store individual lanes with proper layout
                qlut_dst[g + 0 * 16] = vget_lane_s8(i8, 0);
                qlut_dst[g + 1 * 16] = vget_lane_s8(i8, 1);
                qlut_dst[g + 2 * 16] = vget_lane_s8(i8, 2);
                qlut_dst[g + 3 * 16] = vget_lane_s8(i8, 3);
            }
        }
    }

    *lut_scales = scales;
    *lut_biases = biases;
}

//
// GenerateLUT - Entry point for LUT generation
//
static void
GenerateLUT_neon(
    const float* b,
    int8_t* qlut,
    float* lut_scales,
    float* lut_biases,
    size_t M,
    size_t K,
    size_t N,
    size_t act_group_size
)
{
    (void)M;  // silence unused parameter warning
    (void)N;  // silence unused parameter warning

    const int32_t kk_outer_max = static_cast<int32_t>(K / act_group_size);
    const int32_t ags_div32 = static_cast<int32_t>(act_group_size / 32);

    // Phase 1: Compute partial max for each activation group
    for (int32_t kk_outer = 0; kk_outer < kk_outer_max; ++kk_outer) {
        lut_scales[kk_outer] = 0.0f;
        for (int32_t k_outer = 0; k_outer < ags_div32; ++k_outer) {
            partial_max_g4_int8_k8_neon(&lut_scales[kk_outer], &b[(kk_outer * act_group_size) + (k_outer * 32)]);
        }
    }

    // Phase 2: Build quantized LUT
    for (int32_t k_outer_1 = 0; k_outer_1 < kk_outer_max; ++k_outer_1) {
        lut_ctor_g4_int8_impl_neon(
            static_cast<int32_t>(act_group_size),
            &qlut[k_outer_1 * act_group_size * 4],
            &b[k_outer_1 * act_group_size],
            &lut_scales[k_outer_1],
            &lut_biases[k_outer_1]
        );
    }
}

//
// Bit gathering for 2-bit results
//
static inline void
tbl_g4_int8_float_gather_bit2_impl_neon(int32_t m, float* C_global, float* CBits, float* C)
{
    constexpr int32_t bits = 2;

    int32_t m_c_outer_max = m / 32;
    for (int32_t m_c_outer = 0; m_c_outer < m_c_outer_max; ++m_c_outer) {
        int32_t cse_var_2 = m_c_outer * 32 * bits;
        int32_t cse_var_1 = m_c_outer * 32;

        PRAGMA_UNROLL
        for (int32_t m_c_inner = 0; m_c_inner < 32; ++m_c_inner) {
            int32_t bit_offset_0 = (m_c_inner / 8) * 8 * bits + (m_c_inner % 8);
            int32_t bit_offset_1 = bit_offset_0 + 8;
            C_global[cse_var_1 + m_c_inner] =
                (CBits[cse_var_2 + bit_offset_0] * 0.5f) + CBits[cse_var_2 + bit_offset_1];
        }
    }

    // Copy to output
    for (int32_t m_inner_outer = 0; m_inner_outer < m_c_outer_max; ++m_inner_outer) {
        PRAGMA_UNROLL
        for (int32_t m_inner = 0; m_inner < 32; ++m_inner) {
            int offset = m_inner_outer * 32 + m_inner;
            C[offset] = C_global[offset];
        }
    }
}

//
// Core GEMM compute kernel using table lookup
//
template <bool has_scale, int K, int Bits, int ActK, bool FastAggregation, bool ZeroPoint, bool OneScale>
inline int32_t
tbl_g4_int8_float_update_impl_neon(
    int32_t m,
    float* c,
    const int8_t* lut,
    const uint8_t* a,
    const float* scales,
    const float* lut_scales,
    const float* lut_biases
)
{
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);
    int8x16_t vec_lut[K];

    // Load LUT tables
    PRAGMA_UNROLL
    for (int k = 0; k < K; k++) {
        vec_lut[k] = vld1q_s8(lut + k * 16);
    }

    SignedAdder<FastAggregation, ActK> adder_bot, adder_top;

    for (int i = 0; i < m / 2; i += 16) {
        float32x4_t vec_c0 = vdupq_n_f32(0.0f);
        float32x4_t vec_c1 = vdupq_n_f32(0.0f);
        float32x4_t vec_c2 = vdupq_n_f32(0.0f);
        float32x4_t vec_c3 = vdupq_n_f32(0.0f);
        float32x4_t vec_c4 = vdupq_n_f32(0.0f);
        float32x4_t vec_c5 = vdupq_n_f32(0.0f);
        float32x4_t vec_c6 = vdupq_n_f32(0.0f);
        float32x4_t vec_c7 = vdupq_n_f32(0.0f);

        float partial_sum = 0.0f;

        PRAGMA_UNROLL
        for (int kk = 0; kk < K; kk += ActK) {
            PRAGMA_UNROLL
            for (int k = 0; k < ActK; k++) {
                // Load packed 4-bit indices
                uint8x16_t vec_as = vld1q_u8(a + i * K + (kk + k) * 16);

                // Extract nibbles
                uint8x16_t vec_a_bot = vandq_u8(vec_as, vec_mask);   // Lower 4 bits
                uint8x16_t vec_a_top = vshrq_n_u8(vec_as, 4);        // Upper 4 bits

                // TABLE LOOKUP - THE KEY OPERATION
                int8x16_t vec_v_bot = vqtbl1q_s8(vec_lut[kk + k], vreinterpretq_u8_s8(vreinterpretq_s8_u8(vec_a_bot)));
                int8x16_t vec_v_top = vqtbl1q_s8(vec_lut[kk + k], vreinterpretq_u8_s8(vreinterpretq_s8_u8(vec_a_top)));

                adder_bot.push(vec_v_bot, k);
                adder_top.push(vec_v_top, k);
            }

            // Widen to int16
            int16x8_t vec_v_bot_low = adder_bot.get_low();
            int16x8_t vec_v_bot_high = adder_bot.get_high();
            int16x8_t vec_v_top_low = adder_top.get_low();
            int16x8_t vec_v_top_high = adder_top.get_high();

            // Convert to float32 (need to widen int16 -> int32 -> float32)
            float32x4_t vec_v_bot_low_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec_v_bot_low)));
            float32x4_t vec_v_bot_low_high = vcvtq_f32_s32(vmovl_high_s16(vec_v_bot_low));
            float32x4_t vec_v_bot_high_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec_v_bot_high)));
            float32x4_t vec_v_bot_high_high = vcvtq_f32_s32(vmovl_high_s16(vec_v_bot_high));
            float32x4_t vec_v_top_low_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec_v_top_low)));
            float32x4_t vec_v_top_low_high = vcvtq_f32_s32(vmovl_high_s16(vec_v_top_low));
            float32x4_t vec_v_top_high_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec_v_top_high)));
            float32x4_t vec_v_top_high_high = vcvtq_f32_s32(vmovl_high_s16(vec_v_top_high));

            float lut_s = lut_scales[kk / ActK];
            float lut_b = lut_biases[kk / ActK];

            if (ZeroPoint) {
                partial_sum += lut_b;
            }

            if (FastAggregation) {
                lut_s = lut_s * ActK;
                lut_b -= lut_s * (mylog2<ActK>::value / 4 * get_bias_scale<Bits>());
            }

            // FMA operations with conditional bias
#define lut_fma(vs, ib) \
    (((ib) % Bits) ? vmulq_n_f32((vs), lut_s) : vmlaq_n_f32(vdupq_n_f32(lut_b), (vs), lut_s))

            if (kk == 0) {
                vec_c0 = lut_fma(vec_v_bot_low_low, (i / 4));
                vec_c1 = lut_fma(vec_v_bot_low_high, (i / 4 + 1));
                vec_c2 = lut_fma(vec_v_bot_high_low, (i / 4 + 2));
                vec_c3 = lut_fma(vec_v_bot_high_high, (i / 4 + 3));
                vec_c4 = lut_fma(vec_v_top_low_low, (i / 4 + 4));
                vec_c5 = lut_fma(vec_v_top_low_high, (i / 4 + 5));
                vec_c6 = lut_fma(vec_v_top_high_low, (i / 4 + 6));
                vec_c7 = lut_fma(vec_v_top_high_high, (i / 4 + 7));
            } else {
                vec_c0 = vaddq_f32(vec_c0, lut_fma(vec_v_bot_low_low, (i / 4)));
                vec_c1 = vaddq_f32(vec_c1, lut_fma(vec_v_bot_low_high, (i / 4 + 1)));
                vec_c2 = vaddq_f32(vec_c2, lut_fma(vec_v_bot_high_low, (i / 4 + 2)));
                vec_c3 = vaddq_f32(vec_c3, lut_fma(vec_v_bot_high_high, (i / 4 + 3)));
                vec_c4 = vaddq_f32(vec_c4, lut_fma(vec_v_top_low_low, (i / 4 + 4)));
                vec_c5 = vaddq_f32(vec_c5, lut_fma(vec_v_top_low_high, (i / 4 + 5)));
                vec_c6 = vaddq_f32(vec_c6, lut_fma(vec_v_top_high_low, (i / 4 + 6)));
                vec_c7 = vaddq_f32(vec_c7, lut_fma(vec_v_top_high_high, (i / 4 + 7)));
            }
#undef lut_fma
        }

        // Apply weight scales and store
        if (ZeroPoint) {
            float32x4_t vec_s0 = vld1q_f32(scales + ((i / 4) / Bits) * 16);
            float32x4_t vec_s1 = vld1q_f32(scales + ((i / 4 + 1) / Bits) * 16);
            float32x4_t vec_s2 = vld1q_f32(scales + ((i / 4 + 2) / Bits) * 16 + 4);
            float32x4_t vec_s3 = vld1q_f32(scales + ((i / 4 + 3) / Bits) * 16 + 4);

            vec_c0 = vfmaq_f32(vld1q_f32(c + i * 2), vec_c0, vec_s0);
            vec_c1 = vfmaq_f32(vld1q_f32(c + i * 2 + 4), vec_c1, vec_s1);
            vec_c2 = vfmaq_f32(vld1q_f32(c + i * 2 + 8), vec_c2, vec_s2);
            vec_c3 = vfmaq_f32(vld1q_f32(c + i * 2 + 12), vec_c3, vec_s3);

            float32x4_t vec_z0 = vld1q_f32(scales + ((i / 4) / Bits) * 16 + 8);
            float32x4_t vec_z1 = vld1q_f32(scales + ((i / 4 + 1) / Bits) * 16 + 8);
            float32x4_t vec_z2 = vld1q_f32(scales + ((i / 4 + 2) / Bits) * 16 + 12);
            float32x4_t vec_z3 = vld1q_f32(scales + ((i / 4 + 3) / Bits) * 16 + 12);

            partial_sum *= 2;

#define add_zero(cs, zs, ib) \
    (((ib) % Bits) ? (cs) : vfmaq_n_f32((cs), (zs), partial_sum))

            vst1q_f32(c + i * 2, add_zero(vec_c0, vec_z0, (i / 4)));
            vst1q_f32(c + i * 2 + 4, add_zero(vec_c1, vec_z1, (i / 4 + 1)));
            vst1q_f32(c + i * 2 + 8, add_zero(vec_c2, vec_z2, (i / 4 + 2)));
            vst1q_f32(c + i * 2 + 12, add_zero(vec_c3, vec_z3, (i / 4 + 3)));
            vst1q_f32(c + i * 2 + 16, vec_c4);
            vst1q_f32(c + i * 2 + 20, vec_c5);
            vst1q_f32(c + i * 2 + 24, vec_c6);
            vst1q_f32(c + i * 2 + 28, vec_c7);
#undef add_zero
        } else if (OneScale) {
            float single_scale = scales[0];
            float32x4_t vec_s = vdupq_n_f32(single_scale);

            vst1q_f32(c + i * 2, vfmaq_f32(vld1q_f32(c + i * 2), vec_c0, vec_s));
            vst1q_f32(c + i * 2 + 4, vfmaq_f32(vld1q_f32(c + i * 2 + 4), vec_c1, vec_s));
            vst1q_f32(c + i * 2 + 8, vfmaq_f32(vld1q_f32(c + i * 2 + 8), vec_c2, vec_s));
            vst1q_f32(c + i * 2 + 12, vfmaq_f32(vld1q_f32(c + i * 2 + 12), vec_c3, vec_s));
            vst1q_f32(c + i * 2 + 16, vfmaq_f32(vld1q_f32(c + i * 2 + 16), vec_c4, vec_s));
            vst1q_f32(c + i * 2 + 20, vfmaq_f32(vld1q_f32(c + i * 2 + 20), vec_c5, vec_s));
            vst1q_f32(c + i * 2 + 24, vfmaq_f32(vld1q_f32(c + i * 2 + 24), vec_c6, vec_s));
            vst1q_f32(c + i * 2 + 28, vfmaq_f32(vld1q_f32(c + i * 2 + 28), vec_c7, vec_s));
        } else {
            float32x4_t vec_s0 = vld1q_f32(scales + ((i / 4) / Bits) * 8);
            float32x4_t vec_s1 = vld1q_f32(scales + ((i / 4 + 1) / Bits) * 8);
            float32x4_t vec_s2 = vld1q_f32(scales + ((i / 4 + 2) / Bits) * 8 + 4);
            float32x4_t vec_s3 = vld1q_f32(scales + ((i / 4 + 3) / Bits) * 8 + 4);

            vst1q_f32(c + i * 2, vfmaq_f32(vld1q_f32(c + i * 2), vec_c0, vec_s0));
            vst1q_f32(c + i * 2 + 4, vfmaq_f32(vld1q_f32(c + i * 2 + 4), vec_c1, vec_s1));
            vst1q_f32(c + i * 2 + 8, vfmaq_f32(vld1q_f32(c + i * 2 + 8), vec_c2, vec_s2));
            vst1q_f32(c + i * 2 + 12, vfmaq_f32(vld1q_f32(c + i * 2 + 12), vec_c3, vec_s3));
            vst1q_f32(c + i * 2 + 16, vfmaq_f32(vld1q_f32(c + i * 2 + 16), vec_c4, vec_s0));
            vst1q_f32(c + i * 2 + 20, vfmaq_f32(vld1q_f32(c + i * 2 + 20), vec_c5, vec_s1));
            vst1q_f32(c + i * 2 + 24, vfmaq_f32(vld1q_f32(c + i * 2 + 24), vec_c6, vec_s2));
            vst1q_f32(c + i * 2 + 28, vfmaq_f32(vld1q_f32(c + i * 2 + 28), vec_c7, vec_s3));
        }
    }

    return 0;
}

//
// TMACComputeGemm - Entry point for GEMM computation
//
static void
TMACComputeGemm_neon(
    const uint8_t* A,
    const float* Scales,
    const int8_t* LUT,
    const float* LUT_Scales,
    const float* LUT_Biases,
    float* C,
    int K,
    int M,
    int N,
    size_t BlkLen,
    bool HasZeroPoint
)
{
    // Validate batch size
    if (N != 1) {
        MLAS_THROW_EX(std::runtime_error, "N > 1 is not supported yet");
    }

    // Get kernel config
    const MlasTMACKernelParams& tmac_params = MlasGetLutGemmKernelParams(M, K, 2, BlkLen, HasZeroPoint);

    // Configuration
    bool has_zero_point = tmac_params.has_zero_point;
    bool one_scale = tmac_params.one_scale;

    const int32_t bits = static_cast<int32_t>(tmac_params.bits);
    const int32_t g = static_cast<int32_t>(tmac_params.g);
    const int32_t ngroups_per_elem = static_cast<int32_t>(tmac_params.ngroups_per_elem);
    const int32_t kfactor = static_cast<int32_t>(tmac_params.kfactor);

    const bool has_scale = tmac_params.has_scale;

    const int32_t q_group_size = static_cast<int32_t>(tmac_params.q_group_size);
    const int32_t act_group_size = static_cast<int32_t>(tmac_params.act_group_size);
    const int32_t actk = static_cast<int32_t>(tmac_params.actk);

    const int32_t bm = static_cast<int32_t>(tmac_params.bm);
    int32_t m = bm / bits;

    // Validate configuration
    assert(bm % bits == 0);
    assert(K % (kfactor * g) == 0);
    assert(BlkLen % g == 0);

    // Allocate buffers
    std::unique_ptr<float[]> CBits(new float[bm]);
    std::unique_ptr<float[]> C_global(new float[m]);

    std::memset(CBits.get(), 0, bm * sizeof(float));

    // Calculate loop parameters
    const int32_t k_outer_max = K / (kfactor * g);
    const int32_t scale_gs = q_group_size / (kfactor * g);

    int32_t scale_idx_shfr = 0;
    if (scale_gs == 1) {
        scale_idx_shfr = 0;
    } else if (scale_gs == 2) {
        scale_idx_shfr = 1;
    } else if (scale_gs == 4) {
        scale_idx_shfr = 2;
    } else if (scale_gs == 8) {
        scale_idx_shfr = 3;
    } else {
        MLAS_THROW_EX(std::runtime_error, "Unsupported scale_gs configuration");
    }

    // Main computation loop
    for (int32_t k_outer = 0; k_outer < k_outer_max; k_outer++) {
        const uint8_t* a = A + k_outer * bm * kfactor / ngroups_per_elem;

        const float* scales = one_scale ? reinterpret_cast<const float*>(Scales) :
                                  (has_zero_point ? reinterpret_cast<const float*>(Scales) + (k_outer >> scale_idx_shfr) * m * 2 :
                                       reinterpret_cast<const float*>(Scales) + (k_outer >> scale_idx_shfr) * m);

        const int8_t* lut = reinterpret_cast<const int8_t*>(LUT) + k_outer * kfactor * (1 << g);
        const float* lut_scales = reinterpret_cast<const float*>(LUT_Scales) +
                                  (k_outer * kfactor * g / act_group_size);
        const float* lut_biases = reinterpret_cast<const float*>(LUT_Biases) +
                                  (k_outer * kfactor * g / act_group_size);

        // Select appropriate kernel template
        if (has_scale && kfactor == 16 && bits == 2 && actk == 16 && has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl_neon<true, 16, 2, 16, false, true, false>(
                static_cast<int32_t>(bm), CBits.get(), lut, a, scales, lut_scales, lut_biases
            );
        } else if (has_scale && kfactor == 16 && bits == 2 && actk == 16 && !has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl_neon<true, 16, 2, 16, false, false, false>(
                static_cast<int32_t>(bm), CBits.get(), lut, a, scales, lut_scales, lut_biases
            );
        } else if (has_scale && kfactor == 16 && bits == 2 && actk == 16 && !has_zero_point && one_scale) {
            tbl_g4_int8_float_update_impl_neon<true, 16, 2, 16, false, false, true>(
                static_cast<int32_t>(bm), CBits.get(), lut, a, scales, lut_scales, lut_biases
            );
        }
        // actk == 8 variants (for BlkLen=32)
        else if (has_scale && kfactor == 16 && bits == 2 && actk == 8 && has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl_neon<true, 16, 2, 8, false, true, false>(
                static_cast<int32_t>(bm), CBits.get(), lut, a, scales, lut_scales, lut_biases
            );
        } else if (has_scale && kfactor == 16 && bits == 2 && actk == 8 && !has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl_neon<true, 16, 2, 8, false, false, false>(
                static_cast<int32_t>(bm), CBits.get(), lut, a, scales, lut_scales, lut_biases
            );
        } else if (has_scale && kfactor == 16 && bits == 2 && actk == 8 && !has_zero_point && one_scale) {
            tbl_g4_int8_float_update_impl_neon<true, 16, 2, 8, false, false, true>(
                static_cast<int32_t>(bm), CBits.get(), lut, a, scales, lut_scales, lut_biases
            );
        }
        // kfactor == 8 variants
        else if (has_scale && kfactor == 8 && bits == 2 && actk == 8 && has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl_neon<true, 8, 2, 8, false, true, false>(
                static_cast<int32_t>(bm), CBits.get(), lut, a, scales, lut_scales, lut_biases
            );
        } else if (has_scale && kfactor == 8 && bits == 2 && actk == 8 && !has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl_neon<true, 8, 2, 8, false, false, false>(
                static_cast<int32_t>(bm), CBits.get(), lut, a, scales, lut_scales, lut_biases
            );
        } else if (has_scale && kfactor == 8 && bits == 2 && actk == 8 && !has_zero_point && one_scale) {
            tbl_g4_int8_float_update_impl_neon<true, 8, 2, 8, false, false, true>(
                static_cast<int32_t>(bm), CBits.get(), lut, a, scales, lut_scales, lut_biases
            );
        } else {
            MLAS_THROW_EX(std::runtime_error, "No matching kernel found for T-MAC GEMM");
        }
    }

    // Gather results
    tbl_g4_int8_float_gather_bit2_impl_neon(m, C_global.get(), CBits.get(), C);
}

//
// Weight packing for NEON (can use scalar or NEON implementation)
// This is done during model load, so performance is less critical
//
static void
PackQuantBData_neon(
    size_t N,
    size_t K,
    size_t bits,
    size_t g,
    size_t ngroups_per_elem,
    size_t simd_n_in,
    size_t simd_n_out,
    size_t bm,
    size_t kfactor,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* ThreadPool
)
{
    // Only optimized for 2-bit, g=4, ngroups_per_elem=2
    assert(bits == 2 && g == 4 && ngroups_per_elem == 2);

    const size_t mgroup = ngroups_per_elem * simd_n_in;  // 32
    const size_t K_div_g = K / g;

    // Phase 1: Bit-plane decomposition
    std::unique_ptr<uint8_t[]> buf(new uint8_t[N * bits * K_div_g]);

    // Parallelize over N
    MlasTrySimpleParallel(
        ThreadPool, static_cast<ptrdiff_t>(N),
        [&](ptrdiff_t tid) {
            size_t im = static_cast<size_t>(tid);

            const uint8_t* src_row = reinterpret_cast<const uint8_t*>(QuantBDataBegin) + (im * K / 4);
            uint8_t* dst_bit0 = buf.get() + im * bits * K_div_g;
            uint8_t* dst_bit1 = dst_bit0 + K_div_g;

            // Initialize to zero
            std::memset(dst_bit0, 0, K_div_g);
            std::memset(dst_bit1, 0, K_div_g);

            // NEON-accelerated bit extraction
            size_t ik = 0;
            const uint8x16_t mask_2bit = vdupq_n_u8(0x03);
            const uint8x16_t mask_bit0 = vdupq_n_u8(0x01);

            // Process 64 elements at a time (16 bytes input = 64 2-bit elements)
            for (; ik + 64 <= K; ik += 64) {
                uint8x16_t packed = vld1q_u8(src_row + ik / 4);

                // Extract each of 4 positions
                uint8x16_t pos0 = vandq_u8(packed, mask_2bit);
                uint8x16_t pos1 = vandq_u8(vshrq_n_u8(packed, 2), mask_2bit);
                uint8x16_t pos2 = vandq_u8(vshrq_n_u8(packed, 4), mask_2bit);
                uint8x16_t pos3 = vshrq_n_u8(packed, 6);

                // Extract bit 0 from each position
                uint8x16_t b0_pos0 = vandq_u8(pos0, mask_bit0);
                uint8x16_t b0_pos1 = vandq_u8(pos1, mask_bit0);
                uint8x16_t b0_pos2 = vandq_u8(pos2, mask_bit0);
                uint8x16_t b0_pos3 = vandq_u8(pos3, mask_bit0);

                // Combine for bit 0 plane
                uint8x16_t bit0_out = vorrq_u8(
                    vorrq_u8(b0_pos0, vshlq_n_u8(b0_pos1, 1)),
                    vorrq_u8(vshlq_n_u8(b0_pos2, 2), vshlq_n_u8(b0_pos3, 3))
                );

                // Extract bit 1 from each position
                uint8x16_t b1_pos0 = vandq_u8(vshrq_n_u8(pos0, 1), mask_bit0);
                uint8x16_t b1_pos1 = vandq_u8(vshrq_n_u8(pos1, 1), mask_bit0);
                uint8x16_t b1_pos2 = vandq_u8(vshrq_n_u8(pos2, 1), mask_bit0);
                uint8x16_t b1_pos3 = vandq_u8(vshrq_n_u8(pos3, 1), mask_bit0);

                // Combine for bit 1 plane
                uint8x16_t bit1_out = vorrq_u8(
                    vorrq_u8(b1_pos0, vshlq_n_u8(b1_pos1, 1)),
                    vorrq_u8(vshlq_n_u8(b1_pos2, 2), vshlq_n_u8(b1_pos3, 3))
                );

                vst1q_u8(dst_bit0 + ik / g, bit0_out);
                vst1q_u8(dst_bit1 + ik / g, bit1_out);
            }

            // Handle remaining elements with scalar code
            for (; ik < K; ++ik) {
                size_t idx = ik;
                size_t num_elem_per_byte = 4;
                size_t elem_idx = idx % num_elem_per_byte;
                uint8_t v = src_row[idx / num_elem_per_byte] >> (elem_idx * bits);

                size_t new_ik = ik / g;
                size_t shft_left = ik % g;
                dst_bit0[new_ik] += static_cast<uint8_t>(((v >> 0) & 1) << shft_left);
                dst_bit1[new_ik] += static_cast<uint8_t>(((v >> 1) & 1) << shft_left);
            }
        }
    );

    // Phase 2: Multi-reshape/transpose into final layout
    const size_t bm_div_mgroup = bm / mgroup;

    const size_t c2_fac3_div = simd_n_in;
    const size_t c2_fac2_div = kfactor * c2_fac3_div;
    const size_t c2_fac1_div = bm_div_mgroup * c2_fac2_div;
    const size_t c2_fac0_div = K_div_g * bm_div_mgroup * simd_n_in;

    const size_t PackedQuantBDataSize = (N * bits) * (K_div_g / ngroups_per_elem);
    memset(PackedQuantBDataBegin, 0, PackedQuantBDataSize);
    auto* packed_u8 = reinterpret_cast<uint8_t*>(PackedQuantBDataBegin);

    const size_t im_per_tile = ngroups_per_elem * simd_n_out;
    const size_t num_tiles = (N + im_per_tile - 1) / im_per_tile;

    MlasTrySimpleParallel(
        ThreadPool, static_cast<ptrdiff_t>(num_tiles),
        [&](ptrdiff_t tid) {
            const size_t im_start = static_cast<size_t>(tid) * im_per_tile;
            const size_t im_end = std::min(im_start + im_per_tile, N);

            for (size_t im = im_start; im < im_end; ++im) {
                const size_t im0 = im / simd_n_out;
                const size_t isno = im - im0 * simd_n_out;
                const size_t x_base = simd_n_out * (im0 * bits) + isno;

                for (size_t ib = 0; ib < bits; ib++) {
                    const size_t x = x_base + ib * simd_n_out;
                    const size_t new_im1 = x / mgroup;
                    const size_t y = x - new_im1 * mgroup;
                    const size_t new_ing = y / simd_n_in;
                    const size_t new_isni = y - new_ing * simd_n_in;

                    const size_t new_im2 = new_im1 / bm_div_mgroup;
                    const size_t new_ibm = new_im1 - new_im2 * bm_div_mgroup;

                    const size_t base_im = new_im2 * c2_fac0_div + new_ibm * c2_fac2_div + new_isni;
                    const size_t buf_base = im * bits * K_div_g + ib * K_div_g;

                    const uint8_t shift = static_cast<uint8_t>(new_ing * g);
                    const size_t stride = c2_fac3_div;

                    for (size_t ik = 0; ik < K_div_g; ik += kfactor) {
                        const size_t new_ik = ik / kfactor;
                        const size_t base_k = base_im + new_ik * c2_fac1_div;
                        const size_t buf_k = buf_base + ik;

                        uint8_t* dst = packed_u8 + base_k;
                        const uint8_t* src = buf.get() + buf_k;

                        for (size_t ikf = 0; ikf < kfactor; ikf++) {
                            dst[stride * ikf] = static_cast<uint8_t>(dst[stride * ikf] + (src[ikf] << shift));
                        }
                    }
                }
            }
        }
    );
}

//
// Scales and zero points packing
//
template <bool HasZeroPoint>
static void
PackScalesAndZeroPoints_neon_impl(
    size_t N,
    size_t K,
    size_t bits,
    size_t BlkLen,
    size_t simd_n_out,
    size_t bm,
    float* PackedScalesBegin,
    const float* QuantBScale,
    const uint8_t* QuantBZeroPoint,
    MLAS_THREADPOOL* ThreadPool
)
{
    if constexpr (HasZeroPoint) {
        assert(QuantBZeroPoint != nullptr);
    }

    const size_t num_elem_per_byte = 8 / bits;
    const size_t row_blks = K / BlkLen;
    const size_t zp_bytes_per_col = (row_blks + num_elem_per_byte - 1) / num_elem_per_byte;

    const size_t nb1 = K / BlkLen;
    const size_t bm_div_bits = bm / bits;
    const int midpoint = 1 << (bits - 1);
    const uint8_t bits_mask = static_cast<uint8_t>((1 << bits) - 1);

    MlasTrySimpleParallel(
        ThreadPool, static_cast<ptrdiff_t>(N),
        [&](ptrdiff_t tid) {
            size_t im = static_cast<size_t>(tid);
            const size_t new_im = (bm_div_bits > 0) ? (im / bm_div_bits) : 0;
            const size_t new_ibm = (bm_div_bits > 0) ? (im - new_im * bm_div_bits) : 0;

            if constexpr (HasZeroPoint) {
                const size_t new_isimd = new_ibm % simd_n_out;
                const size_t new_ibm_div_simd = new_ibm / simd_n_out;
                const size_t outer_base = new_im * (bm_div_bits * nb1 / simd_n_out) + new_ibm_div_simd;
                const size_t outer_stride = bm_div_bits / simd_n_out;

                for (size_t blk_in_col = 0; blk_in_col < row_blks; blk_in_col++) {
                    const size_t idx = im * nb1 + blk_in_col;
                    const float scale = QuantBScale[idx];

                    size_t zp_byte_idx = im * zp_bytes_per_col + blk_in_col / num_elem_per_byte;
                    size_t elem_idx = blk_in_col % num_elem_per_byte;
                    uint8_t v = (QuantBZeroPoint[zp_byte_idx] >> (elem_idx * bits)) & bits_mask;
                    float zp = static_cast<float>(static_cast<int>(v) - midpoint) * scale;

                    const size_t new_idx_outer = outer_base + blk_in_col * outer_stride;
                    const size_t new_idx_scale = new_idx_outer * (simd_n_out * 2) + new_isimd;
                    const size_t new_idx_zero = new_idx_scale + simd_n_out;

                    PackedScalesBegin[new_idx_scale] = scale;
                    PackedScalesBegin[new_idx_zero] = zp;
                }
            } else {
                const size_t base_idx = new_im * bm_div_bits * nb1 + new_ibm;
                const size_t stride_idx = bm_div_bits;

                for (size_t blk_in_col = 0; blk_in_col < row_blks; blk_in_col++) {
                    const size_t idx = im * nb1 + blk_in_col;
                    const float scale = QuantBScale[idx];
                    const size_t new_idx = base_idx + blk_in_col * stride_idx;
                    PackedScalesBegin[new_idx] = scale;
                }
            }
        }
    );
}

static void
PackScalesAndZeroPoints_neon(
    size_t N,
    size_t K,
    size_t bits,
    size_t BlkLen,
    size_t simd_n_out,
    size_t bm,
    bool HasZeroPoint,
    float* PackedScalesBegin,
    const float* QuantBScale,
    const uint8_t* QuantBZeroPoint,
    MLAS_THREADPOOL* ThreadPool
)
{
    assert(bits == 2);

    if (HasZeroPoint) {
        PackScalesAndZeroPoints_neon_impl<true>(
            N, K, bits, BlkLen, simd_n_out, bm,
            PackedScalesBegin, QuantBScale, QuantBZeroPoint, ThreadPool
        );
    } else {
        PackScalesAndZeroPoints_neon_impl<false>(
            N, K, bits, BlkLen, simd_n_out, bm,
            PackedScalesBegin, QuantBScale, QuantBZeroPoint, ThreadPool
        );
    }
}

//
// Kernel dispatch structure definition
//
const MLAS_QNBIT_LUT_GEMM_DISPATCH MlasLutGenKernelNeon = []() {
    MLAS_QNBIT_LUT_GEMM_DISPATCH d;
    d.GenerateLUT = GenerateLUT_neon;
    d.ComputeGemm = TMACComputeGemm_neon;
    d.PackQuantBData = PackQuantBData_neon;
    d.PackScalesAndZeroPoints = PackScalesAndZeroPoints_neon;
    return d;
}();

#endif  // MLAS_TARGET_ARM64
