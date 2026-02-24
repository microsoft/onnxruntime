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

#include "mlas.h"

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

namespace lutgemm_neon
{

namespace
{

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
// Partial max computation for LUT scale calculation - SCALAR VERSION
// Computes: max(|b0| + |b1| + |b2| + |b3|) for 8 groups of 4 consecutive elements
// This is a direct port of the AVX2 algorithm to scalar for correctness verification
//
static inline void
partial_max_g4_int8_k8_neon(float* lut_scales, const float* b)
{
    // 8 groups of 4 consecutive elements each
    // Groups: {0-3}, {4-7}, {8-11}, {12-15}, {16-19}, {20-23}, {24-27}, {28-31}
    float max_abssum = 0.0f;
    
    for (int group = 0; group < 8; ++group) {
        float abssum = std::abs(b[group * 4 + 0]) +
                       std::abs(b[group * 4 + 1]) +
                       std::abs(b[group * 4 + 2]) +
                       std::abs(b[group * 4 + 3]);
        max_abssum = std::max(max_abssum, abssum);
    }
    
    float scales = max_abssum / 127.0f;
    *lut_scales = std::max(*lut_scales, scales);
}

//
// LUT construction - SCALAR VERSION 
// This is a direct port of the AVX2 algorithm for correctness verification
// Output layout matches AVX2: qlut[k * 128 + group * 16 + lut_entry]
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
    float biases = 0.0f;
    float scales = *lut_scales;
    float t_scales = scales ? 1.0f / scales : 0.0f;

    for (int k = 0; k < act_k / 32; ++k) {
        // For each of 8 groups of 4 consecutive elements
        // Group g contains elements: b[k*32 + g*4 + 0..3]
        float lut[16][8];  // lut[lut_entry][group]
        
        for (int group = 0; group < 8; ++group) {
            // Get the 4 elements in this group
            float b0 = b[k * 32 + group * 4 + 0];
            float b1 = b[k * 32 + group * 4 + 1];
            float b2 = b[k * 32 + group * 4 + 2];
            float b3 = b[k * 32 + group * 4 + 3];
            
            // Build 16-entry LUT using ±b0 ±b1 ±b2 ±b3
            // Odd entries first (g = 1, 3, 5, ..., 15)
            for (int g = 1; g < 16; g += 2) {
                float val = b0;
                if (g & 0b0010) {
                    val += b1;
                } else {
                    val -= b1;
                }
                if (g & 0b0100) {
                    val += b2;
                } else {
                    val -= b2;
                }
                if (g & 0b1000) {
                    val += b3;
                } else {
                    val -= b3;
                }
                lut[g][group] = val;
            }
            
            // Even entries: lut[g] = -lut[15 - g]
            for (int g = 0; g < 16; g += 2) {
                lut[g][group] = -lut[15 - g][group];
            }
        }
        
        // Accumulate bias from lut[0] (sum across all 8 groups)
        for (int group = 0; group < 8; ++group) {
            biases += lut[0][group];
        }
        
        // Scale and quantize, then store
        // Output layout: qlut[k * 128 + group * 16 + lut_entry]
        for (int group = 0; group < 8; ++group) {
            for (int g = 0; g < 16; ++g) {
                float scaled = lut[g][group] * t_scales;
                // Round to nearest, clamp to int8 range
                int32_t rounded = static_cast<int32_t>(std::round(scaled));
                rounded = std::max(-128, std::min(127, rounded));
                qlut[k * 128 + group * 16 + g] = static_cast<int8_t>(rounded);
            }
        }
    }
    
    *lut_scales = scales;
    *lut_biases = biases;
}

}  // namespace

//
// LutGemmGenerateLUT_CompFp32 - Entry point for LUT generation
//
void
LutGemmGenerateLUT_CompFp32(
    const float* b,
    int8_t* qlut,
    float* lut_scales,
    float* lut_biases,
    size_t M,
    size_t K,
    size_t N,
    size_t act_group_size,
    size_t lut_stride
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
            &qlut[k_outer_1 * lut_stride],
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

    // Handle tail cases where m is not a multiple of 32.
    // This ensures C_global is fully initialized for all m elements.
    int32_t m_tail = m % 32;
    if (m_tail > 0) {
        int32_t m_c_outer = m_c_outer_max;
        int32_t cse_var_2 = (m_c_outer * 32 * bits);
        int32_t cse_var_1 = (m_c_outer * 32);
        for (int32_t m_c_inner = 0; m_c_inner < m_tail; ++m_c_inner) {
            int32_t bit_offset_0 = (m_c_inner / 8) * 8 * bits + (m_c_inner % 8);
            int32_t bit_offset_1 = (m_c_inner / 8) * 8 * bits + (m_c_inner % 8) + 8;
            C_global[cse_var_1 + m_c_inner] = (CBits[cse_var_2 + bit_offset_0] * 0.5f) + CBits[cse_var_2 + bit_offset_1];
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

    // Transfer the remaining tail results from C_global to the final output matrix C.
    // This is necessary when m is not a multiple of 32, ensuring all output features
    // are correctly written to the destination buffer.
    if (m_tail > 0) {
        int offset_base = m_c_outer_max * 32;
        for (int32_t m_inner = 0; m_inner < m_tail; ++m_inner) {
            int offset = offset_base + m_inner;
            C[offset] = C_global[offset];
        }
    }
}

//
// Core GEMM compute kernel using table lookup - NEON FP32 VERSION
// Adapted from llama.cpp T-MAC FP16 NEON to use FP32
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

    // Load LUT vectors
    PRAGMA_UNROLL
    for (int k = 0; k < K; k++) {
        vec_lut[k] = vld1q_s8(lut + k * 16);
    }

    SignedAdder<FastAggregation, ActK> adder_bot, adder_top;
    
    for (int i = 0; i < m / 2; i += 16) {
        // For FP32, we need 8 vectors of 4 floats each to cover 32 outputs
        // (compared to FP16's 4 vectors of 8 floats)
        float32x4_t vec_c0, vec_c1, vec_c2, vec_c3, vec_c4, vec_c5, vec_c6, vec_c7;

        float partial_sum = 0.0f;

        PRAGMA_UNROLL
        for (int kk = 0; kk < K; kk += ActK) {
            PRAGMA_UNROLL
            for (int k = 0; k < ActK; k++) {
                // Load 16 packed bytes containing 32 4-bit indices
                uint8x16_t vec_as = vld1q_u8(a + i * K + (kk + k) * 16);
                uint8x16_t vec_a_top = vshrq_n_u8(vec_as, 4);
                uint8x16_t vec_a_bot = vandq_u8(vec_as, vec_mask);

                // Table lookup - get int8 values from LUT
                // Note: vqtbl1q_s8 takes uint8x16_t as index type
                int8x16_t vec_v_bot_tmp = vqtbl1q_s8(vec_lut[kk + k], vec_a_bot);
                int8x16_t vec_v_top_tmp = vqtbl1q_s8(vec_lut[kk + k], vec_a_top);
                
                // Accumulate using appropriate adder
                adder_bot.push(vec_v_bot_tmp, k);
                adder_top.push(vec_v_top_tmp, k);
            }

            // Get accumulated int16 values
            int16x8_t sum_bot_low = adder_bot.get_low();    // bot elements 0-7
            int16x8_t sum_bot_high = adder_bot.get_high();  // bot elements 8-15
            int16x8_t sum_top_low = adder_top.get_low();    // top elements 0-7
            int16x8_t sum_top_high = adder_top.get_high();  // top elements 8-15

            // Convert to FP32 - each int16x8_t becomes two float32x4_t
            // vec_v_*_lo = first 4 elements, vec_v_*_hi = last 4 elements
            float32x4_t vec_v_bot_low_lo  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(sum_bot_low)));
            float32x4_t vec_v_bot_low_hi  = vcvtq_f32_s32(vmovl_high_s16(sum_bot_low));
            float32x4_t vec_v_bot_high_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(sum_bot_high)));
            float32x4_t vec_v_bot_high_hi = vcvtq_f32_s32(vmovl_high_s16(sum_bot_high));
            float32x4_t vec_v_top_low_lo  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(sum_top_low)));
            float32x4_t vec_v_top_low_hi  = vcvtq_f32_s32(vmovl_high_s16(sum_top_low));
            float32x4_t vec_v_top_high_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(sum_top_high)));
            float32x4_t vec_v_top_high_hi = vcvtq_f32_s32(vmovl_high_s16(sum_top_high));

            float lut_s = lut_scales[kk / (ActK * 4)];
            float lut_b = lut_biases[kk / (ActK * 4)];

            if (ZeroPoint) {
                partial_sum += lut_b;
            }

            if (FastAggregation) {
                lut_s = lut_s * ActK;
                lut_b -= lut_s * (mylog2<ActK>::value / 4.0f * get_bias_scale<Bits>());
            }

            float32x4_t vec_lut_s = vdupq_n_f32(lut_s);
            float32x4_t vec_lut_b = vdupq_n_f32(lut_b);

            // lut_fma: ((ib % Bits) ? (v * lut_s) : (v * lut_s + lut_b))
            // ib for each group:
            // Group 0 (c0,c1): ib = i/4
            // Group 1 (c2,c3): ib = i/4 + 1
            // Group 2 (c4,c5): ib = i/4 + 2
            // Group 3 (c6,c7): ib = i/4 + 3
            
            int ib0 = i / 4;
            int ib1 = i / 4 + 1;
            int ib2 = i / 4 + 2;
            int ib3 = i / 4 + 3;

#define LUT_FMA(vec_v, ib_val) \
    (((ib_val) % Bits) ? vmulq_f32(vec_v, vec_lut_s) : vmlaq_f32(vec_lut_b, vec_v, vec_lut_s))

            if (kk == 0) {
                vec_c0 = LUT_FMA(vec_v_bot_low_lo, ib0);
                vec_c1 = LUT_FMA(vec_v_bot_low_hi, ib0);
                vec_c2 = LUT_FMA(vec_v_bot_high_lo, ib1);
                vec_c3 = LUT_FMA(vec_v_bot_high_hi, ib1);
                vec_c4 = LUT_FMA(vec_v_top_low_lo, ib2);
                vec_c5 = LUT_FMA(vec_v_top_low_hi, ib2);
                vec_c6 = LUT_FMA(vec_v_top_high_lo, ib3);
                vec_c7 = LUT_FMA(vec_v_top_high_hi, ib3);
            } else {
                vec_c0 = vaddq_f32(vec_c0, LUT_FMA(vec_v_bot_low_lo, ib0));
                vec_c1 = vaddq_f32(vec_c1, LUT_FMA(vec_v_bot_low_hi, ib0));
                vec_c2 = vaddq_f32(vec_c2, LUT_FMA(vec_v_bot_high_lo, ib1));
                vec_c3 = vaddq_f32(vec_c3, LUT_FMA(vec_v_bot_high_hi, ib1));
                vec_c4 = vaddq_f32(vec_c4, LUT_FMA(vec_v_top_low_lo, ib2));
                vec_c5 = vaddq_f32(vec_c5, LUT_FMA(vec_v_top_low_hi, ib2));
                vec_c6 = vaddq_f32(vec_c6, LUT_FMA(vec_v_top_high_lo, ib3));
                vec_c7 = vaddq_f32(vec_c7, LUT_FMA(vec_v_top_high_hi, ib3));
            }
#undef LUT_FMA
        }

        // Apply weight scales and store
        if (ZeroPoint) {
            partial_sum *= 2;
            float32x4_t vec_ps = vdupq_n_f32(partial_sum);

            // For ZeroPoint mode, scales are interleaved with zero points
            // scales[base+0..7] = scales, scales[base+8..15] = zero_points
            int base0 = ((i / 4) / Bits) * 16;
            int base1 = ((i / 4 + 1) / Bits) * 16;
            int base2 = ((i / 4 + 2) / Bits) * 16;
            int base3 = ((i / 4 + 3) / Bits) * 16;

            // Load scales (first 8 of each 16-element group)
            float32x4_t s0_lo = vld1q_f32(scales + base0);
            float32x4_t s0_hi = vld1q_f32(scales + base0 + 4);
            float32x4_t s1_lo = vld1q_f32(scales + base1);
            float32x4_t s1_hi = vld1q_f32(scales + base1 + 4);
            float32x4_t s2_lo = vld1q_f32(scales + base2);
            float32x4_t s2_hi = vld1q_f32(scales + base2 + 4);
            float32x4_t s3_lo = vld1q_f32(scales + base3);
            float32x4_t s3_hi = vld1q_f32(scales + base3 + 4);

            // Load zero points (second 8 of each 16-element group)
            float32x4_t z0_lo = vld1q_f32(scales + base0 + 8);
            float32x4_t z0_hi = vld1q_f32(scales + base0 + 12);
            float32x4_t z1_lo = vld1q_f32(scales + base1 + 8);
            float32x4_t z1_hi = vld1q_f32(scales + base1 + 12);
            float32x4_t z2_lo = vld1q_f32(scales + base2 + 8);
            float32x4_t z2_hi = vld1q_f32(scales + base2 + 12);
            float32x4_t z3_lo = vld1q_f32(scales + base3 + 8);
            float32x4_t z3_hi = vld1q_f32(scales + base3 + 12);

            // Load previous C values
            float32x4_t prev0 = vld1q_f32(c + i * 2);
            float32x4_t prev1 = vld1q_f32(c + i * 2 + 4);
            float32x4_t prev2 = vld1q_f32(c + i * 2 + 8);
            float32x4_t prev3 = vld1q_f32(c + i * 2 + 12);
            float32x4_t prev4 = vld1q_f32(c + i * 2 + 16);
            float32x4_t prev5 = vld1q_f32(c + i * 2 + 20);
            float32x4_t prev6 = vld1q_f32(c + i * 2 + 24);
            float32x4_t prev7 = vld1q_f32(c + i * 2 + 28);

            // result = prev + acc * scale + (zero * partial_sum if ib % Bits == 0)
            int ib0 = i / 4;
            int ib1 = i / 4 + 1;
            int ib2 = i / 4 + 2;
            int ib3 = i / 4 + 3;

            vec_c0 = vmlaq_f32(prev0, vec_c0, s0_lo);
            vec_c1 = vmlaq_f32(prev1, vec_c1, s0_hi);
            vec_c2 = vmlaq_f32(prev2, vec_c2, s1_lo);
            vec_c3 = vmlaq_f32(prev3, vec_c3, s1_hi);
            vec_c4 = vmlaq_f32(prev4, vec_c4, s2_lo);
            vec_c5 = vmlaq_f32(prev5, vec_c5, s2_hi);
            vec_c6 = vmlaq_f32(prev6, vec_c6, s3_lo);
            vec_c7 = vmlaq_f32(prev7, vec_c7, s3_hi);

            if ((ib0 % Bits) == 0) {
                vec_c0 = vmlaq_f32(vec_c0, z0_lo, vec_ps);
                vec_c1 = vmlaq_f32(vec_c1, z0_hi, vec_ps);
            }
            if ((ib1 % Bits) == 0) {
                vec_c2 = vmlaq_f32(vec_c2, z1_lo, vec_ps);
                vec_c3 = vmlaq_f32(vec_c3, z1_hi, vec_ps);
            }
            if ((ib2 % Bits) == 0) {
                vec_c4 = vmlaq_f32(vec_c4, z2_lo, vec_ps);
                vec_c5 = vmlaq_f32(vec_c5, z2_hi, vec_ps);
            }
            if ((ib3 % Bits) == 0) {
                vec_c6 = vmlaq_f32(vec_c6, z3_lo, vec_ps);
                vec_c7 = vmlaq_f32(vec_c7, z3_hi, vec_ps);
            }

            // Store results
            vst1q_f32(c + i * 2, vec_c0);
            vst1q_f32(c + i * 2 + 4, vec_c1);
            vst1q_f32(c + i * 2 + 8, vec_c2);
            vst1q_f32(c + i * 2 + 12, vec_c3);
            vst1q_f32(c + i * 2 + 16, vec_c4);
            vst1q_f32(c + i * 2 + 20, vec_c5);
            vst1q_f32(c + i * 2 + 24, vec_c6);
            vst1q_f32(c + i * 2 + 28, vec_c7);
        } else if (OneScale) {
            float32x4_t vec_s = vdupq_n_f32(scales[0]);

            vec_c0 = vaddq_f32(vld1q_f32(c + i * 2), vmulq_f32(vec_c0, vec_s));
            vec_c1 = vaddq_f32(vld1q_f32(c + i * 2 + 4), vmulq_f32(vec_c1, vec_s));
            vec_c2 = vaddq_f32(vld1q_f32(c + i * 2 + 8), vmulq_f32(vec_c2, vec_s));
            vec_c3 = vaddq_f32(vld1q_f32(c + i * 2 + 12), vmulq_f32(vec_c3, vec_s));
            vec_c4 = vaddq_f32(vld1q_f32(c + i * 2 + 16), vmulq_f32(vec_c4, vec_s));
            vec_c5 = vaddq_f32(vld1q_f32(c + i * 2 + 20), vmulq_f32(vec_c5, vec_s));
            vec_c6 = vaddq_f32(vld1q_f32(c + i * 2 + 24), vmulq_f32(vec_c6, vec_s));
            vec_c7 = vaddq_f32(vld1q_f32(c + i * 2 + 28), vmulq_f32(vec_c7, vec_s));

            vst1q_f32(c + i * 2, vec_c0);
            vst1q_f32(c + i * 2 + 4, vec_c1);
            vst1q_f32(c + i * 2 + 8, vec_c2);
            vst1q_f32(c + i * 2 + 12, vec_c3);
            vst1q_f32(c + i * 2 + 16, vec_c4);
            vst1q_f32(c + i * 2 + 20, vec_c5);
            vst1q_f32(c + i * 2 + 24, vec_c6);
            vst1q_f32(c + i * 2 + 28, vec_c7);
        } else {
            // Symmetric quantization without zero points
            int base0 = ((i / 4) / Bits) * 8;
            int base1 = ((i / 4 + 1) / Bits) * 8;
            int base2 = ((i / 4 + 2) / Bits) * 8;
            int base3 = ((i / 4 + 3) / Bits) * 8;

            float32x4_t s0_lo = vld1q_f32(scales + base0);
            float32x4_t s0_hi = vld1q_f32(scales + base0 + 4);
            float32x4_t s1_lo = vld1q_f32(scales + base1);
            float32x4_t s1_hi = vld1q_f32(scales + base1 + 4);
            float32x4_t s2_lo = vld1q_f32(scales + base2);
            float32x4_t s2_hi = vld1q_f32(scales + base2 + 4);
            float32x4_t s3_lo = vld1q_f32(scales + base3);
            float32x4_t s3_hi = vld1q_f32(scales + base3 + 4);

            vec_c0 = vmlaq_f32(vld1q_f32(c + i * 2), vec_c0, s0_lo);
            vec_c1 = vmlaq_f32(vld1q_f32(c + i * 2 + 4), vec_c1, s0_hi);
            vec_c2 = vmlaq_f32(vld1q_f32(c + i * 2 + 8), vec_c2, s1_lo);
            vec_c3 = vmlaq_f32(vld1q_f32(c + i * 2 + 12), vec_c3, s1_hi);
            vec_c4 = vmlaq_f32(vld1q_f32(c + i * 2 + 16), vec_c4, s2_lo);
            vec_c5 = vmlaq_f32(vld1q_f32(c + i * 2 + 20), vec_c5, s2_hi);
            vec_c6 = vmlaq_f32(vld1q_f32(c + i * 2 + 24), vec_c6, s3_lo);
            vec_c7 = vmlaq_f32(vld1q_f32(c + i * 2 + 28), vec_c7, s3_hi);

            vst1q_f32(c + i * 2, vec_c0);
            vst1q_f32(c + i * 2 + 4, vec_c1);
            vst1q_f32(c + i * 2 + 8, vec_c2);
            vst1q_f32(c + i * 2 + 12, vec_c3);
            vst1q_f32(c + i * 2 + 16, vec_c4);
            vst1q_f32(c + i * 2 + 20, vec_c5);
            vst1q_f32(c + i * 2 + 24, vec_c6);
            vst1q_f32(c + i * 2 + 28, vec_c7);
        }
    }
    
    return 0;
}

//
// LutGemmCompute_CompFp32 - Entry point for GEMM computation
//
void
LutGemmCompute_CompFp32(
    const uint8_t* A,
    const float* Scales,
    const int8_t* LUT,
    const float* LUT_Scales,
    const float* LUT_Biases,
    float* C,
    int K,
    int M,
    int N,
    int TotalN,
    size_t BlkLen,
    bool HasZeroPoint
)
{
    // Validate batch size (M)
    // For now, TMAC NEON kernel processes one batch row at a time.
    if (M != 1) {
        MLAS_THROW_EX(std::runtime_error, "M > 1 is not supported yet in TMAC NEON kernel");
    }

    // Get kernel config using the total output features (TotalN)
    // This matches the parameters used during weight packing.
    const MlasTMACKernelParams& tmac_params = MlasGetLutGemmKernelParams(TotalN, K, 2, BlkLen, HasZeroPoint);

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
    // m is the number of output features this kernel tile produces.
    // We clamp m by N (the number of features in the current chunk) to ensure
    // we don't read or write past the tile boundary during the gather phase.
    int32_t m_full = bm / bits;
    int32_t m = std::min(m_full, N);

    // Validate configuration
    assert(bm % bits == 0);
    assert(K % (kfactor * g) == 0);
    assert(BlkLen % g == 0);

    // Allocate buffers
    std::unique_ptr<float[]> CBits(new float[bm]);
    std::unique_ptr<float[]> C_global(new float[m]);

    // Explicitly zero-initialize accumulation buffers to ensure determinism.
    std::memset(CBits.get(), 0, bm * sizeof(float));
    std::memset(C_global.get(), 0, m * sizeof(float));

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
// LutGemmPackQuantBData_CompFp32 - Weight packing for NEON
// This is done during model load, so performance is less critical
//
void
LutGemmPackQuantBData_CompFp32(
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
// LutGemmPackScalesAndZeroPoints_CompFp32 - Scales and zero points packing
//
template <bool HasZeroPoint>
void
LutGemmPackScalesAndZeroPoints_CompFp32_Impl(
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

void
LutGemmPackScalesAndZeroPoints_CompFp32(
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
        LutGemmPackScalesAndZeroPoints_CompFp32_Impl<true>(
            N, K, bits, BlkLen, simd_n_out, bm,
            PackedScalesBegin, QuantBScale, QuantBZeroPoint, ThreadPool
        );
    } else {
        LutGemmPackScalesAndZeroPoints_CompFp32_Impl<false>(
            N, K, bits, BlkLen, simd_n_out, bm,
            PackedScalesBegin, QuantBScale, QuantBZeroPoint, ThreadPool
        );
    }
}

}  // namespace lutgemm_neon

//
// Kernel dispatch structure definition
//
const MLAS_QNBIT_LUT_GEMM_DISPATCH MlasLutGemmDispatchNeon = []() {
    MLAS_QNBIT_LUT_GEMM_DISPATCH d;
    d.GenerateLUT = lutgemm_neon::LutGemmGenerateLUT_CompFp32;
    d.ComputeGemm = lutgemm_neon::LutGemmCompute_CompFp32;
    d.PackQuantBData = lutgemm_neon::LutGemmPackQuantBData_CompFp32;
    d.PackScalesAndZeroPoints = lutgemm_neon::LutGemmPackScalesAndZeroPoints_CompFp32;
    return d;
}();

#endif  // MLAS_TARGET_ARM64
