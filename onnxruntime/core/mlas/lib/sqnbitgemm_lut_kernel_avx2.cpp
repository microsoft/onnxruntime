/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_lut_kernel_avx2.cpp

Abstract:

    This module implements x64 AVX2 kernel functions for LUT-based quantized
    n-bit integer matrix multiplication.

    It provides optimized AVX2 implementations for lookup table generation,
    GEMM computation, and related operations on quantized weight and activation
    matrices.

--*/

#include <cstddef>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>
// AVX2 intrinsics
#include <immintrin.h>

#include "qlutgemm.h"
#include "qnbitgemm.h"
#include "sqnbitgemm_q8_block.h"

static inline float
_mm256_addv_ps(const __m256 v)
{
    __m128 res = _mm256_extractf128_ps(v, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(v));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// Conditional pragma unroll for compiler compatibility
#if defined(__INTEL_COMPILER) || defined(__clang__)
#define PRAGMA_UNROLL _Pragma("unroll")
#else
#define PRAGMA_UNROLL
#endif

// Helper macros for extracting and widening vectors
#define extract_low_epi8_epi16(v) _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v))
#define extract_high_epi8_epi16(v) _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1))
#define extract_low_epi16_epi32(v) _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v))
#define extract_high_epi16_epi32(v) _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1))

// Template classes for accumulation
template <int N>
struct SignedHalvingAdder {
    SignedHalvingAdder<N / 2> adder;
    __m256i lhs = _mm256_setzero_si256();

    inline void push(__m256i v, int k)
    {
        if (k < N / 2) {
            adder.push(v, k);
            if (k == N / 2 - 1) {
                lhs = adder.get();
            }
        } else {
            adder.push(v, k - N / 2);
            if (k == N - 1) {
                lhs = _mm256_avg_epu8(lhs, adder.get());
            }
        }
    }

    inline __m256i get()
    {
        return lhs;
    }

    inline __m256i get_low()
    {
        return extract_low_epi8_epi16(lhs);
    }

    inline __m256i get_high()
    {
        return extract_high_epi8_epi16(lhs);
    }
};

template <>
struct SignedHalvingAdder<2> {
    __m256i lhs = _mm256_setzero_si256();

    inline void push(__m256i v, int k)
    {
        if (k == 0) {
            lhs = v;
        } else {
            lhs = _mm256_avg_epu8(lhs, v);
        }
    }

    inline __m256i get()
    {
        return lhs;
    }

    inline __m256i get_low()
    {
        return extract_low_epi8_epi16(lhs);
    }

    inline __m256i get_high()
    {
        return extract_high_epi8_epi16(lhs);
    }
};

template <int N>
struct SignedWideningAdder {
    __m256i lhs_low = _mm256_setzero_si256();
    __m256i lhs_high = _mm256_setzero_si256();

    inline void push(__m256i v, int k)
    {
        if (k == 0) {
            lhs_low = extract_low_epi8_epi16(v);
            lhs_high = extract_high_epi8_epi16(v);
        } else {
            lhs_low = _mm256_add_epi16(lhs_low, extract_low_epi8_epi16(v));
            lhs_high = _mm256_add_epi16(lhs_high, extract_high_epi8_epi16(v));
        }
    }

    inline __m256i get_low()
    {
        return lhs_low;
    }

    inline __m256i get_high()
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
    // if constexpr (bits == 4) {
    //     return 15;
    // } else if constexpr (bits == 3) {
    //     return 7;
    // } else if constexpr (bits == 2) {
    //     return 3;
    // } else if constexpr (bits == 1) {
    //     return 1;
    // } else {
    //     return 0;
    // }
    return 3;
}

void
partial_max_g4_int8_k8(float* lut_scales, const float* b)
{
    // TODO(vraspar): add support for arm neon
    const __m256i vec_bi = _mm256_set_epi32(112, 96, 80, 64, 48, 32, 16, 0);
    __m256 vec_b0 = _mm256_i32gather_ps(b + 0, vec_bi, 1);
    __m256 vec_b1 = _mm256_i32gather_ps(b + 1, vec_bi, 1);
    __m256 vec_b2 = _mm256_i32gather_ps(b + 2, vec_bi, 1);
    __m256 vec_b3 = _mm256_i32gather_ps(b + 3, vec_bi, 1);
    const __m256 vec_sign = _mm256_set1_ps(-0.0f);
    __m256 vec_babs0 = _mm256_andnot_ps(vec_sign, vec_b0);
    __m256 vec_babs1 = _mm256_andnot_ps(vec_sign, vec_b1);
    __m256 vec_babs2 = _mm256_andnot_ps(vec_sign, vec_b2);
    __m256 vec_babs3 = _mm256_andnot_ps(vec_sign, vec_b3);
    __m256 abssum = _mm256_add_ps(_mm256_add_ps(vec_babs0, vec_babs1), _mm256_add_ps(vec_babs2, vec_babs3));
    __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(abssum, 1), _mm256_castps256_ps128(abssum));
    max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
    float scales = _mm_cvtss_f32(max4) / 127;
    *lut_scales = std::max(*lut_scales, scales);
}

// Current implementation requires (K * 4) == act_group_size and K >= 8
// s0 = -1, s1 = 1
// TODO: loop K
inline void
lut_ctor_g4_int8_impl(
    int32_t act_k,
    int8_t* qlut,
    const float* b,
    float* lut_scales,
    float* lut_biases
)
{
    __m256 vec_lut[16];
    float biases = 0.0;
    const __m256i vec_bi = _mm256_set_epi32(112, 96, 80, 64, 48, 32, 16, 0);
    float scales = *lut_scales;
    float t_scales = scales ? 1.0f / scales : 0.0f;

    for (int k = 0; k < act_k / 32; ++k) {
        __m256 vec_b0 = _mm256_i32gather_ps(b + k * 32 + 0, vec_bi, 1);
        __m256 vec_b1 = _mm256_i32gather_ps(b + k * 32 + 1, vec_bi, 1);
        __m256 vec_b2 = _mm256_i32gather_ps(b + k * 32 + 2, vec_bi, 1);
        __m256 vec_b3 = _mm256_i32gather_ps(b + k * 32 + 3, vec_bi, 1);

        PRAGMA_UNROLL
        for (int g = 1; g < 16; g += 2) {
            vec_lut[g] = vec_b0;
            if (g & 0b0010) {
                vec_lut[g] = _mm256_add_ps(vec_lut[g], vec_b1);
            } else {
                vec_lut[g] = _mm256_sub_ps(vec_lut[g], vec_b1);
            }
            if (g & 0b0100) {
                vec_lut[g] = _mm256_add_ps(vec_lut[g], vec_b2);
            } else {
                vec_lut[g] = _mm256_sub_ps(vec_lut[g], vec_b2);
            }
            if (g & 0b1000) {
                vec_lut[g] = _mm256_add_ps(vec_lut[g], vec_b3);
            } else {
                vec_lut[g] = _mm256_sub_ps(vec_lut[g], vec_b3);
            }
        }
        PRAGMA_UNROLL
        for (int g = 0; g < 16; g += 2) {
            // vec_lut[g] = -vec_lut[15 - g];
            const __m256 neg_mask = _mm256_set1_ps(-0.0f);  // all lanes have sign bit set
            vec_lut[g] = _mm256_xor_ps(vec_lut[15 - g], neg_mask);
        }

        biases += _mm256_addv_ps(vec_lut[0]);

        PRAGMA_UNROLL
        for (int g = 0; g < 16; ++g) {
            vec_lut[g] = _mm256_mul_ps(vec_lut[g], _mm256_set1_ps(t_scales));
        }

        __m256i vec_qlut[4];
        const __m256i shuf = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
        PRAGMA_UNROLL
        for (int g = 0; g < 4; g += 1) {
            __m256i i0 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 0], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 1], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i2 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 2], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i3 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 3], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

            i0 = _mm256_packs_epi32(i0, i1);              // 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
            i2 = _mm256_packs_epi32(i2, i3);              // 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                                          // Convert int16 to int8
            i0 = _mm256_packs_epi16(i0, i2);              // 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7,  12, 13, 14, 15,  20, 21, 22, 23,  28, 29, 30, 31
            vec_qlut[g] = _mm256_shuffle_epi8(i0, shuf);  // 0, 8, 16, 24,  1, 9, 17, 25,  2, 10, 18, 26,  3, 11, 19, 27,  4, 12, 20, 28,  5, 13, 21, 29,  6, 14, 22, 30,  7, 15, 23, 31
        }

        int32_t* qlut_i32 = reinterpret_cast<int32_t*>(qlut);
        PRAGMA_UNROLL
        for (int g = 0; g < 4; ++g) {
            qlut_i32[k * 32 + 0 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 0);
        }
        PRAGMA_UNROLL
        for (int g = 0; g < 4; ++g) {
            qlut_i32[k * 32 + 1 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 1);
        }
        PRAGMA_UNROLL
        for (int g = 0; g < 4; ++g) {
            qlut_i32[k * 32 + 2 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 2);
        }
        PRAGMA_UNROLL
        for (int g = 0; g < 4; ++g) {
            qlut_i32[k * 32 + 3 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 3);
        }
        PRAGMA_UNROLL
        for (int g = 0; g < 4; ++g) {
            qlut_i32[k * 32 + 4 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 4);
        }
        PRAGMA_UNROLL
        for (int g = 0; g < 4; ++g) {
            qlut_i32[k * 32 + 5 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 5);
        }
        PRAGMA_UNROLL
        for (int g = 0; g < 4; ++g) {
            qlut_i32[k * 32 + 6 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 6);
        }
        PRAGMA_UNROLL
        for (int g = 0; g < 4; ++g) {
            qlut_i32[k * 32 + 7 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 7);
        }
    }

    *lut_scales = scales;
    *lut_biases = biases;
}

// based on lut_ctor_g4_int8_impl
void
GenerateLUT_avx2(
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
    // TODO: handle bitnet here
    const int32_t kk_outer_max = static_cast<int32_t>(K / act_group_size);
    const int32_t ags_div32 = static_cast<int32_t>(act_group_size / 32);

    for (int32_t kk_outer = 0; kk_outer < kk_outer_max; ++kk_outer) {
        // compute partial max - directly reset scale to 0.0
        lut_scales[kk_outer] = 0.0f;  // partial max reset
        for (int32_t k_outer = 0; k_outer < ags_div32; ++k_outer) {
            partial_max_g4_int8_k8(&lut_scales[kk_outer], &b[(kk_outer * act_group_size) + (k_outer * 32)]);
        }
    }

    for (int32_t k_outer_1 = 0; k_outer_1 < kk_outer_max; ++k_outer_1) {
        lut_ctor_g4_int8_impl(static_cast<int32_t>(act_group_size), (&(qlut[(k_outer_1 * act_group_size * 4)])), (&(b[(k_outer_1 * act_group_size)])), (&(lut_scales[k_outer_1])), (&(lut_biases[k_outer_1])));
    }
}

inline void
tbl_g4_int8_float_gather_bit2_impl(int32_t m, float* C_global, float* CBits, float* C)
{
    constexpr int32_t bits = 2;

    int32_t m_c_outer_max = m / 32;
    for (int32_t m_c_outer = 0; m_c_outer < m_c_outer_max; ++m_c_outer) {
        int32_t cse_var_2 = (m_c_outer * 32 * bits);
        int32_t cse_var_1 = (m_c_outer * 32);
        PRAGMA_UNROLL
        for (int32_t m_c_inner = 0; m_c_inner < 32; ++m_c_inner) {
            int32_t bit_offset_0 = (m_c_inner / 8) * 8 * bits + (m_c_inner % 8);
            int32_t bit_offset_1 = (m_c_inner / 8) * 8 * bits + (m_c_inner % 8) + 8;
            C_global[cse_var_1 + m_c_inner] = (CBits[cse_var_2 + bit_offset_0] * (float)5.000000e-01f) + (CBits[cse_var_2 + bit_offset_1]);
        }
    }

    for (int32_t m_inner_outer = 0; m_inner_outer < m_c_outer_max; ++m_inner_outer) {
        PRAGMA_UNROLL
        for (int32_t m_inner = 0; m_inner < 32; ++m_inner) {
            int offset = m_inner_outer * 32 + m_inner;
            C[offset] = C_global[offset];
        }
    }
}

// When FastAggregation is enabled, FastAggregationK = ActK
// zero_points is merged into scales to maintain API
template <bool has_scale, int K, int Bits, int ActK, bool FastAggregation, bool ZeroPoint, bool OneScale>
inline int32_t
tbl_g4_int8_float_update_impl(int32_t m, float* c, const int8_t* lut, const uint8_t* a, const float* scales, const float* lut_scales, const float* lut_biases)
{
    const __m128i vec_mask = _mm_set1_epi8(0x0f);
    __m128i vec_lut[K];

    PRAGMA_UNROLL
    for (int k = 0; k < K; k++) {
        vec_lut[k] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut + k * 16));
    }

    SignedAdder<FastAggregation, ActK> adder;
    for (int i = 0; i < m / 2; i += 16) {
        __m256 vec_c0 = _mm256_setzero_ps();
        __m256 vec_c1 = _mm256_setzero_ps();
        __m256 vec_c2 = _mm256_setzero_ps();
        __m256 vec_c3 = _mm256_setzero_ps();

        float partial_sum = -0.0f;
        PRAGMA_UNROLL
        for (int kk = 0; kk < K; kk += ActK) {
            PRAGMA_UNROLL
            for (int k = 0; k < ActK; k++) {
                // (M // bm, KK / K / 4, bm / 16 / 2, K * 16)
                __m128i vec_as = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i * K + (kk + k) * 16));
                __m128i vec_a_bot = _mm_and_si128(vec_as, vec_mask);
                __m128i vec_a_top = _mm_and_si128(_mm_srli_epi16(vec_as, 4), vec_mask);

                __m256i vec_lut_ = _mm256_set_m128i(vec_lut[kk + k], vec_lut[kk + k]);
                __m256i vec_a = _mm256_set_m128i(vec_a_top, vec_a_bot);
                __m256i vec_v = _mm256_shuffle_epi8(vec_lut_, vec_a);
                adder.push(vec_v, k);
            }

            __m256 vec_v_low_low = _mm256_cvtepi32_ps(extract_low_epi16_epi32(adder.get_low()));
            __m256 vec_v_low_high = _mm256_cvtepi32_ps(extract_high_epi16_epi32(adder.get_low()));
            __m256 vec_v_high_low = _mm256_cvtepi32_ps(extract_low_epi16_epi32(adder.get_high()));
            __m256 vec_v_high_high = _mm256_cvtepi32_ps(extract_high_epi16_epi32(adder.get_high()));

            float lut_s = lut_scales[kk / ActK];
            float lut_b = lut_biases[kk / ActK];

            partial_sum += lut_b;

            if (FastAggregation) {
                lut_s = lut_s * ActK;
                lut_b -= lut_s * (mylog2<ActK>::value / 4 * get_bias_scale<Bits>());
            }

#define lut_fma(vs, ib)                                          \
    ((ib) % Bits) ? (_mm256_mul_ps((vs), _mm256_set1_ps(lut_s))) \
                  : (_mm256_fmadd_ps((vs), _mm256_set1_ps(lut_s), _mm256_set1_ps(lut_b)))
            if (kk == 0) {
                vec_c0 = lut_fma(vec_v_low_low, (i / 4));
                vec_c1 = lut_fma(vec_v_low_high, (i / 4 + 1));
                vec_c2 = lut_fma(vec_v_high_low, (i / 4 + 2));
                vec_c3 = lut_fma(vec_v_high_high, (i / 4 + 3));
            } else {
                vec_c0 = _mm256_add_ps(vec_c0, lut_fma(vec_v_low_low, (i / 4)));
                vec_c1 = _mm256_add_ps(vec_c1, lut_fma(vec_v_low_high, (i / 4 + 1)));
                vec_c2 = _mm256_add_ps(vec_c2, lut_fma(vec_v_high_low, (i / 4 + 2)));
                vec_c3 = _mm256_add_ps(vec_c3, lut_fma(vec_v_high_high, (i / 4 + 3)));
            }
#undef lut_fma
        }

        if (ZeroPoint) {
            __m256 vec_s0 = _mm256_loadu_ps(scales + ((i / 4) / Bits) * 16);
            __m256 vec_s1 = _mm256_loadu_ps(scales + ((i / 4 + 1) / Bits) * 16);
            __m256 vec_s2 = _mm256_loadu_ps(scales + ((i / 4 + 2) / Bits) * 16);
            __m256 vec_s3 = _mm256_loadu_ps(scales + ((i / 4 + 3) / Bits) * 16);
            vec_c0 = _mm256_fmadd_ps(vec_c0, vec_s0, _mm256_loadu_ps(c + i * 2));
            vec_c1 = _mm256_fmadd_ps(vec_c1, vec_s1, _mm256_loadu_ps(c + i * 2 + 8));
            vec_c2 = _mm256_fmadd_ps(vec_c2, vec_s2, _mm256_loadu_ps(c + i * 2 + 16));
            vec_c3 = _mm256_fmadd_ps(vec_c3, vec_s3, _mm256_loadu_ps(c + i * 2 + 24));
            __m256 vec_z0 = _mm256_loadu_ps(scales + ((i / 4) / Bits) * 16 + 8);
            __m256 vec_z1 = _mm256_loadu_ps(scales + ((i / 4 + 1) / Bits) * 16 + 8);
            __m256 vec_z2 = _mm256_loadu_ps(scales + ((i / 4 + 2) / Bits) * 16 + 8);
            __m256 vec_z3 = _mm256_loadu_ps(scales + ((i / 4 + 3) / Bits) * 16 + 8);
            partial_sum *= 2;
#define add_zero(cs, zs, ib) \
    ((ib) % Bits) ? ((cs))   \
                  : (_mm256_fmadd_ps((zs), _mm256_set1_ps(partial_sum), (cs)))
            _mm256_storeu_ps(c + i * 2, add_zero(vec_c0, vec_z0, (i / 4)));
            _mm256_storeu_ps(c + i * 2 + 8, add_zero(vec_c1, vec_z1, (i / 4 + 1)));
            _mm256_storeu_ps(c + i * 2 + 16, add_zero(vec_c2, vec_z2, (i / 4 + 2)));
            _mm256_storeu_ps(c + i * 2 + 24, add_zero(vec_c3, vec_z3, (i / 4 + 3)));
#undef add_zero
        } else if (OneScale) {
            float single_scale = scales[0];
            __m256 vec_s = _mm256_set1_ps(single_scale);
            _mm256_storeu_ps(c + i * 2, _mm256_fmadd_ps(vec_c0, vec_s, _mm256_loadu_ps(c + i * 2)));
            _mm256_storeu_ps(c + i * 2 + 8, _mm256_fmadd_ps(vec_c1, vec_s, _mm256_loadu_ps(c + i * 2 + 8)));
            _mm256_storeu_ps(c + i * 2 + 16, _mm256_fmadd_ps(vec_c2, vec_s, _mm256_loadu_ps(c + i * 2 + 16)));
            _mm256_storeu_ps(c + i * 2 + 24, _mm256_fmadd_ps(vec_c3, vec_s, _mm256_loadu_ps(c + i * 2 + 24)));
        } else {
            __m256 vec_s0 = _mm256_loadu_ps(scales + ((i / 4) / Bits) * 8);
            __m256 vec_s1 = _mm256_loadu_ps(scales + ((i / 4 + 1) / Bits) * 8);
            __m256 vec_s2 = _mm256_loadu_ps(scales + ((i / 4 + 2) / Bits) * 8);
            __m256 vec_s3 = _mm256_loadu_ps(scales + ((i / 4 + 3) / Bits) * 8);
            _mm256_storeu_ps(c + i * 2, _mm256_fmadd_ps(vec_c0, vec_s0, _mm256_loadu_ps(c + i * 2)));
            _mm256_storeu_ps(c + i * 2 + 8, _mm256_fmadd_ps(vec_c1, vec_s1, _mm256_loadu_ps(c + i * 2 + 8)));
            _mm256_storeu_ps(c + i * 2 + 16, _mm256_fmadd_ps(vec_c2, vec_s2, _mm256_loadu_ps(c + i * 2 + 16)));
            _mm256_storeu_ps(c + i * 2 + 24, _mm256_fmadd_ps(vec_c3, vec_s3, _mm256_loadu_ps(c + i * 2 + 24)));
        }
    }

    return 0;
}

int32_t
tbl_int32_reset(int32_t m, int32_t* c)
{
    memset(c, 0, m * sizeof(int32_t));
    return 0;
}

// based on qgemm_lut_int8_g4
// Simplified version with hardcoded configuration for 2-bit quantization
void
TMACComputeGemm_avx2(
    const uint8_t* A,         // Quantized packed weights
    const float* Scales,      // Weight scales (and optionally zero-points)
    const int8_t* LUT,        // Pre-computed quantized lookup table
    const float* LUT_Scales,  // LUT scales from activation quantization
    const float* LUT_Biases,  // LUT biases from activation quantization
    float* C,                 // Output buffer
    int K,
    int M,
    int N,
    size_t BlkLen,  // Weight quantization group size (q_group_size)
    bool HasZeroPoint
)
{
    // Validate batch size
    if (N != 1) {
        MLAS_THROW_EX(std::runtime_error, "N > 1 is not supported yet");
    }

    // get kernel config
    const MlasTMACKernelParams& tmac_params = MlasGetLutGemmKernelParams(M, K, 2, BlkLen, HasZeroPoint);

    // ==================== CONFIGURATION ====================
    // Fixed parameters for this kernel implementation
    bool has_zero_point = tmac_params.has_zero_point;  // Whether weights have zero-points (interleaved with scales)
    bool one_scale = tmac_params.one_scale;            // Whether using single global scale for all weights

    const int32_t bits = static_cast<int32_t>(tmac_params.bits);                          // 2-bit quantization
    const int32_t g = static_cast<int32_t>(tmac_params.g);                                // Packing group size
    const int32_t ngroups_per_elem = static_cast<int32_t>(tmac_params.ngroups_per_elem);  // 8 / g = 2
    const int32_t kfactor = static_cast<int32_t>(tmac_params.kfactor);                    // K-dimension blocking factor

    const bool has_scale = tmac_params.has_scale;  // Always use weight scales

    // Parameters derived from inputs
    const int32_t q_group_size = static_cast<int32_t>(tmac_params.q_group_size);      // Weight quant group size
    const int32_t act_group_size = static_cast<int32_t>(tmac_params.act_group_size);  // Activation group size (same as weight)
    const int32_t actk = static_cast<int32_t>(tmac_params.actk);                      // CRITICAL: = 16 for BlkLen=64, NOT BlkLen!

    const int32_t bm = static_cast<int32_t>(tmac_params.bm);
    int32_t m = bm / bits;

    // Validate configuration
    assert(bm % bits == 0);
    assert(K % (kfactor * g) == 0);
    assert(BlkLen % g == 0);

    // Validate configuration
    assert(bm % bits == 0);
    assert(K % (kfactor * g) == 0);
    assert(BlkLen % g == 0);

    // ==================== ALLOCATE BUFFERS ====================
    // Use float for now (can be changed to _Float16 if needed)

    float* CBits = new float[bm];
    float* C_global = new float[m];

    // Reset accumulator buffer to zero
    tbl_int32_reset(bm * sizeof(float) / sizeof(int32_t), reinterpret_cast<int32_t*>(CBits));

    // ==================== CALCULATE LOOP PARAMETERS ====================
    const int32_t k_outer_max = K / (kfactor * g);
    const int32_t scale_gs = q_group_size / (kfactor * g);

    // Calculate bit shift for scale indexing
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
        MLAS_THROW_EX(std::runtime_error,
                      ("Unsupported scale_gs=" + std::to_string(scale_gs) +
                       " (q_group_size=" + std::to_string(q_group_size) +
                       ", kfactor=" + std::to_string(kfactor) +
                       ", g=" + std::to_string(g) + "). Expected {1,2,4,8}.").c_str());
    }

    // ==================== MAIN COMPUTATION LOOP ====================
    for (int32_t k_outer = 0; k_outer < k_outer_max; k_outer++) {
        // Calculate pointers for this K-outer iteration
        const uint8_t* a = A + k_outer * bm * kfactor / ngroups_per_elem;

        // Calculate scales pointer based on configuration
        const float* scales = one_scale ? reinterpret_cast<const float*>(Scales) :                                                  // Single global scale
                                  (has_zero_point ? reinterpret_cast<const float*>(Scales) + (k_outer >> scale_idx_shfr) * m * 2 :  // Scale + zero_point pairs
                                       reinterpret_cast<const float*>(Scales) + (k_outer >> scale_idx_shfr) * m);                   // Scales only

        // Calculate LUT pointers
        const int8_t* lut = reinterpret_cast<const int8_t*>(LUT) + k_outer * kfactor * (1 << g);  // 2^g = 16 for g=4
        const float* lut_scales = reinterpret_cast<const float*>(LUT_Scales) +
                                  (k_outer * kfactor * g / act_group_size);
        const float* lut_biases = reinterpret_cast<const float*>(LUT_Biases) +
                                  (k_outer * kfactor * g / act_group_size);

        // Select appropriate kernel template based on configuration
        // For standard 2-bit, kfactor=16, BlkLen=64: actk = 64/4 = 16
        if (has_scale && kfactor == 16 && bits == 2 && actk == 16 && has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 2, 16, false, true, false>(
                static_cast<int32_t>(bm), CBits, lut, a, scales, lut_scales, lut_biases
            );
        } else if (has_scale && kfactor == 16 && bits == 2 && actk == 16 && !has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 2, 16, false, false, false>(
                static_cast<int32_t>(bm), CBits, lut, a, scales, lut_scales, lut_biases
            );
        } else if (has_scale && kfactor == 16 && bits == 2 && actk == 16 && !has_zero_point && one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 2, 16, false, false, true>(
                static_cast<int32_t>(bm), CBits, lut, a, scales, lut_scales, lut_biases
            );
        }
        // actk == 8 variants (for BlkLen=32)
        else if (has_scale && kfactor == 16 && bits == 2 && actk == 8 && has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 2, 8, false, true, false>(
                static_cast<int32_t>(bm), CBits, lut, a, scales, lut_scales, lut_biases
            );
        } else if (has_scale && kfactor == 16 && bits == 2 && actk == 8 && !has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 2, 8, false, false, false>(
                static_cast<int32_t>(bm), CBits, lut, a, scales, lut_scales, lut_biases
            );
        } else if (has_scale && kfactor == 16 && bits == 2 && actk == 8 && !has_zero_point && one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 2, 8, false, false, true>(
                static_cast<int32_t>(bm), CBits, lut, a, scales, lut_scales, lut_biases
            );
        }
        // kfactor == 8 variants
        else if (has_scale && kfactor == 8 && bits == 2 && actk == 8 && has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 8, 2, 8, false, true, false>(
                static_cast<int32_t>(bm), CBits, lut, a, scales, lut_scales, lut_biases
            );
        } else if (has_scale && kfactor == 8 && bits == 2 && actk == 8 && !has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 8, 2, 8, false, false, false>(
                static_cast<int32_t>(bm), CBits, lut, a, scales, lut_scales, lut_biases
            );
        } else if (has_scale && kfactor == 8 && bits == 2 && actk == 8 && !has_zero_point && one_scale) {
            tbl_g4_int8_float_update_impl<true, 8, 2, 8, false, false, true>(
                static_cast<int32_t>(bm), CBits, lut, a, scales, lut_scales, lut_biases
            );
        } else {
            // No matching kernel template found
            MLAS_THROW_EX(std::runtime_error, "No matching kernel found for T-MAC GEMM");
        }
    }

    // ==================== GATHER RESULTS ====================
    // Gather bit-plane results into final output
    // Only support 2-bit in this implementation
    // TODO(vraspar): extend to other bit-widths
    tbl_g4_int8_float_gather_bit2_impl(m, C_global, CBits, C);

    // ==================== CLEANUP ====================
    delete[] C_global;
    delete[] CBits;
}

//
// AVX2 optimized weight packing for T-MAC LUT GEMM
// This performs the same transformation as the scalar version but uses SIMD operations
//
void
PackQuantBData_avx2(
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

    // Phase 1: Bit-plane decomposition with grouping by g=4
    // For 2-bit: each input byte has 4 elements, we extract bit planes and group 4 consecutive bits
    std::unique_ptr<uint8_t[]> buf(new uint8_t[N * bits * K_div_g]);

    // Masks for 2-bit extraction
    const __m256i mask_2bit = _mm256_set1_epi8(0x03);  // mask for 2-bit values
    const __m256i mask_bit0 = _mm256_set1_epi8(0x01);  // bit 0 of each 2-bit element

    // Phase 1: Parallelize over N (each thread processes one row)
    MlasTrySimpleParallel(
        ThreadPool, static_cast<ptrdiff_t>(N),
        [&](ptrdiff_t tid) {
            size_t im = static_cast<size_t>(tid);

            // Process in chunks of 32 bytes (128 2-bit elements = 32 groups of 4)
            const uint8_t* src_row = reinterpret_cast<const uint8_t*>(QuantBDataBegin) + (im * K / 4);
            uint8_t* dst_bit0 = buf.get() + im * bits * K_div_g;
            uint8_t* dst_bit1 = dst_bit0 + K_div_g;

            size_t ik = 0;
            // Process 128 elements at a time (32 bytes input = 128 2-bit elements = 32 output bytes per bit plane)
            for (; ik + 128 <= K; ik += 128) {
                // Load 32 bytes = 128 2-bit elements
                __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_row + ik / 4));

                // Extract each of 4 positions within each byte
                // pos0: bits 0-1, pos1: bits 2-3, pos2: bits 4-5, pos3: bits 6-7
                __m256i pos0 = _mm256_and_si256(packed, mask_2bit);
                __m256i pos1 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask_2bit);
                __m256i pos2 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_2bit);
                __m256i pos3 = _mm256_srli_epi16(packed, 6);

                // For g=4: we need to group 4 consecutive elements
                // Each output byte contains bits from 4 consecutive input elements, shifted by their position
                // Output for bit0: (pos0_bit0 << 0) | (pos1_bit0 << 1) | (pos2_bit0 << 2) | (pos3_bit0 << 3)

                // Extract bit 0 from each position
                __m256i b0_pos0 = _mm256_and_si256(pos0, mask_bit0);              // bit0 at position 0
                __m256i b0_pos1 = _mm256_and_si256(pos1, mask_bit0);              // bit0 at position 0
                __m256i b0_pos2 = _mm256_and_si256(pos2, mask_bit0);              // bit0 at position 0
                __m256i b0_pos3 = _mm256_and_si256(pos3, mask_bit0);              // bit0 at position 0

                // Combine: shift each position's bit to its final location
                __m256i bit0_out = _mm256_or_si256(
                    _mm256_or_si256(b0_pos0, _mm256_slli_epi16(b0_pos1, 1)),
                    _mm256_or_si256(_mm256_slli_epi16(b0_pos2, 2), _mm256_slli_epi16(b0_pos3, 3))
                );

                // Extract bit 1 from each position and shift down first
                __m256i b1_pos0 = _mm256_and_si256(_mm256_srli_epi16(pos0, 1), mask_bit0);
                __m256i b1_pos1 = _mm256_and_si256(_mm256_srli_epi16(pos1, 1), mask_bit0);
                __m256i b1_pos2 = _mm256_and_si256(_mm256_srli_epi16(pos2, 1), mask_bit0);
                __m256i b1_pos3 = _mm256_and_si256(_mm256_srli_epi16(pos3, 1), mask_bit0);

                // Combine for bit 1 plane
                __m256i bit1_out = _mm256_or_si256(
                    _mm256_or_si256(b1_pos0, _mm256_slli_epi16(b1_pos1, 1)),
                    _mm256_or_si256(_mm256_slli_epi16(b1_pos2, 2), _mm256_slli_epi16(b1_pos3, 3))
                );

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_bit0 + ik / g), bit0_out);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_bit1 + ik / g), bit1_out);
            }

            // Handle remaining elements with scalar code
            for (; ik < K; ++ik) {
                size_t idx = ik;
                size_t num_elem_per_byte = 4;  // 2-bit: 4 elements per byte
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
    // Optimize parallelization: use 2D partitioning over N and K/g for better load balancing
    const size_t c0_fac2 = K_div_g;
    const size_t c0_fac1 = simd_n_out * c0_fac2;
    const size_t c0_fac0 = bits * c0_fac1;

    const size_t c1_nb2 = K_div_g;
    const size_t c1_nb1 = simd_n_in * c1_nb2;
    const size_t c1_nb0 = ngroups_per_elem * c1_nb1;
    const size_t c1_fac2 = K_div_g;
    const size_t c1_fac1 = ngroups_per_elem * c1_fac2;
    const size_t c1_fac0 = simd_n_in * c1_fac1;

    const size_t c2_nb4 = kfactor;
    const size_t c2_nb3 = K_div_g / kfactor * c2_nb4;
    const size_t c2_nb2 = ngroups_per_elem * c2_nb3;
    const size_t c2_nb1 = simd_n_in * c2_nb2;
    const size_t c2_nb0 = bm / mgroup * c2_nb1;
    const size_t c2_fac3 = simd_n_in * ngroups_per_elem;
    const size_t c2_fac2 = kfactor * c2_fac3;
    const size_t c2_fac1 = bm / mgroup * c2_fac2;
    const size_t c2_fac0 = K_div_g / kfactor * c2_fac1;

    const size_t PackedQuantBDataSize = (N * bits) * (K_div_g / ngroups_per_elem);
    memset(PackedQuantBDataBegin, 0, PackedQuantBDataSize);

    // Phase 2: Parallelize over N - each thread handles all (bits, K/g) work for its assigned rows
    // This ensures no write conflicts since each im writes to disjoint output regions
    MlasTrySimpleParallel(
        ThreadPool, static_cast<ptrdiff_t>(N),
        [&](ptrdiff_t tid) {
            size_t im = static_cast<size_t>(tid);
            for (size_t ib = 0; ib < bits; ib++) {
                for (size_t ik = 0; ik < K_div_g; ik++) {
                    // w = w.reshape(M // bits // simd_n_out, simd_n_out, bits, K // g).transpose(0, 2, 1, 3)
                    size_t new_im = im / simd_n_out;
                    size_t new_isno = im % simd_n_out;
                    size_t new_idx = new_im * c0_fac0 + ib * c0_fac1 + new_isno * c0_fac2 + ik;

                    // w = w.reshape(M // mgroup, ngroups_per_elem, simd_n_in, K // g).transpose(0, 2, 1, 3)
                    new_im = new_idx / c1_nb0;
                    size_t new_ing = (new_idx % c1_nb0) / c1_nb1;
                    size_t new_isni = (new_idx % c1_nb1) / c1_nb2;
                    size_t new_ik = (new_idx % c1_nb2);
                    new_idx = new_im * c1_fac0 + new_isni * c1_fac1 + new_ing * c1_fac2 + new_ik;

                    // w = w.reshape(M // bm, bm // mgroup, simd_n_in, ngroups_per_elem, K // g // kfactor, kfactor).transpose(0, 4, 1, 5, 2, 3)
                    new_im = new_idx / c2_nb0;
                    size_t new_ibm = (new_idx % c2_nb0) / c2_nb1;
                    new_isni = (new_idx % c2_nb1) / c2_nb2;
                    new_ing = (new_idx % c2_nb2) / c2_nb3;
                    new_ik = (new_idx % c2_nb3) / c2_nb4;
                    size_t new_ikf = (new_idx % c2_nb4);
                    new_idx = new_im * c2_fac0 +
                              new_ik * c2_fac1 +
                              new_ibm * c2_fac2 +
                              new_ikf * c2_fac3 +
                              new_isni * ngroups_per_elem +
                              new_ing;
                    new_idx = new_idx / ngroups_per_elem;
                    size_t buf_idx = im * bits * K_div_g + ib * K_div_g + ik;
                    uint8_t buf_val = buf[buf_idx];

                    // w = sum([(w[:, :, :, :, :, ng] << (ng * g)) for ng in range(ngroups_per_elem)])
                    PackedQuantBDataBegin[new_idx] = static_cast<std::byte>(
                        static_cast<unsigned>(PackedQuantBDataBegin[new_idx]) +
                        (buf_val << (new_ing * g))
                    );
                }
            }
        }
    );
}

//
// AVX2 optimized scales and zero points packing for T-MAC LUT GEMM
// This performs the same transformation as the scalar version but uses SIMD operations
//
void
PackScalesAndZeroPoints_avx2(
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
    // Only optimized for 2-bit quantization
    assert(bits == 2);
    
    const size_t num_elem_per_byte = 8 / bits;  // 4 for 2-bit
    const size_t row_blks = K / BlkLen;         // number of blocks per column
    const size_t zp_bytes_per_col = (row_blks + num_elem_per_byte - 1) / num_elem_per_byte;
    
    // Precompute constants
    const size_t nb1 = K / BlkLen;
    const size_t nb0 = bm / bits * nb1;
    const int midpoint = 1 << (bits - 1);  // 2 for 2-bit
    const uint8_t bits_mask = static_cast<uint8_t>((1 << bits) - 1);
    
    // Calculate optimal parallelization
    // Total work: N * (K / BlkLen) items
    // Each thread should have enough work to amortize overhead
    const size_t TotalBlocks = N * row_blks;
    ptrdiff_t MaxThreads = MlasGetMaximumThreadCount(ThreadPool);
    
    // Use 2D parallelization over (N, K/BlkLen) for better load balancing
    // when N is smaller than available threads
    if (N >= static_cast<size_t>(MaxThreads) || row_blks <= 1) {
        // Parallelize over N only (original approach - good cache locality per row)
        MlasTrySimpleParallel(
            ThreadPool, static_cast<ptrdiff_t>(N),
            [&](ptrdiff_t tid) {
                size_t im = static_cast<size_t>(tid);
                for (size_t blk_in_col = 0; blk_in_col < row_blks; blk_in_col++) {
                    size_t idx = im * nb1 + blk_in_col;
                    float scale = QuantBScale[idx];
                    float zp = 0.0f;
                    if (HasZeroPoint) {
                        size_t zp_byte_idx = im * zp_bytes_per_col + blk_in_col / num_elem_per_byte;
                        size_t elem_idx = blk_in_col % num_elem_per_byte;
                        uint8_t v = (QuantBZeroPoint[zp_byte_idx] >> (elem_idx * bits)) & bits_mask;
                        zp = static_cast<float>(static_cast<int>(v) - midpoint) * scale;
                    }

                    size_t new_im, new_ibm, new_ik;
                    if (nb1 == 0) {
                        new_im = 0;
                        new_ibm = 0;
                        new_ik = 0;
                    } else {
                        new_im = idx / nb0;
                        new_ibm = (idx % nb0) / nb1;
                        new_ik = (idx % nb1);
                    }

                    if (HasZeroPoint) {
                        size_t new_isimd = new_ibm % simd_n_out;
                        size_t new_idx_outer = new_im * bm / bits * nb1 / simd_n_out + 
                                               new_ik * bm / bits / simd_n_out + 
                                               new_ibm / simd_n_out;
                        size_t new_idx_scale = new_idx_outer * (simd_n_out * 2) + new_isimd;
                        size_t new_idx_zero = new_idx_outer * (simd_n_out * 2) + simd_n_out + new_isimd;

                        PackedScalesBegin[new_idx_scale] = scale;
                        PackedScalesBegin[new_idx_zero] = zp;
                    } else {
                        size_t new_idx = new_im * bm / bits * nb1 + new_ik * bm / bits + new_ibm;
                        PackedScalesBegin[new_idx] = scale;
                    }
                }
            }
        );
    } else {
        // 2D parallelization over N * row_blks for better thread utilization
        MlasTrySimpleParallel(
            ThreadPool, static_cast<ptrdiff_t>(TotalBlocks),
            [&](ptrdiff_t tid) {
                size_t block_idx = static_cast<size_t>(tid);
                size_t im = block_idx / row_blks;
                size_t blk_in_col = block_idx % row_blks;
                
                size_t idx = im * nb1 + blk_in_col;
                float scale = QuantBScale[idx];
                float zp = 0.0f;
                if (HasZeroPoint) {
                    size_t zp_byte_idx = im * zp_bytes_per_col + blk_in_col / num_elem_per_byte;
                    size_t elem_idx = blk_in_col % num_elem_per_byte;
                    uint8_t v = (QuantBZeroPoint[zp_byte_idx] >> (elem_idx * bits)) & bits_mask;
                    zp = static_cast<float>(static_cast<int>(v) - midpoint) * scale;
                }

                size_t new_im, new_ibm, new_ik;
                if (nb1 == 0) {
                    new_im = 0;
                    new_ibm = 0;
                    new_ik = 0;
                } else {
                    new_im = idx / nb0;
                    new_ibm = (idx % nb0) / nb1;
                    new_ik = (idx % nb1);
                }

                if (HasZeroPoint) {
                    size_t new_isimd = new_ibm % simd_n_out;
                    size_t new_idx_outer = new_im * bm / bits * nb1 / simd_n_out + 
                                           new_ik * bm / bits / simd_n_out + 
                                           new_ibm / simd_n_out;
                    size_t new_idx_scale = new_idx_outer * (simd_n_out * 2) + new_isimd;
                    size_t new_idx_zero = new_idx_outer * (simd_n_out * 2) + simd_n_out + new_isimd;

                    PackedScalesBegin[new_idx_scale] = scale;
                    PackedScalesBegin[new_idx_zero] = zp;
                } else {
                    size_t new_idx = new_im * bm / bits * nb1 + new_ik * bm / bits + new_ibm;
                    PackedScalesBegin[new_idx] = scale;
                }
            }
        );
    }
}

// Kernel dispatch structure definition.

const MLAS_QNBIT_LUT_GEMM_DISPATCH MlasLutGenKernelAvx2 = []() {
    MLAS_QNBIT_LUT_GEMM_DISPATCH d;
    d.GenerateLUT = GenerateLUT_avx2;
    d.ComputeGemm = TMACComputeGemm_avx2;
    d.PackQuantBData = PackQuantBData_avx2;
    d.PackScalesAndZeroPoints = PackScalesAndZeroPoints_avx2;
    return d;
}();
