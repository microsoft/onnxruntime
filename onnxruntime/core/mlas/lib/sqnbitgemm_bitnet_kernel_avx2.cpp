/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx2.cpp.h

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for x64 avx2.

--*/

#include <cstddef>
#include <type_traits>
#include <vector>
// AVX2 intrinsics
#include <immintrin.h>

#include "qnbitgemm.h"
#include "qlutgemm.h"
#include "sqnbitgemm_q8_block.h"

static inline float _mm256_addv_ps(const __m256 v) {
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

    inline void push(__m256i v, int k) {
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

    inline __m256i get() {
        return lhs;
    }

    inline __m256i get_low() {
        return extract_low_epi8_epi16(lhs);
    }

    inline __m256i get_high() {
        return extract_high_epi8_epi16(lhs);
    }
};

template <>
struct SignedHalvingAdder<2> {
    __m256i lhs = _mm256_setzero_si256();

    inline void push(__m256i v, int k) {
        if (k == 0) {
            lhs = v;
        } else {
            lhs = _mm256_avg_epu8(lhs, v);
        }
    }

    inline __m256i get() {
        return lhs;
    }

    inline __m256i get_low() {
        return extract_low_epi8_epi16(lhs);
    }

    inline __m256i get_high() {
        return extract_high_epi8_epi16(lhs);
    }
};

template <int N>
struct SignedWideningAdder {
    __m256i lhs_low = _mm256_setzero_si256();
    __m256i lhs_high = _mm256_setzero_si256();

    inline void push(__m256i v, int k) {
        if (k == 0) {
            lhs_low = extract_low_epi8_epi16(v);
            lhs_high = extract_high_epi8_epi16(v);
        } else {
            lhs_low = _mm256_add_epi16(lhs_low, extract_low_epi8_epi16(v));
            lhs_high = _mm256_add_epi16(lhs_high, extract_high_epi8_epi16(v));
        }
    }

    inline __m256i get_low() {
        return lhs_low;
    }

    inline __m256i get_high() {
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
template<int bits>
constexpr int get_bias_scale() {
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

size_t
Q2BitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t /*BlkLen*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
  // TODO: This code shall change according to T-Mac.
  // Modify based on tmac compute type if needed.
    MLAS_UNREFERENCED_PARAMETER(ComputeType);  // same size regardless of ComputeType

    // const size_t PackedQuantBDataSize = N * K / 8;
    constexpr size_t BlkBitWidth = 2;
    constexpr size_t g = 4; // group size
    const size_t ngroups_per_elem = 8 / g;
    const size_t PackedQuantBDataSize = (N * BlkBitWidth) * (K / g / ngroups_per_elem);
    return PackedQuantBDataSize; // 1048576
}

void SQ2BitGemmPackQuantBData(
  size_t N,
  size_t K,
  size_t BlkLen,
  MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/,
  const std::byte* QuantBDataBegin,
  std::byte* PackedQuantBDataBegin,
  MLAS_THREADPOOL* ThreadPool
)
{
    //decompose W into w1,... w_bits create temp buffer buf2 of size N * bits * (K/g)

    // T-MAC like configuration (approved):
    // bits=2, g=4, ngroups_per_elem=8/g=2, simd_n_in=16, simd_n_out=8, bm=256, kfactor=16
    const MlasTMACKernelParams& tmac_params = GetTMACKernelParams(N, K, 2);
    const size_t bits = 2;
    const size_t g = tmac_params.g;
    const size_t ngroups_per_elem = tmac_params.ngroups_per_elem;
    const size_t simd_n_in = tmac_params.simd_n_in;
    const size_t simd_n_out = tmac_params.simd_n_out;
    const size_t bm = tmac_params.bm;
    const size_t kfactor = tmac_params.kfactor;

    // Basic checks
    MLAS_UNREFERENCED_PARAMETER(K);
    assert(BlkLen % g == 0);
    assert((BlkLen / g) % kfactor == 0);
    const int mgroup = ngroups_per_elem * simd_n_in; // 32
    assert(bm % mgroup == 0);
    assert(bm % bits == 0);

    uint8_t * buf = new uint8_t[N * bits * (K / g)];
    memset(buf, 0, N * bits * (K / g));

    const size_t Iterations = N; // we parallelize over N, TODO:: tune if needed

    MlasTrySimpleParallel(
        ThreadPool, Iterations,
        [&](ptrdiff_t tid) {
            size_t im = static_cast<size_t>(tid);
            for (size_t ik = 0; ik < K; ++ik) {
                size_t idx = (im * K + ik);
                size_t num_elem_per_byte = 8 / bits;
                size_t elem_idx = idx % num_elem_per_byte;

                uint8_t v = ((const uint8_t *)QuantBDataBegin)[idx / num_elem_per_byte] >> (elem_idx * bits);

                for (size_t ib =0; ib < bits; ++ib) {
                    size_t new_ik = ik / g;
                    size_t shft_left = ik % g;
                    buf[im * bits * K / g + ib * K /g  + new_ik] += ((v >> ib) & 1) << shft_left;
                }
            }
        }
    );

    // Now buf contains the bit planes grouped by g along K
    // Next, we need to do a multi-reshape/transpose into the final layout


    const size_t c0_fac2 = K / g;
    const size_t c0_fac1 = simd_n_out * c0_fac2;
    const size_t c0_fac0 = bits * c0_fac1;

    const size_t c1_nb2 = K / g;
    const size_t c1_nb1 = simd_n_in * c1_nb2;
    const size_t c1_nb0 = ngroups_per_elem * c1_nb1;
    const size_t c1_fac2 = K / g;
    const size_t c1_fac1 = ngroups_per_elem * c1_fac2;
    const size_t c1_fac0 = simd_n_in * c1_fac1;


    const size_t c2_nb4 = kfactor;
    const size_t c2_nb3 = K / g / kfactor * c2_nb4;
    const size_t c2_nb2 = ngroups_per_elem * c2_nb3;
    const size_t c2_nb1 = simd_n_in * c2_nb2;
    const size_t c2_nb0 = bm / mgroup * c2_nb1;
    const size_t c2_fac3 = simd_n_in * ngroups_per_elem;
    const size_t c2_fac2 = kfactor * c2_fac3;
    const size_t c2_fac1 = bm / mgroup * c2_fac2;
    const size_t c2_fac0 = K / g / kfactor * c2_fac1;

    const size_t PackedQuantBDataSize = (N * bits) * (K / g / ngroups_per_elem);
    memset(PackedQuantBDataBegin, 0, PackedQuantBDataSize); // TODO: is this needed?

    MlasTrySimpleParallel(
        ThreadPool, Iterations,
        [&](ptrdiff_t tid) {
            size_t im = static_cast<size_t>(tid);
            for (size_t ib = 0; ib < bits; ib++) {
                for (size_t ik = 0; ik < K / g; ik++) {
                    // w = w.reshape(M // bits // simd_n_out, simd_n_out, bits, K // g).transpose(0, 2, 1, 3)
                    size_t new_im = im / simd_n_out;
                    size_t new_isno = im % simd_n_out;
                    size_t new_ib = ib;
                    size_t new_ik = ik;
                    size_t new_idx = new_im * c0_fac0 + new_ib * c0_fac1 + new_isno * c0_fac2 + new_ik;

                    // w = w.reshape(M // mgroup, ngroups_per_elem, simd_n_in, K // g).transpose(0, 2, 1, 3)
                    new_im = new_idx / c1_nb0;
                    size_t new_ing = (new_idx % c1_nb0) / c1_nb1;
                    size_t new_isni = (new_idx % c1_nb1) / c1_nb2;
                    new_ik = (new_idx % c1_nb2);
                    new_idx = new_im * c1_fac0 + new_isni * c1_fac1 + new_ing * c1_fac2 + new_ik;

                    // #             0        1             2             3                 4                  5
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
                    size_t buf_idx = im * bits * K / g + ib * K / g + ik;
                    uint8_t buf_val = buf[buf_idx];

                    // w = sum([(w[:, :, :, :, :, ng] << (ng * g)) for ng in range(ngroups_per_elem)])
                    PackedQuantBDataBegin[new_idx] = static_cast<std::byte>(
                        static_cast<unsigned>(PackedQuantBDataBegin[new_idx]) +
                        (buf_val << (new_ing * g)));
                }
            }
        }
    );
    delete[] buf;
}

size_t
Q2BitGemmPerGemmWorkspaceSize(
    size_t M,
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(N);

    switch (ComputeType) {
        case SQNBIT_CompInt8: {
            // workspace buffer is used for block quantization of A to int8
            const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
            // QuantData + Scale
            const size_t PerGemmWorkspaceSize = M * BlockCountK * Q8BlkSize(BlkLen);
            return PerGemmWorkspaceSize;
        }
        default: {
            return 0;
        }
    }
}

void partial_max_g4_int8_k8(float* lut_scales, const float* b) {
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

inline void lut_ctor_g4_int8_impl(
    int32_t act_k,
    int8_t* qlut,
    const float* b,
    float* lut_scales,
    float* lut_biases
) {

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
            //vec_lut[g] = -vec_lut[15 - g];
            const __m256 neg_mask = _mm256_set1_ps(-0.0f);  // all lanes have sign bit set
            vec_lut[g] = _mm256_xor_ps(vec_lut[15 - g], neg_mask);
        }

        biases += _mm256_addv_ps(vec_lut[0]);

PRAGMA_UNROLL
        for (int g = 0; g < 16; ++g) {
            vec_lut[g] = _mm256_mul_ps(vec_lut[g], _mm256_set1_ps(t_scales));
        }

        __m256i vec_qlut[4];
        const __m256i shuf = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                              0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
PRAGMA_UNROLL
        for (int g = 0; g < 4; g += 1) {
            __m256i i0 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 0], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 1], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i2 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 2], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i3 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 3], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

            i0 = _mm256_packs_epi32(i0, i1);	         // 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
            i2 = _mm256_packs_epi32(i2, i3);	         // 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                                         // Convert int16 to int8
            i0 = _mm256_packs_epi16(i0, i2);	         // 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7,  12, 13, 14, 15,  20, 21, 22, 23,  28, 29, 30, 31
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
) {
    const size_t kk_outer_max = K / act_group_size;

    for (int32_t kk_outer = 0; kk_outer < kk_outer_max; ++kk_outer) {
        // compute partial max - directly reset scale to 0.0
        lut_scales[kk_outer] = 0.0f; // partial max reset
        for (int32_t k_outer = 0; k_outer <act_group_size / 32; ++k_outer) {
            partial_max_g4_int8_k8(&lut_scales[kk_outer], &b[(kk_outer * act_group_size) + (k_outer * 32)]);
        }
    }

    for (int32_t k_outer_1 = 0; k_outer_1 < kk_outer_max; ++k_outer_1) {
        lut_ctor_g4_int8_impl(act_group_size, (&(qlut[(k_outer_1 * act_group_size * 4)])), (&(b[(k_outer_1 * act_group_size)])), (&(lut_scales[k_outer_1])), (&(lut_biases[k_outer_1])));
    }

}

inline void tbl_g4_int8_float_gather_bit2_impl(int32_t m, float* C_global, float* CBits, float* C) {
    constexpr int32_t bits = 2;

    int32_t m_c_outer_max = m / 32;
    for (int32_t m_c_outer = 0; m_c_outer < m_c_outer_max; ++m_c_outer) {
        int32_t cse_var_2 = (m_c_outer * 32 * bits);
        int32_t cse_var_1 = (m_c_outer * 32);
        PRAGMA_UNROLL
        for (int32_t m_c_inner = 0; m_c_inner < 32; ++m_c_inner) {
            int32_t bit_offset_0 = (m_c_inner / 8) * 8 * bits + (m_c_inner % 8);
            int32_t bit_offset_1 = (m_c_inner / 8) * 8 * bits + (m_c_inner % 8) + 8;
            C_global[cse_var_1 + m_c_inner] = (CBits[cse_var_2 + bit_offset_0] * (float)5.000000e-01f)
                                            + (CBits[cse_var_2 + bit_offset_1]);
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
inline int32_t tbl_g4_int8_float_update_impl(int32_t m, float* c, const int8_t* lut, const uint8_t* a, const float* scales, const float* lut_scales, const float* lut_biases) {
    const __m128i vec_mask = _mm_set1_epi8(0x0f);
    __m128i vec_lut[K];

    PRAGMA_UNROLL
    for (int k = 0; k < K; k++) {
        vec_lut[k] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut + k * 16));
    }

    SignedAdder<FastAggregation, ActK> adder;
    for (int i = 0; i < m / 2; i += 16) {
        __m256 vec_c0, vec_c1, vec_c2, vec_c3;

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

#define lut_fma(vs, ib) \
    ((ib) % Bits) ? (_mm256_mul_ps((vs),   _mm256_set1_ps(lut_s))) \
                  : (_mm256_fmadd_ps((vs), _mm256_set1_ps(lut_s), _mm256_set1_ps(lut_b)))
            if (kk == 0) {
                vec_c0 = lut_fma(vec_v_low_low,   (i / 4    ));
                vec_c1 = lut_fma(vec_v_low_high,  (i / 4 + 1));
                vec_c2 = lut_fma(vec_v_high_low,  (i / 4 + 2));
                vec_c3 = lut_fma(vec_v_high_high, (i / 4 + 3));
            } else {
                vec_c0 = _mm256_add_ps(vec_c0, lut_fma(vec_v_low_low,   (i / 4    )));
                vec_c1 = _mm256_add_ps(vec_c1, lut_fma(vec_v_low_high,  (i / 4 + 1)));
                vec_c2 = _mm256_add_ps(vec_c2, lut_fma(vec_v_high_low,  (i / 4 + 2)));
                vec_c3 = _mm256_add_ps(vec_c3, lut_fma(vec_v_high_high, (i / 4 + 3)));
            }
#undef lut_fma
        }

        if (ZeroPoint) {
            __m256 vec_s0 = _mm256_loadu_ps(scales + ((i / 4    ) / Bits) * 16);
            __m256 vec_s1 = _mm256_loadu_ps(scales + ((i / 4 + 1) / Bits) * 16);
            __m256 vec_s2 = _mm256_loadu_ps(scales + ((i / 4 + 2) / Bits) * 16);
            __m256 vec_s3 = _mm256_loadu_ps(scales + ((i / 4 + 3) / Bits) * 16);
            vec_c0 = _mm256_fmadd_ps(vec_c0, vec_s0, _mm256_loadu_ps(c + i * 2));
            vec_c1 = _mm256_fmadd_ps(vec_c1, vec_s1, _mm256_loadu_ps(c + i * 2 + 8));
            vec_c2 = _mm256_fmadd_ps(vec_c2, vec_s2, _mm256_loadu_ps(c + i * 2 + 16));
            vec_c3 = _mm256_fmadd_ps(vec_c3, vec_s3, _mm256_loadu_ps(c + i * 2 + 24));
            __m256 vec_z0 = _mm256_loadu_ps(scales + ((i / 4    ) / Bits) * 16 + 8);
            __m256 vec_z1 = _mm256_loadu_ps(scales + ((i / 4 + 1) / Bits) * 16 + 8);
            __m256 vec_z2 = _mm256_loadu_ps(scales + ((i / 4 + 2) / Bits) * 16 + 8);
            __m256 vec_z3 = _mm256_loadu_ps(scales + ((i / 4 + 3) / Bits) * 16 + 8);
            partial_sum *= 2;
#define add_zero(cs, zs, ib) \
    ((ib) % Bits) ? ((cs)) \
                  : (_mm256_fmadd_ps((zs), _mm256_set1_ps(partial_sum), (cs)))
            _mm256_storeu_ps(c + i * 2,      add_zero(vec_c0, vec_z0, (i / 4    )));
            _mm256_storeu_ps(c + i * 2 + 8,  add_zero(vec_c1, vec_z1, (i / 4 + 1)));
            _mm256_storeu_ps(c + i * 2 + 16, add_zero(vec_c2, vec_z2, (i / 4 + 2)));
            _mm256_storeu_ps(c + i * 2 + 24, add_zero(vec_c3, vec_z3, (i / 4 + 3)));
#undef add_zero
        } else if (OneScale) {
            float single_scale = scales[0];
            __m256 vec_s = _mm256_set1_ps(single_scale);
            _mm256_storeu_ps(c + i * 2,      _mm256_fmadd_ps(vec_c0, vec_s, _mm256_loadu_ps(c + i * 2)));
            _mm256_storeu_ps(c + i * 2 + 8,  _mm256_fmadd_ps(vec_c1, vec_s, _mm256_loadu_ps(c + i * 2 + 8)));
            _mm256_storeu_ps(c + i * 2 + 16, _mm256_fmadd_ps(vec_c2, vec_s, _mm256_loadu_ps(c + i * 2 + 16)));
            _mm256_storeu_ps(c + i * 2 + 24, _mm256_fmadd_ps(vec_c3, vec_s, _mm256_loadu_ps(c + i * 2 + 24)));
        } else {
            __m256 vec_s0 = _mm256_loadu_ps(scales + ((i / 4    ) / Bits) * 8);
            __m256 vec_s1 = _mm256_loadu_ps(scales + ((i / 4 + 1) / Bits) * 8);
            __m256 vec_s2 = _mm256_loadu_ps(scales + ((i / 4 + 2) / Bits) * 8);
            __m256 vec_s3 = _mm256_loadu_ps(scales + ((i / 4 + 3) / Bits) * 8);
            _mm256_storeu_ps(c + i * 2,      _mm256_fmadd_ps(vec_c0, vec_s0, _mm256_loadu_ps(c + i * 2)));
            _mm256_storeu_ps(c + i * 2 + 8,  _mm256_fmadd_ps(vec_c1, vec_s1, _mm256_loadu_ps(c + i * 2 + 8)));
            _mm256_storeu_ps(c + i * 2 + 16, _mm256_fmadd_ps(vec_c2, vec_s2, _mm256_loadu_ps(c + i * 2 + 16)));
            _mm256_storeu_ps(c + i * 2 + 24, _mm256_fmadd_ps(vec_c3, vec_s3, _mm256_loadu_ps(c + i * 2 + 24)));
        }
    }

    return 0;
}

int32_t tbl_int32_reset(int32_t m, int32_t* c) {
    memset(c, 0, m * sizeof(int32_t));
    return 0;
}

// based on qgemm_lut_int8_g4
// Simplified version with hardcoded configuration for 2-bit quantization
void TMACComputeGemm_avx2(
            const uint8_t* A,            // Quantized packed weights
            const float* Scales,       // Weight scales (and optionally zero-points)
            const int8_t* LUT,          // Pre-computed quantized lookup table
            const float* LUT_Scales,   // LUT scales from activation quantization
            const float* LUT_Biases,   // LUT biases from activation quantization
            void* C,            // Output buffer
            int K,
            int M,
            int N,
            size_t BlkLen      // Weight quantization group size (q_group_size)
            ) {
    // // Validate batch size
    if (N != 1) {
        throw std::runtime_error("N > 1 is not supported yet");
    }

    // get kernel config
    const MlasTMACKernelParams& tmac_params = GetTMACKernelParams(M, K, 2);



    // ==================== CONFIGURATION ====================
    // Fixed parameters for this kernel implementation
    bool has_zero_point = tmac_params.has_zero_point; // Whether weights have zero-points (interleaved with scales)
    bool one_scale = tmac_params.one_scale;     // Whether using single global scale for all weights

    const int bits = tmac_params.bits;              // 2-bit quantization
    const int g = tmac_params.g;                 // Packing group size
    const int ngroups_per_elem = tmac_params.ngroups_per_elem;  // 8 / g = 2
    const int kfactor = tmac_params.kfactor;          // K-dimension blocking factor

    const bool has_scale = tmac_params.has_scale;     // Always use weight scales

    // Parameters derived from inputs
    const int q_group_size = tmac_params.q_group_size;  // Weight quant group size
    const int act_group_size = tmac_params.act_group_size; // Activation group size (same as weight)
    const int actk = tmac_params.actk;  // CRITICAL: = 16 for BlkLen=64, NOT BlkLen!

    const int bm = tmac_params.bm;
    int m = bm / bits;

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
    tbl_int32_reset(bm * sizeof(float) / sizeof(int32_t),
                    reinterpret_cast<int32_t*>(CBits));

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
        fprintf(stderr, "q_group_size=%d, kfactor=%d, g=%d\n", q_group_size, kfactor, g);
        fprintf(stderr, "Unsupported scale group size over kfactor. Expected {1,2,4,8}, got %d.\n", scale_gs);
        throw std::runtime_error("Invalid scale group size configuration");
    }

    // ==================== MAIN COMPUTATION LOOP ====================
    for (int32_t k_outer = 0; k_outer < k_outer_max; k_outer++) {
        // Calculate pointers for this K-outer iteration
        const uint8_t* a = A + k_outer * bm * kfactor / ngroups_per_elem;

        // Calculate scales pointer based on configuration
        const float* scales = one_scale ?
            reinterpret_cast<const float*>(Scales) :  // Single global scale
            (has_zero_point ?
                reinterpret_cast<const float*>(Scales) + (k_outer >> scale_idx_shfr) * m * 2 :  // Scale + zero_point pairs
                reinterpret_cast<const float*>(Scales) + (k_outer >> scale_idx_shfr) * m);       // Scales only

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
                static_cast<int32_t>(bm), CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 16 && bits == 2 && actk == 16 && !has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 2, 16, false, false, false>(
                static_cast<int32_t>(bm), CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 16 && bits == 2 && actk == 16 && !has_zero_point && one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 2, 16, false, false, true>(
                static_cast<int32_t>(bm), CBits, lut, a, scales, lut_scales, lut_biases);
        }
        // actk == 8 variants (for BlkLen=32)
        else if (has_scale && kfactor == 16 && bits == 2 && actk == 8 && has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 2, 8, false, true, false>(
                static_cast<int32_t>(bm), CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 16 && bits == 2 && actk == 8 && !has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 2, 8, false, false, false>(
                static_cast<int32_t>(bm), CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 16 && bits == 2 && actk == 8 && !has_zero_point && one_scale) {
            tbl_g4_int8_float_update_impl<true, 16, 2, 8, false, false, true>(
                static_cast<int32_t>(bm), CBits, lut, a, scales, lut_scales, lut_biases);
        }
        // kfactor == 8 variants
        else if (has_scale && kfactor == 8 && bits == 2 && actk == 8 && has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 8, 2, 8, false, true, false>(
                static_cast<int32_t>(bm), CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 8 && bits == 2 && actk == 8 && !has_zero_point && !one_scale) {
            tbl_g4_int8_float_update_impl<true, 8, 2, 8, false, false, false>(
                static_cast<int32_t>(bm), CBits, lut, a, scales, lut_scales, lut_biases);
        } else if (has_scale && kfactor == 8 && bits == 2 && actk == 8 && !has_zero_point && one_scale) {
            tbl_g4_int8_float_update_impl<true, 8, 2, 8, false, false, true>(
                static_cast<int32_t>(bm), CBits, lut, a, scales, lut_scales, lut_biases);
        } else {
            // No matching kernel template found
            // ORT_THROW("No matching kernel: has_scale=%d, kfactor=%d, bits=%d, actk=%d, has_zero_point=%d, one_scale=%d\n",
            //         has_scale, kfactor, bits, actk, has_zero_point, one_scale);
            ORT_THROW("Reached else case");
        }
    }

    // ==================== GATHER RESULTS ====================
    // Gather bit-plane results into final output
    // Only support 2-bit in this implementation
    // TODO(vraspar): extend to other bit-widths
    tbl_g4_int8_float_gather_bit2_impl(m, C_global, CBits, reinterpret_cast<float*>(C));

    // ==================== CLEANUP ====================
    delete[] C_global;
    delete[] CBits;
}

void
QuantizeARow_CompInt8(
    size_t /*BlkLen*/,
    const float* /*A*/,
    size_t /*CountK*/,
    std::byte* /*QuantA*/
) {
    // placeholder so that dispatch doesn't break
    // TODO: figure out a way that we can omit this altogether
}


// Kernel dispatch structure definition.

const MLAS_QNBIT_LUT_GEMM_DISPATCH MlasLUTGenKernelAvx2 = []() {
    MLAS_QNBIT_LUT_GEMM_DISPATCH d;
    d.GenerateLUT = GenerateLUT_avx2;
    d.ComputeGemm = TMACComputeGemm_avx2;
    return d;
}();
