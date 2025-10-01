/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx2.cpp.h

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for x64 avx2.

--*/

#include "qnbitgemm.h"
#include "qlutgemm.h"
#include "sqnbitgemm_q8_block.h"
#include <vector>
// AVX2 intrinsics
#include <immintrin.h>

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
    // bits=2, g=4, ngroups_per_elem=8/g=2, simd_n_in=16, simd_n_out=8, bm=512, kfactor=16
    constexpr size_t bits = 2;
    constexpr size_t g = 4;
    constexpr size_t ngroups_per_elem = 8 / g; // 2
    constexpr size_t simd_n_in = 16;
    constexpr size_t simd_n_out = 8;
    constexpr size_t bm = 256;      // tune as needed; must be multiple of bits and mgroup
    constexpr size_t kfactor = 16;  // tune as needed; must divide K/g per block

    // Basic checks
    MLAS_UNREFERENCED_PARAMETER(K);
    assert(BlkLen % g == 0);
    assert((BlkLen / g) % kfactor == 0);
    const int mgroup = ngroups_per_elem * simd_n_in; // 32
    assert(bm % mgroup == 0);
    assert(bm % bits == 0);

    uint8_t * buf = new uint8_t[N * bits * (K / g)];
    memset(buf, 0, N * bits * (K / g));

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(bits, BlkLen); // BlkLen/4 bytes
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

// pass in LUT for
size_t
SQ2BitGemmKernel_CompInt8_avx2(
    size_t BlkLen, // group
    const std::byte* QuantA,
    const std::byte* QuantBData, // we pass in the LUT here
    const float* QuantBScale, // LUT scales
    const std::byte* QuantBZeroPoint, // LUT zero points
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t /*BlockCountK*/, // number of k blocks of length blklen??
        size_t /*ldc*/, // leading dimension for c (unused for CountN==1 path)
    const float* /*Bias*/ // bias per output col for c
)
{
    // Implement qgemm_lut_int8_g4 (AVX2 path) for Bits=2, g=4, ActK=16, CountN == 1, K % 16 == 0.
    // Notes:
    // - This uses the same A/LUT/scales/biases layout assumptions as tmac's tbl.cpp AVX2 path.
    // - C is updated in the same lane order as tmac (tile-local contiguous), which is fine for CountN==1.

    constexpr int Bits = 2;
    constexpr int ActK = 16;
    MLAS_UNREFERENCED_PARAMETER(BlkLen);

    // Preconditions we support in this initial implementation.
    if (CountN != 1 || (CountK % ActK) != 0) {
        return 0; // not handled
    }

    const uint8_t* a = reinterpret_cast<const uint8_t*>(QuantA);
    const int8_t* lut = reinterpret_cast<const int8_t*>(QuantBData);
    const float* lut_scales = QuantBScale; // one per kk-chunk (ActK)
    const float* lut_biases = reinterpret_cast<const float*>(QuantBZeroPoint); // one per kk-chunk (ActK)
    float* c = C;

    // Process rows in groups of 32 as in tmac AVX2 path (i iterates 16 over m/2).
    size_t rows_handled = (CountM / 32) * 32;
    if (rows_handled == 0) {
        return 0;
    }

    const __m128i vec_mask = _mm_set1_epi8(0x0f);

    for (size_t i = 0; i < rows_handled / 2; i += 16) {
        __m256 vec_c0{}, vec_c1{}, vec_c2{}, vec_c3{};
        bool c_initialized = false;
        float partial_sum = -0.0f;

        for (size_t kk = 0; kk < CountK; kk += ActK) {
            // Accumulators for this kk-chunk: sum 16 int8 lookups across ActK into 4x8 lanes
            __m128i acc_lo_low = _mm_setzero_si128();
            __m128i acc_lo_high = _mm_setzero_si128();
            __m128i acc_hi_low = _mm_setzero_si128();
            __m128i acc_hi_high = _mm_setzero_si128();

            for (int k = 0; k < ActK; ++k) {
                // Load 16 LUT entries for this k (indices 0..15)
                const __m128i vec_lut_k = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut + (kk + k) * 16));
                // Load 16 selector bytes for bottom/top nibbles from A for this (i-block, k)
                const __m128i vec_as = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i * CountK + (kk + k) * 16));
                const __m128i vec_a_bot = _mm_and_si128(vec_as, vec_mask);
                const __m128i vec_a_top = _mm_and_si128(_mm_srli_epi16(vec_as, 4), vec_mask);

                // Shuffle-gather from LUT using bottom and top nibble indices
                const __m256i vec_lut_dup = _mm256_set_m128i(vec_lut_k, vec_lut_k);
                const __m256i vec_a_bt = _mm256_set_m128i(vec_a_top, vec_a_bot);
                const __m256i vec_v = _mm256_shuffle_epi8(vec_lut_dup, vec_a_bt); // 32 int8 results

                // Split to 2x16 and sign-extend to int16
                const __m128i v_bot8 = _mm256_castsi256_si128(vec_v);
                const __m128i v_top8 = _mm256_extracti128_si256(vec_v, 1);

                const __m256i vb16 = _mm256_cvtepi8_epi16(v_bot8);
                const __m256i vt16 = _mm256_cvtepi8_epi16(v_top8);

                const __m128i vb16_low = _mm256_castsi256_si128(vb16);
                const __m128i vb16_high = _mm256_extracti128_si256(vb16, 1);
                const __m128i vt16_low = _mm256_castsi256_si128(vt16);
                const __m128i vt16_high = _mm256_extracti128_si256(vt16, 1);

                acc_lo_low  = _mm_add_epi16(acc_lo_low,  vb16_low);
                acc_lo_high = _mm_add_epi16(acc_lo_high, vb16_high);
                acc_hi_low  = _mm_add_epi16(acc_hi_low,  vt16_low);
                acc_hi_high = _mm_add_epi16(acc_hi_high, vt16_high);
            }

            // Convert to float vectors (4 groups of 8)
            const __m256 vec_v_low_low   = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(acc_lo_low));
            const __m256 vec_v_low_high  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(acc_lo_high));
            const __m256 vec_v_high_low  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(acc_hi_low));
            const __m256 vec_v_high_high = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(acc_hi_high));

            float lut_s = lut_scales[kk / ActK];
            float lut_b = lut_biases ? lut_biases[kk / ActK] : 0.0f;
            partial_sum += lut_b;

            // Apply per-bit-group bias pattern: add bias only when (ib % Bits == 0)
            auto fma_with_bias = [&](const __m256& vs, size_t ib) {
                if ((ib % Bits) == 0) {
                    return _mm256_fmadd_ps(vs, _mm256_set1_ps(lut_s), _mm256_set1_ps(lut_b));
                } else {
                    return _mm256_mul_ps(vs, _mm256_set1_ps(lut_s));
                }
            };

            if (!c_initialized) {
                vec_c0 = fma_with_bias(vec_v_low_low,   (i / 4));
                vec_c1 = fma_with_bias(vec_v_low_high,  (i / 4 + 1));
                vec_c2 = fma_with_bias(vec_v_high_low,  (i / 4 + 2));
                vec_c3 = fma_with_bias(vec_v_high_high, (i / 4 + 3));
                c_initialized = true;
            } else {
                vec_c0 = _mm256_add_ps(vec_c0, fma_with_bias(vec_v_low_low,   (i / 4)));
                vec_c1 = _mm256_add_ps(vec_c1, fma_with_bias(vec_v_low_high,  (i / 4 + 1)));
                vec_c2 = _mm256_add_ps(vec_c2, fma_with_bias(vec_v_high_low,  (i / 4 + 2)));
                vec_c3 = _mm256_add_ps(vec_c3, fma_with_bias(vec_v_high_high, (i / 4 + 3)));
            }
        } // kk

        // Store back to C in tmac lane order: 8 floats x 4 groups
        _mm256_storeu_ps(c + i * 2,       vec_c0);
        _mm256_storeu_ps(c + i * 2 + 8,   vec_c1);
        _mm256_storeu_ps(c + i * 2 + 16,  vec_c2);
        _mm256_storeu_ps(c + i * 2 + 24,  vec_c3);
    }

    return rows_handled;
}

void partial_max_g4_int8_k8(float* lut_scales, float* b) {
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

void lut_ctor_g4_int8_impl(
    int32_t group_size,
    int8_t* qlut,
    float* b,
    float* lut_scales,
    float* lut_biases
) {
    const int act_k = group_size; // we assume K == group_size for now

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
	int32_t group_size,
	int8_t* lut,
	float* b,
	float* scales,
	float* biases,
    int K
) {
    const int kk_outer_max = K / group_size;

    for (int32_t kk_outer = 0; kk_outer < kk_outer_max; ++kk_outer) {
        // compute partial max - directly reset scale to 0.0
        scales[kk_outer] = 0.0f;
        for (int32_t k_outer = 0; k_outer < group_size / 32; ++k_outer) {
            partial_max_g4_int8_k8(&scales[kk_outer], &b[(kk_outer * group_size) + (k_outer * 32)]);
        }
    }

    for (int32_t k_outer_1 = 0; k_outer_1 < kk_outer_max; ++k_outer_1) {
        lut_ctor_g4_int8_impl(group_size, (&(lut[(k_outer_1 * group_size * 4)])), (&(b[(k_outer_1 * group_size)])), (&(scales[k_outer_1])), (&(biases[k_outer_1])));
    }

}

// try adding this back in:

void
QuantizeARow_CompInt8(
    size_t /*BlkLen*/,
    const float* /*A*/,
    size_t /*CountK*/,
    std::byte* /*QuantA*/
) {
    // Not implemented yet.
}

// Kernel dispatch structure definition.

const MLAS_QNBIT_LUT_GEMM_DISPATCH MlasLUTGenKernelAvx2 = []() {
    MLAS_QNBIT_LUT_GEMM_DISPATCH d;
    d.GenerateLUT = GenerateLUT_avx2;
    return d;
}();
