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
    MLAS_UNREFERENCED_PARAMETER(ComputeType);  // same size regardless of ComputeType

    const size_t PackedQuantBDataSize = N * K / 8;
    return PackedQuantBDataSize;
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
    // T-MAC like configuration (approved):
    // bits=2, g=4, ngroups_per_elem=8/g=2, simd_n_in=16, simd_n_out=8, bm=512, kfactor=16
    constexpr int bits = 2;
    constexpr int g = 4;
    constexpr int ngroups_per_elem = 8 / g; // 2
    constexpr int simd_n_in = 16;
    constexpr int simd_n_out = 8;
    constexpr int bm = 512;      // tune as needed; must be multiple of bits and mgroup
    constexpr int kfactor = 16;  // tune as needed; must divide K/g per block

    // Basic checks
    MLAS_UNREFERENCED_PARAMETER(K);
    assert(BlkLen % g == 0);
    assert((BlkLen / g) % kfactor == 0);
    const int mgroup = ngroups_per_elem * simd_n_in; // 32
    assert(bm % mgroup == 0);
    assert(bm % bits == 0);

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(bits, BlkLen); // BlkLen/4 bytes

    const int m_block = bm / bits;       // number of original rows (columns of B) per tile
    assert(N % static_cast<size_t>(m_block) == 0);
    const size_t tiles_in_m = N / static_cast<size_t>(m_block);

    const int K_over_g = static_cast<int>(BlkLen / g);

    // We write destination in block-major layout: for each k-block, its N columns packed contiguously.
    // Per (k_blk, tile) we produce a chunk of size m_block * BlkDataSize bytes.
    const size_t tile_chunk_bytes = static_cast<size_t>(m_block) * BlkDataSize; // = m_block * BlkLen/4

    const size_t Iterations = BlockCountK * tiles_in_m;

    MlasTrySimpleParallel(
        ThreadPool, Iterations,
        [&](ptrdiff_t tid) {
            const size_t k_blk = static_cast<size_t>(tid) / tiles_in_m;
            const size_t tile_idx = static_cast<size_t>(tid) % tiles_in_m;

            // Temporary buffers per tile
            // buf2: size = (m_block * bits) * (BlkLen/g)
            // tilechunk: size = m_block * BlkLen/4 bytes
            std::vector<uint8_t> buf2(static_cast<size_t>(m_block) * bits * K_over_g, 0);
            std::vector<uint8_t> tilechunk(tile_chunk_bytes, 0);

            // Stage 1: build buf2 (bit-planes grouped along K by g)
            for (int im = 0; im < m_block; ++im) {
                const size_t n_col = tile_idx * static_cast<size_t>(m_block) + static_cast<size_t>(im);
                const size_t src_block_offset = n_col * BlockCountK * BlkDataSize + k_blk * BlkDataSize;
                const std::byte* src_block = QuantBDataBegin + src_block_offset;

                for (int ik = 0; ik < static_cast<int>(BlkLen); ++ik) {
                    const int byte_idx = ik >> 2;                 // ik/4
                    const int lane = ik & 3;                      // ik%4
                    const uint8_t src_byte = static_cast<uint8_t>(src_block[byte_idx]);
                    const uint8_t v = static_cast<uint8_t>((src_byte >> (lane * bits)) & 0x3u);

                    const int ik_g = ik / g;
                    const int shft_left = ik % g; // 0..3
                    for (int ib = 0; ib < bits; ++ib) {
                        const size_t idx = static_cast<size_t>(im) * bits * K_over_g + static_cast<size_t>(ib) * K_over_g + static_cast<size_t>(ik_g);
                        buf2[idx] = static_cast<uint8_t>(buf2[idx] + (((v >> ib) & 0x1u) << shft_left));
                    }
                }
            }

            // Precompute reshape/transpose factors (use K' = BlkLen)
            const int c0_fac2 = K_over_g;
            const int c0_fac1 = simd_n_out * c0_fac2;
            const int c0_fac0 = bits * c0_fac1;

            const int c1_nb2 = K_over_g;
            const int c1_nb1 = simd_n_in * c1_nb2;
            const int c1_nb0 = ngroups_per_elem * c1_nb1;
            const int c1_fac2 = K_over_g;
            const int c1_fac1 = ngroups_per_elem * c1_fac2;
            const int c1_fac0 = simd_n_in * c1_fac1;

            const int c2_nb4 = kfactor;
            const int c2_nb3 = (K_over_g / kfactor) * c2_nb4;
            const int c2_nb2 = ngroups_per_elem * c2_nb3;
            const int c2_nb1 = simd_n_in * c2_nb2;
            const int c2_nb0 = (bm / mgroup) * c2_nb1;
            const int c2_fac3 = simd_n_in * ngroups_per_elem;
            const int c2_fac2 = kfactor * c2_fac3;
            const int c2_fac1 = (bm / mgroup) * c2_fac2;
            const int c2_fac0 = (K_over_g / kfactor) * c2_fac1;

            // Stage 2: multi-reshape/transpose into tilechunk
            for (int im = 0; im < m_block; ++im) {
                for (int ib = 0; ib < bits; ++ib) {
                    for (int ik = 0; ik < K_over_g; ++ik) {
                        // w = w.reshape(M // bits // simd_n_out, simd_n_out, bits, K // g).transpose(0, 2, 1, 3)
                        int new_im = im / simd_n_out;
                        int new_isno = im % simd_n_out;
                        int new_ib = ib;
                        int new_ik = ik;
                        int new_idx = new_im * c0_fac0 + new_ib * c0_fac1 + new_isno * c0_fac2 + new_ik;

                        // w = w.reshape(M // mgroup, ngroups_per_elem, simd_n_in, K // g).transpose(0, 2, 1, 3)
                        new_im = new_idx / c1_nb0;
                        int new_ing = (new_idx % c1_nb0) / c1_nb1;
                        int new_isni = (new_idx % c1_nb1) / c1_nb2;
                        new_ik = (new_idx % c1_nb2);
                        new_idx = new_im * c1_fac0 + new_isni * c1_fac1 + new_ing * c1_fac2 + new_ik;

                        // w = w.reshape(M // bm, bm // mgroup, simd_n_in, ngroups_per_elem, K // g // kfactor, kfactor).transpose(0, 4, 1, 5, 2, 3)
                        new_im = new_idx / c2_nb0;
                        int new_ibm = (new_idx % c2_nb0) / c2_nb1;
                        new_isni = (new_idx % c2_nb1) / c2_nb2;
                        new_ing = (new_idx % c2_nb2) / c2_nb3;
                        new_ik = (new_idx % c2_nb3) / c2_nb4;
                        int new_ikf = (new_idx % c2_nb4);
                        new_idx = new_im * c2_fac0 + new_ik * c2_fac1 + new_ibm * c2_fac2 + new_ikf * c2_fac3 + new_isni * ngroups_per_elem + new_ing;

                        // Collapse ngroups into byte by left-shifting lanes of g
                        const size_t src_idx = static_cast<size_t>(im) * bits * K_over_g + static_cast<size_t>(ib) * K_over_g + static_cast<size_t>(ik);
                        const uint8_t v = buf2[src_idx];
                        const size_t dst_idx = static_cast<size_t>(new_idx / ngroups_per_elem);
                        tilechunk[dst_idx] = static_cast<uint8_t>(tilechunk[dst_idx] + (v << (new_ing * g)));
                    }
                }
            }

            // Store the tile chunk into destination
            std::byte* dst_block_base = PackedQuantBDataBegin + k_blk * (N * BlkDataSize);
            std::byte* tile_dest = dst_block_base + tile_idx * tile_chunk_bytes;
            // copy bytes
            for (size_t i = 0; i < tile_chunk_bytes; ++i) {
                tile_dest[i] = static_cast<std::byte>(tilechunk[i]);
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
            vec_lut[g] = -vec_lut[15 - g];
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
// TODO: change the name of this + change the dispatch
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

int32_t tbl_int32_reset(int32_t m, int32_t* c) {
    memset(c, 0, m * sizeof(int32_t));
    return 0;
}

inline void tbl_g4_int8_float_gather_bit2_impl(int32_t m, float* C_global, float* CBits, float* C) {
    constexpr int32_t bits = 2;

    int32_t m_c_outer_max = m / 32;
    for (int32_t m_c_outer = 0; m_c_outer < m_c_outer_max; ++m_c_outer) {
        int32_t cse_var_2 = (m_c_outer * 32 * bits);
        int32_t cse_var_1 = (m_c_outer * 32);
        #pragma unroll
        for (int32_t m_c_inner = 0; m_c_inner < 32; ++m_c_inner) {
            int32_t bit_offset_0 = (m_c_inner / 8) * 8 * bits + (m_c_inner % 8);
            int32_t bit_offset_1 = (m_c_inner / 8) * 8 * bits + (m_c_inner % 8) + 8;
            C_global[cse_var_1 + m_c_inner] = (CBits[cse_var_2 + bit_offset_0] * (tmac_float_type)5.000000e-01f)
                                            + (CBits[cse_var_2 + bit_offset_1]);
        }
    }

    for (int32_t m_inner_outer = 0; m_inner_outer < m_c_outer_max; ++m_inner_outer) {
        #pragma unroll
        for (int32_t m_inner = 0; m_inner < 32; ++m_inner) {
            int offset = m_inner_outer * 32 + m_inner;
            C[offset] = C_global[offset];
        }
    }
}

// When FastAggregation is enabled, FastAggregationK = ActK
// zero_points is merged into scales to maintain API
template <bool has_scale, int K, int Bits, int ActK, bool FastAggregation, bool ZeroPoint, bool OneScale>
inline int32_t tbl_g4_int8_float_update_impl(int32_t m, float* c, int8_t* lut, uint8_t* a, float* scales, float* lut_scales, float* lut_biases) {
    const __m128i vec_mask = _mm_set1_epi8(0x0f);
    __m128i vec_lut[K];

#pragma unroll
    for (int k = 0; k < K; k++) {
        vec_lut[k] = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 16));
    }

    SignedAdder<FastAggregation, ActK> adder;
    for (int i = 0; i < m / 2; i += 16) {
        __m256 vec_c0, vec_c1, vec_c2, vec_c3;

        tmac_float_type partial_sum = (tmac_float_type)-0.0f;
#pragma unroll
        for (int kk = 0; kk < K; kk += ActK) {
#pragma unroll
            for (int k = 0; k < ActK; k++) {
                // (M // bm, KK / K / 4, bm / 16 / 2, K * 16)
                __m128i vec_as = _mm_loadu_si128(reinterpret_cast<__m128i*>(a + i * K + (kk + k) * 16));
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
        
            tmac_float_type lut_s = lut_scales[kk / ActK];
            tmac_float_type lut_b = lut_biases[kk / ActK];

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
            tmac_float_type single_scale = scales[0];
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

// based on qgemm_lut_int8_g4
// Simplified version with hardcoded configuration for 2-bit quantization
void TMACComputeGemm_avx2(
            void* A,            // Quantized packed weights
            void* Scales,       // Weight scales (and optionally zero-points)
            void* LUT,          // Pre-computed quantized lookup table
            void* LUT_Scales,   // LUT scales from activation quantization
            void* LUT_Biases,   // LUT biases from activation quantization
            void* C,            // Output buffer
            int bm,             // Bit-rows tile size (typically 512 for 2-bit)
            int K,              // K dimension (inner dimension)
            int N,              // Batch size (must be 1 for now)
            size_t BlkLen      // Weight quantization group size (q_group_size)
            ) {
    // Validate batch size
    if (N != 1) {
        throw std::runtime_error("N > 1 is not supported yet");
    }

    // ==================== CONFIGURATION ====================
    // Fixed parameters for this kernel implementation
    bool has_zero_point = true; // Whether weights have zero-points (interleaved with scales)
    bool one_scale = false;     // Whether using single global scale for all weights
    constexpr int bits = 2;              // 2-bit quantization
    constexpr int g = 4;                 // Packing group size
    constexpr int ngroups_per_elem = 2;  // 8 / g = 2
    constexpr int kfactor = 16;          // K-dimension blocking factor
    constexpr bool has_scale = true;     // Always use weight scales
    
    // Parameters derived from inputs
    const int q_group_size = static_cast<int>(BlkLen);  // Weight quant group size
    const int act_group_size = static_cast<int>(BlkLen); // Activation group size (same as weight)
    const int actk = act_group_size / g;  // CRITICAL: = 16 for BlkLen=64, NOT BlkLen!
    const int m = bm / bits;              // Actual number of output rows
    
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
    using tmac_float_type = float;
    
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
        uint8_t* a = reinterpret_cast<uint8_t*>(A) + k_outer * bm * kfactor / ngroups_per_elem;
        
        // Calculate scales pointer based on configuration
        tmac_float_type* scales = one_scale ? 
            reinterpret_cast<tmac_float_type*>(Scales) :  // Single global scale
            (has_zero_point ? 
                reinterpret_cast<tmac_float_type*>(Scales) + (k_outer >> scale_idx_shfr) * m * 2 :  // Scale + zero_point pairs
                reinterpret_cast<tmac_float_type*>(Scales) + (k_outer >> scale_idx_shfr) * m);       // Scales only
        
        // Calculate LUT pointers
        int8_t* lut = reinterpret_cast<int8_t*>(LUT) + k_outer * kfactor * (1 << g);  // 2^g = 16 for g=4
        tmac_float_type* lut_scales = reinterpret_cast<tmac_float_type*>(LUT_Scales) + 
                                      (k_outer * kfactor * g / act_group_size);
        tmac_float_type* lut_biases = reinterpret_cast<tmac_float_type*>(LUT_Biases) + 
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
            fprintf(stderr, "No matching kernel: has_scale=%d, kfactor=%d, bits=%d, actk=%d, has_zero_point=%d, one_scale=%d\n",
                    has_scale, kfactor, bits, actk, has_zero_point, one_scale);
            throw std::runtime_error("No matching T-MAC kernel template found for configuration");
        }
    }

    // ==================== GATHER RESULTS ====================
    // Gather bit-plane results into final output
    // Only support 2-bit in this implementation
    tbl_g4_int8_float_gather_bit2_impl(m, C_global, CBits, reinterpret_cast<float*>(C));

    // ==================== CLEANUP ====================
    delete[] C_global;
    delete[] CBits;
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