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

size_t
Q2BitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
  // TODO: This code shall change according to T-Mac.
    MLAS_UNREFERENCED_PARAMETER(ComputeType);  // same size regardless of ComputeType
    constexpr size_t BlkBitWidth = 2;

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t PackedQuantBDataSize = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
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

// TODO: do we need this..?
void
QuantizeARow_CompInt8(
    size_t /*BlkLen*/,
    const float* /*A*/,
    size_t /*CountK*/,
    std::byte* /*QuantA*/
)
{
  // shall be similar to QuantizeARow_CompInt8_avx2 without blksum related code.
  // we don't need this function -- remove from dispatch? 
}

// based on lut_ctor_g4_int8_impl
void 
GenerateLUT_avx2(
	int32_t group_size,
	int8_t* lut,
	onnxruntime::MLFloat16* b,
	onnxruntime::MLFloat16* scales,
	onnxruntime::MLFloat16* biases
) {
    // Helper to horizontally add all 8 lanes of a __m256
    auto addv_ps = [](const __m256 v) -> float {
        __m128 res = _mm256_extractf128_ps(v, 1);
        res = _mm_add_ps(res, _mm256_castps256_ps128(v));
        res = _mm_add_ps(res, _mm_movehl_ps(res, res));
        res = _mm_add_ss(res, _mm_movehdup_ps(res));
        return _mm_cvtss_f32(res);
    };

    // Read scale (already computed elsewhere) and prepare its reciprocal.
    const float scale_f = static_cast<float>(scales[0]);
    const float t_scale = scale_f != 0.0f ? (1.0f / scale_f) : 0.0f;

    // Accumulate bias across blocks of 32 (matches tmac layout: 4 interleaved streams of 8)
    float bias_acc = 0.0f;

    // Temporary buffers for converted floats
    float tmp[32];
    float b0[8], b1[8], b2[8], b3[8];

    // We produce 16 vectors per 32-wide chunk, then pack to int8 and store
    // Each block of 32 half values contributes 32 int8 entries per LUT row (16 entries x 2 halves) arranged like tmac
    for (int kblk = 0; kblk < group_size / 32; ++kblk) {
        // Convert 32 halfs to float
        const onnxruntime::MLFloat16* base = b + kblk * 32;
        for (int i = 0; i < 32; ++i) tmp[i] = static_cast<float>(base[i]);

        // De-interleave to 4 streams of 8
        for (int i = 0; i < 8; ++i) {
            b0[i] = tmp[i * 4 + 0];
            b1[i] = tmp[i * 4 + 1];
            b2[i] = tmp[i * 4 + 2];
            b3[i] = tmp[i * 4 + 3];
        }

        __m256 vec_b0 = _mm256_loadu_ps(b0);
        __m256 vec_b1 = _mm256_loadu_ps(b1);
        __m256 vec_b2 = _mm256_loadu_ps(b2);
        __m256 vec_b3 = _mm256_loadu_ps(b3);

        __m256 vec_lut[16];

        // Build odd indices 1..15: b0 +/- b1 +/- b2 +/- b3 depending on bits of g
        for (int g = 1; g < 16; g += 2) {
            __m256 v = vec_b0;
            v = (g & 0b0010) ? _mm256_add_ps(v, vec_b1) : _mm256_sub_ps(v, vec_b1);
            v = (g & 0b0100) ? _mm256_add_ps(v, vec_b2) : _mm256_sub_ps(v, vec_b2);
            v = (g & 0b1000) ? _mm256_add_ps(v, vec_b3) : _mm256_sub_ps(v, vec_b3);
            vec_lut[g] = v;
        }

        // Even indices are negatives of mirrored odd indices
        for (int g = 0; g < 16; g += 2) {
            vec_lut[g] = _mm256_sub_ps(_mm256_setzero_ps(), vec_lut[15 - g]);
        }

        // Accumulate bias from entry 0 (before scaling)
        bias_acc += addv_ps(vec_lut[0]);

        // Apply inverse scale
        const __m256 vs = _mm256_set1_ps(t_scale);
        for (int g = 0; g < 16; ++g) {
            vec_lut[g] = _mm256_mul_ps(vec_lut[g], vs);
        }

        // Round to nearest, pack to int8 with saturate, and shuffle into the final lane order
        __m256i vec_qlut[4];
        const __m256i shuf = _mm256_setr_epi8(
            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

        for (int g = 0; g < 4; ++g) {
            __m256i i0 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 0], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 1], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i2 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 2], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i3 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 3], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

            i0 = _mm256_packs_epi32(i0, i1);
            i2 = _mm256_packs_epi32(i2, i3);
            __m256i i8 = _mm256_packs_epi16(i0, i2);
            vec_qlut[g] = _mm256_shuffle_epi8(i8, shuf);
        }

        // Store 8 lanes x 4 rows for this 32-wide block
        int32_t* qlut_i32 = reinterpret_cast<int32_t*>(lut);
        for (int lane = 0; lane < 8; ++lane) {
            for (int g = 0; g < 4; ++g) {
                qlut_i32[kblk * 32 + lane * 4 + g] = _mm256_extract_epi32(vec_qlut[g], lane);
            }
        }
    }

    // Write back bias and leave scale as-is
    biases[0] = onnxruntime::MLFloat16(bias_acc);
    // scales[0] unchanged
    return;
}

// Kernel dispatch structure definition.

const MLAS_QNBIT_LUT_GEMM_DISPATCH MlasLUTGenKernelAvx2 = []() {
    MLAS_QNBIT_LUT_GEMM_DISPATCH d;
    d.GenerateLUT = GenerateLUT_avx2;
    return d;
}();