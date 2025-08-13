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
#include "sqnbitgemm_q8_block.h"
#include <vector>

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

size_t
SQ2BitGemmKernel_CompInt8_avx2(
    size_t /*BlkLen*/,
    const std::byte* /*QuantA*/,
    const std::byte* /*QuantBData*/,
    const float* /*QuantBScale*/,
    const std::byte* /*QuantBZeroPoint*/,
    float* /*C*/,
    size_t /*CountM*/,
    size_t /*CountN*/,
    size_t /*CountK*/,
    size_t /*BlockCountK*/,
    size_t /*ldc*/,
    const float* /*Bias*/
)
{
  // reference SQ4BitGemmKernel_CompInt8_avx2
    return 0;
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
}
