/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qnbitgemm_kernel_rvv.cpp

Abstract:

    This module implements the RISC-V Vector (RVV) kernels for n-bit quantized
    GEMM (MatMulNBits).

    Implemented:
      - Packed-B / per-GEMM workspace sizing helpers.
      - SQNBIT_CompFp32 (4-bit weights): M==1 GEMV kernel + dequantize-B-for-SGEMM.
      - SQNBIT_CompInt8 (4-bit weights): QuantizeARow + int8xint4 kernel.
      - SQNBIT_CompInt8 (8-bit weights): pack-with-blksum, QuantizeARowComputeBlkSum
        and the int8xint8 BlkSum kernel.
      - HQNBIT_CompFp16 pack helpers (fp16 dequant/kernel live in
        hqnbitgemm_kernel_rvv.cpp, which requires Zvfh).

    The packed-B / block-sum layouts are private to this dispatch (produced here
    and consumed only by these kernels), so plain layouts are used throughout.

--*/

#include "mlasi.h"

#if defined(MLAS_USE_RVV)

#include <riscv_vector.h>

#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "qnbitgemm.h"
#include "sqnbitgemm_q8_block.h"

namespace
{

//
// Quantized B data packing.
//
// The packing is a pure byte-layout transform shared with the other backends
// (see the NEON implementation); it contains no architecture-specific
// intrinsics, so the RVV path reuses the same logic.
//

size_t
RvvQ4BitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    bool /*HasZeroPoint*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/,  // same size regardless of ComputeType
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG*     /*BackendKernelSelectorConfig*/
)
{
    constexpr size_t BlkBitWidth = 4;
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t PackedQuantBDataSize = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    return PackedQuantBDataSize;
}

// CompInt8 packed layouts (private to this dispatch, 4-bit and 8-bit alike).
//
// The layouts follow the vector length of the core that packs, so the kernels
// can fill their registers on any VLEN: the K dimension of every column is
// cut into chunks of ChunkElems = min(VLENB, 128) elements, which is exactly
// the number of int8 x int8 products two e8mf2 registers deliver into one
// e16m1 register. A chunk holds SegsPerChunk "segments": whole blocks when
// BlkLen <= ChunkElems, or the ChunkElems-wide slice of one block otherwise.
// Within a chunk the segments' first halves are stored back to back, then
// their second halves, so lane i of the int16 partial always pairs element i
// with element (i + SegHalf) of the same segment, and the lanes of segment t
// are [t * SegHalf, (t + 1) * SegHalf). The last chunk of a column may hold
// fewer segments.
//
// Columns are grouped in tiles of CompInt8ColTile; within a tile the chunks
// are outermost and the tile's columns are interleaved per chunk:
//   tile t at (t * CompInt8ColTile) * ldb: [ChunkCount][width][chunk bytes]
// where width = min(CompInt8ColTile, N - t * CompInt8ColTile). A full tile
// therefore occupies the same bytes as its columns would in a plain [N][ldb]
// layout, so the driver's "QuantBData + n * ldb" addressing (n a multiple of
// the tile width) still lands on the tile, and a kernel walking a tile down K
// reads one sequential stream.
//
// The quantized A rows use the same chunk order for their data (4-bit rows
// keep the block scales first, then the data; 8-bit rows are data only), so
// a chunk of A and a chunk of B line up lane for lane. On VLEN = 256 every
// layout coincides with a plain per-block one for BlkLen >= 32.
//
// The VLENB is read once and cached: a process that packs on one core and
// computes on a core with a different VLEN would otherwise disagree on the
// layout.
constexpr size_t CompInt8ColTile = 8;
constexpr size_t CompInt8MaxChunkElems = 128;
constexpr size_t CompInt8MaxBlkLen = 256;  // keeps SegsPerChunk <= 8 (masks and scales per chunk)

MLAS_FORCEINLINE size_t
CompInt8ChunkElems()
{
    static const size_t ChunkElems = std::min<size_t>(__riscv_vlenb(), CompInt8MaxChunkElems);
    return ChunkElems;
}

struct CompInt8Geometry {
    size_t BlkLen;
    size_t BlockCountK;
    size_t ChunkElems;      // elements of one column per full chunk
    size_t SegLen;          // min(BlkLen, ChunkElems)
    size_t SegHalf;         // SegLen / 2: lanes per segment
    size_t SegsPerChunk;    // ChunkElems / SegLen
    size_t ChunksPerBlock;  // BlkLen / ChunkElems, or 1
    size_t SegCount;        // segments per column
    size_t ChunkCount;      // chunks per column

    CompInt8Geometry(size_t blk_len, size_t block_count_k)
        : BlkLen(blk_len), BlockCountK(block_count_k), ChunkElems(CompInt8ChunkElems())
    {
        SegLen = std::min(BlkLen, ChunkElems);
        SegHalf = SegLen / 2;
        SegsPerChunk = ChunkElems / SegLen;
        ChunksPerBlock = std::max<size_t>(BlkLen / ChunkElems, 1);
        SegCount = BlockCountK * (BlkLen / SegLen);
        ChunkCount = MlasDivRoundup(SegCount, SegsPerChunk);
    }

    size_t SegsInChunk(size_t chunk) const
    {
        return std::min(SegsPerChunk, SegCount - chunk * SegsPerChunk);
    }

    // Offset of (block, element) within a column's dense int8 data (A rows,
    // and the per-column element order of B).
    size_t ElementOffset(size_t block, size_t element) const
    {
        const size_t seg = block * (BlkLen / SegLen) + element / SegLen;
        const size_t e = element % SegLen;
        const size_t chunk = seg / SegsPerChunk;
        const size_t t = seg % SegsPerChunk;
        const size_t segs = SegsInChunk(chunk);
        return chunk * ChunkElems + (e / SegHalf) * (segs * SegHalf) + t * SegHalf + (e % SegHalf);
    }

    // The first half of segment 'seg' of a column starts here; the second
    // half is 'SegsInChunk(chunk) * SegHalf' further on.
    size_t SegmentOffset(size_t seg) const
    {
        const size_t chunk = seg / SegsPerChunk;
        return chunk * ChunkElems + (seg % SegsPerChunk) * SegHalf;
    }
};

// SQ8 (8-bit weight) CompInt8 packed-B workspace sizing. The workspace holds
// the packed 8-bit B data, then the per-(N,block) B block-sums, then the B
// scales (matching PackedQuantBDataStruct's signed-QuantA layout). Sizes and
// alignment slack mirror the portable reference so the struct's offsets fit.
size_t
RvvQ8BitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    bool /*HasZeroPoint*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*BackendKernelSelectorConfig*/
)
{
    constexpr size_t BlkBitWidth = 8;
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    size_t PackedQuantBDataSize = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);

    if (ComputeType == SQNBIT_CompInt8) {
        const size_t ScaleSize = N * BlockCountK * sizeof(float);
        size_t BlkSumSize = MlasDivRoundup(N, 16) * BlockCountK * 16 * sizeof(float);

        constexpr size_t PackedQuantBDataAlignment = 32;
        PackedQuantBDataSize += PackedQuantBDataAlignment - 1;
        constexpr size_t BlkSumAlignment = MlasQNBitQuantBBlkSumAlignment();
        BlkSumSize += BlkSumAlignment - 1;

        return PackedQuantBDataSize + ScaleSize + BlkSumSize;
    }
    return PackedQuantBDataSize;
}

// Pack 8-bit B and compute per-block sums. The packed data, scales and
// block-sums are private to the RVV dispatch: the data uses the chunked,
// column-tiled layout described at CompInt8Geometry, scales and block-sums
// plain [N][BlockCountK]. B is stored centered (raw - 128, as int8) and
// QuantBBlkSum[n][b] = bScale * (bZeroPoint - 128) (bZeroPoint defaults to 128
// when zero points are absent); the kernel subtracts ABlockSum * this.
void
RvvSQ8BitGemmPackQuantBDataAndBlkSum(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/,
    const std::byte* QuantBDataBegin,
    const float* QuantBScaleBegin,
    bool HasZeroPoint,
    const std::byte* QuantBZPBegin,
    PackedQuantBDataStruct<float, 8>& PackedQuantB,
    MLAS_THREADPOOL* ThreadPool,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*BackendKernelSelectorConfig*/
)
{
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t DataBytesPerCol = BlockCountK * BlkLen;  // 8-bit: one byte per weight
    const CompInt8Geometry Geom(BlkLen, BlockCountK);

    std::byte* PackedData = PackedQuantB.PackedQuantBData;
    float* PackedScale = PackedQuantB.PackedQuantBScale;
    float* BlkSum = PackedQuantB.QuantBBlkSum;

    // MatMulNBits prepacks weights, scales and zero points in three separate
    // calls (each with the others null). The block sum needs both the scale and
    // the zero point, which arrive in different calls, so it is finalized from
    // the already-packed scale: with a zero point when it arrives, or with the
    // default zero point of 128 at scale-packing time when there is none.
    MlasTrySimpleParallel(
        ThreadPool, static_cast<ptrdiff_t>(N),
        [&](ptrdiff_t n) {
            const size_t row = static_cast<size_t>(n) * BlockCountK;

            // B is stored centered by 128 (int8 = raw ^ 0x80) so the kernel can
            // accumulate two int8 x int8 products in int16; the 128 is folded
            // into the block sum: BlkSum = bScale * (bZeroPoint - 128).
            if (QuantBDataBegin != nullptr) {
                const size_t tile = static_cast<size_t>(n) / CompInt8ColTile;
                const size_t col = static_cast<size_t>(n) % CompInt8ColTile;
                const size_t width = std::min(CompInt8ColTile, N - tile * CompInt8ColTile);
                const std::byte* src = QuantBDataBegin + static_cast<size_t>(n) * DataBytesPerCol;
                std::byte* PackedTile = PackedData + tile * CompInt8ColTile * DataBytesPerCol;
                const size_t SegsPerBlock = BlkLen / Geom.SegLen;
                for (size_t chunk = 0; chunk < Geom.ChunkCount; ++chunk) {
                    const size_t segs = Geom.SegsInChunk(chunk);
                    std::byte* dst = PackedTile + (chunk * Geom.ChunkElems) * width + col * (segs * Geom.SegLen);
                    for (size_t t = 0; t < segs; ++t) {
                        const size_t seg = chunk * Geom.SegsPerChunk + t;
                        const std::byte* s0 = src + (seg / SegsPerBlock) * BlkLen + (seg % SegsPerBlock) * Geom.SegLen;
                        for (size_t i = 0; i < Geom.SegHalf; ++i) {
                            dst[t * Geom.SegHalf + i] = s0[i] ^ std::byte{0x80};
                            dst[segs * Geom.SegHalf + t * Geom.SegHalf + i] = s0[Geom.SegHalf + i] ^ std::byte{0x80};
                        }
                    }
                }
            }

            if (QuantBScaleBegin != nullptr) {
                for (size_t b = 0; b < BlockCountK; ++b) {
                    const float scale = QuantBScaleBegin[row + b];
                    PackedScale[row + b] = scale;
                    if (!HasZeroPoint) {
                        BlkSum[row + b] = 0.0f;
                    }
                }
            }

            if (QuantBZPBegin != nullptr) {
                for (size_t b = 0; b < BlockCountK; ++b) {
                    const float zp = static_cast<float>(std::to_integer<uint8_t>(QuantBZPBegin[row + b]));
                    BlkSum[row + b] = PackedScale[row + b] * (zp - 128.0f);
                }
            }
        }
    );
}

#if defined(MLAS_USE_RVV_ZVFH)
// Plain 8-bit B-data packing for the HQNBIT_CompFp16 path (private to this
// dispatch; the fp16 dequant reads it as [N][BlockCountK][BlkLen] bytes).
void
RvvHQ8BitGemmPackQuantBData(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* /*ThreadPool*/,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*BackendKernelSelectorConfig*/
)
{
    if (QuantBDataBegin == nullptr) {
        return;
    }
    const size_t total = N * MlasDivRoundup(K, BlkLen) * BlkLen;  // 8-bit: one byte per weight
    std::memcpy(PackedQuantBDataBegin, QuantBDataBegin, total);
}
#endif  // MLAS_USE_RVV_ZVFH

void
RvvSQ4BitGemmPackQuantBData(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* ThreadPool,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*BackendKernelSelectorConfig*/
)
{
    constexpr size_t BlkBitWidth = 4;

    assert(BlkLen >= 16 && BlkLen % 16 == 0);

    // MatMulNBits prepacks weights, scales and zero points in separate calls;
    // this function only packs the weight data, so ignore the calls where the
    // weight data is absent.
    if (QuantBDataBegin == nullptr) {
        return;
    }

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t Iterations = N * BlockCountK;  // one iteration per block

    // The packed layouts are private to the RVV dispatch (produced here,
    // consumed only by the RVV compute kernels).
    //
    // CompInt8 uses the chunked, column-tiled layout described at
    // CompInt8Geometry, with a half-split nibble order within each segment:
    // byte i holds element i in its low nibble and element (i + SegHalf) in
    // its high nibble, so a run of bytes unpacks into two runs of elements
    // that pair with the two halves of the int8 A chunk.
    if (ComputeType == SQNBIT_CompInt8) {
        const CompInt8Geometry Geom(BlkLen, BlockCountK);
        const size_t SegsPerBlock = BlkLen / Geom.SegLen;
        MlasTrySimpleParallel(
            ThreadPool, static_cast<ptrdiff_t>(N),
            [&](ptrdiff_t tid) {
                const size_t n = static_cast<size_t>(tid);
                const size_t tile = n / CompInt8ColTile;
                const size_t col = n % CompInt8ColTile;
                const size_t width = std::min(CompInt8ColTile, N - tile * CompInt8ColTile);

                const std::byte* QuantBData = QuantBDataBegin + n * BlockCountK * BlkDataSize;
                std::byte* PackedTile = PackedQuantBDataBegin + tile * CompInt8ColTile * BlockCountK * BlkDataSize;

                // source: byte e/2 holds element e in nibble (e & 1)
                auto nibble = [](const std::byte* blk, size_t e) {
                    return (e & 1) ? (blk[e / 2] >> 4) : (blk[e / 2] & std::byte{0x0F});
                };

                for (size_t chunk = 0; chunk < Geom.ChunkCount; ++chunk) {
                    const size_t segs = Geom.SegsInChunk(chunk);
                    std::byte* dst = PackedTile + (chunk * Geom.ChunkElems / 2) * width + col * (segs * Geom.SegHalf);
                    for (size_t t = 0; t < segs; ++t) {
                        const size_t seg = chunk * Geom.SegsPerChunk + t;
                        const std::byte* blk = QuantBData + (seg / SegsPerBlock) * BlkDataSize;
                        const size_t e0 = (seg % SegsPerBlock) * Geom.SegLen;
                        // byte i of the segment: element (e0 + i) low, element (e0 + SegHalf + i) high
                        for (size_t i = 0; i < Geom.SegHalf; ++i) {
                            dst[t * Geom.SegHalf + i] = nibble(blk, e0 + i) | (nibble(blk, e0 + Geom.SegHalf + i) << 4);
                        }
                    }
                }
            }
        );
        return;
    }

    // CompFp32 / CompFp16: SubBlkLen == 16 interleaved layout.
    const size_t SubBlkLen = 16;

    const size_t SubBlkDataSize = SubBlkLen / 2;
    const size_t SubBlkBytePairCount = SubBlkLen / 4;

    //
    // For SubBlkLen == 16, pack 16 4-bit values (8 bytes) at a time like this:
    //
    // src: | v0 v1 | v2 v3 | v4 v5 | v6 v7 | v8 v9 | vA vB | vC vD | vE vF |
    //   =>
    // dst: | v0 v8 | v1 v9 | v2 vA | v3 vB | v4 vC | v5 vD | v6 vE | v7 vF |
    //

    MlasTrySimpleParallel(
        ThreadPool, Iterations,
        [&](ptrdiff_t tid) {
            const size_t n = tid / BlockCountK;
            const size_t k_blk = tid % BlockCountK;

            const size_t data_offset = n * BlockCountK * BlkDataSize + k_blk * BlkDataSize;
            const std::byte* QuantBData = QuantBDataBegin + data_offset;
            std::byte* PackedQuantBData = PackedQuantBDataBegin + data_offset;

            for (size_t kk = 0; kk < BlkLen; kk += SubBlkLen) {
                for (size_t byte_pair_idx = 0; byte_pair_idx < SubBlkBytePairCount; ++byte_pair_idx) {
                    const std::byte src0 = QuantBData[byte_pair_idx];
                    const std::byte src1 = QuantBData[byte_pair_idx + SubBlkDataSize / 2];

                    std::byte& dst0 = PackedQuantBData[2 * byte_pair_idx];
                    std::byte& dst1 = PackedQuantBData[2 * byte_pair_idx + 1];

                    dst0 = (src0 & std::byte{0x0F}) | ((src1 & std::byte{0x0F}) << 4);
                    dst1 = (src0 >> 4) | ((src1 >> 4) << 4);
                }

                QuantBData += SubBlkDataSize;
                PackedQuantBData += SubBlkDataSize;
            }
        }
    );
}

//
// Per-GEMM intermediate workspace sizing.
//
// The CompInt8 path uses the workspace to hold the block-quantized int8 copy
// of A (data + scale + block-sum). This sizing is architecture-independent.
//

size_t
RvvQNBitGemmPerGemmWorkspaceSize(
    size_t M,
    size_t N,
    size_t K,
    size_t BlkLen,
    bool /*HasZeroPoint*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    size_t /*BlkBitWidth*/,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*BackendKernelSelectorConfig*/
)
{
    MLAS_UNREFERENCED_PARAMETER(N);

    switch (ComputeType) {
        case SQNBIT_CompInt8: {
            // workspace buffer is used for block quantization of A to int8
            const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
            // QuantData + Scale + BlkSum
            const size_t PerGemmWorkspaceSize = M * BlockCountK * (Q8BlkSize(BlkLen) + sizeof(float));
            return PerGemmWorkspaceSize;
        }
        default: {
            return 0;
        }
    }
}

size_t
RvvQNBitGemmPerGemmWorkspaceAlignment(
    size_t /*BlkLen*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    switch (ComputeType) {
        case SQNBIT_CompInt8: {
            return Q8BlkAlignment();
        }
        default: {
            return 1;
        }
    }
}

#if defined(MLAS_USE_RVV)

//
// SQNBIT_CompFp32 kernels for 4-bit weights.
//
// Both kernels consume the packed B produced by RvvSQ4BitGemmPackQuantBData
// with SubBlkLen == 16: within each 16-element sub-block (8 bytes), byte b
// holds element b in its low nibble and element (b + 8) in its high nibble.
// Dequantization is value = (nibble - offset) * scale, where offset is the
// block's zero point, or 8 when zero points are not provided.
//

constexpr size_t SubBlkLen = 16;

// Dequantize one sub-block (up to 16 elements) of a single B column into a
// natural-order float buffer. `packed` points at the sub-block's bytes;
// `len` valid elements are written to `out[0..len-1]`.
MLAS_FORCEINLINE void
DequantSubblockToFloat(
    const uint8_t* packed,
    size_t len,
    float offset,
    float scale,
    float* out
)
{
    const size_t low_count = std::min(len, SubBlkLen / 2);

    // low nibbles -> elements [0, low_count)
    {
        const size_t vl = __riscv_vsetvl_e8m1(low_count);
        const vuint8m1_t b = __riscv_vle8_v_u8m1(packed, vl);
        const vuint8m1_t nib = __riscv_vand_vx_u8m1(b, 0x0F, vl);
        const vuint16m2_t w16 = __riscv_vzext_vf2_u16m2(nib, vl);
        const vuint32m4_t w32 = __riscv_vzext_vf2_u32m4(w16, vl);
        vfloat32m4_t f = __riscv_vfcvt_f_xu_v_f32m4(w32, vl);
        f = __riscv_vfsub_vf_f32m4(f, offset, vl);
        f = __riscv_vfmul_vf_f32m4(f, scale, vl);
        __riscv_vse32_v_f32m4(out, f, vl);
    }

    // high nibbles -> elements [8, len)
    if (len > SubBlkLen / 2) {
        const size_t high_count = len - SubBlkLen / 2;
        const size_t vl = __riscv_vsetvl_e8m1(high_count);
        const vuint8m1_t b = __riscv_vle8_v_u8m1(packed, vl);
        const vuint8m1_t nib = __riscv_vsrl_vx_u8m1(b, 4, vl);
        const vuint16m2_t w16 = __riscv_vzext_vf2_u16m2(nib, vl);
        const vuint32m4_t w32 = __riscv_vzext_vf2_u32m4(w16, vl);
        vfloat32m4_t f = __riscv_vfcvt_f_xu_v_f32m4(w32, vl);
        f = __riscv_vfsub_vf_f32m4(f, offset, vl);
        f = __riscv_vfmul_vf_f32m4(f, scale, vl);
        __riscv_vse32_v_f32m4(out + SubBlkLen / 2, f, vl);
    }
}

// Extract the 4-bit zero point for block `blk_idx` of a single column.
MLAS_FORCEINLINE float
DequantOffset(const std::byte* zp_col, size_t blk_idx, bool has_zero_point)
{
    if (!has_zero_point) {
        return 8.0f;
    }
    const std::byte zp_packed = zp_col[blk_idx / 2];
    const uint8_t zp = ((blk_idx & 1) == 1)
                           ? std::to_integer<uint8_t>(zp_packed >> 4)
                           : std::to_integer<uint8_t>(zp_packed & std::byte{0x0F});
    return static_cast<float>(zp);
}

// Compute the dot product of one A row with one dequantized B column.
template <bool HasZeroPoint>
MLAS_FORCEINLINE float
ComputeColumnDot_CompFp32(
    size_t BlkLen,
    const float* ARow,
    const uint8_t* QuantBDataCol,
    const float* QuantBScaleCol,
    const std::byte* QuantBZeroPointCol,
    size_t CountK
)
{
    constexpr size_t BlkBitWidth = 4;
    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);

    const size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);

    float bdeq[SubBlkLen];

    size_t blk_idx = 0;
    for (size_t k = 0; k < CountK; k += BlkLen, ++blk_idx) {
        const float scale = QuantBScaleCol[blk_idx];
        const float offset = DequantOffset(QuantBZeroPointCol, blk_idx, HasZeroPoint);
        const size_t k_blk_len = std::min(CountK - k, BlkLen);
        const uint8_t* blk_ptr = QuantBDataCol + blk_idx * BlkDataSize;

        for (size_t kk = 0; kk < k_blk_len; kk += SubBlkLen) {
            const size_t len = std::min(k_blk_len - kk, SubBlkLen);
            DequantSubblockToFloat(blk_ptr + kk / 2, len, offset, scale, bdeq);

            const float* a_ptr = ARow + k + kk;
            for (size_t off = 0; off < len;) {
                const size_t vl = __riscv_vsetvl_e32m1(len - off);
                const vfloat32m1_t av = __riscv_vle32_v_f32m1(a_ptr + off, vl);
                const vfloat32m1_t bv = __riscv_vle32_v_f32m1(bdeq + off, vl);
                // tail-undisturbed: preserve accumulator lanes [vl, vlmax)
                acc = __riscv_vfmacc_vv_f32m1_tu(acc, av, bv, vl);
                off += vl;
            }
        }
    }

    vfloat32m1_t red = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    red = __riscv_vfredusum_vs_f32m1_f32m1(acc, red, vlmax);
    return __riscv_vfmv_f_s_f32m1_f32(red);
}

template <bool HasZeroPoint>
void
SQ4BitGemmM1Kernel_CompFp32_Impl(
    size_t BlkLen,
    const float* A,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias
)
{
    constexpr size_t BlkBitWidth = 4;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    for (size_t n = 0; n < CountN; ++n) {
        const uint8_t* b_data = reinterpret_cast<const uint8_t*>(QuantBData) + n * StrideQuantBData;
        const float* b_scale = QuantBScale + n * StrideQuantBScale;
        const std::byte* b_zp =
            HasZeroPoint ? QuantBZeroPoint + n * StrideQuantBZeroPoint : nullptr;

        float dot = ComputeColumnDot_CompFp32<HasZeroPoint>(
            BlkLen, A, b_data, b_scale, b_zp, CountK
        );

        if (Bias != nullptr) {
            dot += Bias[n];
        }
        C[n] = dot;
    }
}

// Zero `count` floats starting at `p`.
MLAS_FORCEINLINE void
ZeroFloats(float* p, size_t count)
{
    const size_t vlmax = __riscv_vsetvlmax_e32m1();
    const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    for (size_t off = 0; off < count;) {
        const size_t vl = __riscv_vsetvl_e32m1(count - off);
        __riscv_vse32_v_f32m1(p + off, zero, vl);
        off += vl;
    }
}

template <bool HasZeroPoint>
void
Q4BitBlkDequantBForSgemm_CompFp32_Impl(
    size_t BlkLen,
    float* FpData,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK
)
{
    constexpr size_t BlkBitWidth = 4;
    constexpr size_t PackWidth = 16;  // SGEMM CopyPackB column-panel width

    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBData = BlockCountK * BlkDataSize;
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    // Destination layout matches "dequantize B then MlasSgemmCopyPackB": B is
    // stored as PackWidth-column panels, K rows each; element (k, n_local) of
    // panel g lives at FpData[g * PackWidth * CountK + k * PackWidth + n_local].
    const ptrdiff_t DstColStride = static_cast<ptrdiff_t>(PackWidth * sizeof(float));

    float bdeq[SubBlkLen];

    size_t panel = 0;
    for (size_t n0 = 0; n0 < CountN; n0 += PackWidth, ++panel) {
        float* panel_base = FpData + panel * PackWidth * CountK;
        const size_t n_cols = std::min(CountN - n0, PackWidth);

        // Unused columns of a partial panel must read as zero for SGEMM.
        if (n_cols < PackWidth) {
            ZeroFloats(panel_base, PackWidth * CountK);
        }

        for (size_t nl = 0; nl < n_cols; ++nl) {
            const size_t n = n0 + nl;
            const uint8_t* b_data = reinterpret_cast<const uint8_t*>(QuantBData) + n * StrideQuantBData;
            const float* b_scale = QuantBScale + n * StrideQuantBScale;
            const std::byte* b_zp =
                HasZeroPoint ? QuantBZeroPoint + n * StrideQuantBZeroPoint : nullptr;

            size_t blk_idx = 0;
            for (size_t k = 0; k < CountK; k += BlkLen, ++blk_idx) {
                const float scale = b_scale[blk_idx];
                const float offset = DequantOffset(b_zp, blk_idx, HasZeroPoint);
                const size_t k_blk_len = std::min(CountK - k, BlkLen);
                const uint8_t* blk_ptr = b_data + blk_idx * BlkDataSize;

                for (size_t kk = 0; kk < k_blk_len; kk += SubBlkLen) {
                    const size_t len = std::min(k_blk_len - kk, SubBlkLen);
                    DequantSubblockToFloat(blk_ptr + kk / 2, len, offset, scale, bdeq);

                    // Scatter the sub-block down column `nl` with panel stride.
                    float* dst = panel_base + (k + kk) * PackWidth + nl;
                    for (size_t off = 0; off < len;) {
                        const size_t vl = __riscv_vsetvl_e32m1(len - off);
                        const vfloat32m1_t v = __riscv_vle32_v_f32m1(bdeq + off, vl);
                        __riscv_vsse32_v_f32m1(dst + off * PackWidth, DstColStride, v, vl);
                        off += vl;
                    }
                }
            }
        }
    }
}

//
// SQNBIT_CompInt8 kernels for 4-bit weights.
//
// A is block-quantized to int8 (row: [BlockCountK scales][data in chunk
// order], see RvvQuantizeARow_CompInt8_Impl); B is the packed 4-bit layout
// above. For each block the integer dot
// sum_i qa_i * (qb_i - offset) is computed exactly, then scaled by
// (a_scale * b_scale) and accumulated across blocks.
//

// Quantize one block of A to int8 into 'out' (zero-padded to BlkLen) and
// return its scale (amax / 127).
MLAS_FORCEINLINE float
QuantizeABlock(const float* a_ptr, size_t len, size_t BlkLen, int8_t* out)
{
    vfloat32m1_t vmax = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    for (size_t off = 0; off < len;) {
        const size_t vl = __riscv_vsetvl_e32m1(len - off);
        const vfloat32m1_t v = __riscv_vfabs_v_f32m1(__riscv_vle32_v_f32m1(a_ptr + off, vl), vl);
        vmax = __riscv_vfredmax_vs_f32m1_f32m1(v, vmax, vl);
        off += vl;
    }
    const float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);
    const float scale = amax / 127.0f;
    const float inv_scale = (amax != 0.0f) ? (127.0f / amax) : 0.0f;

    // q = clamp(round(a * inv_scale), -127, 127)
    for (size_t off = 0; off < len;) {
        const size_t vl = __riscv_vsetvl_e32m4(len - off);
        vfloat32m4_t v = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(a_ptr + off, vl), inv_scale, vl);
        vint32m4_t iv = __riscv_vfcvt_x_f_v_i32m4(v, vl);
        iv = __riscv_vmax_vx_i32m4(iv, -127, vl);
        iv = __riscv_vmin_vx_i32m4(iv, 127, vl);
        const vint16m2_t i16 = __riscv_vncvt_x_x_w_i16m2(iv, vl);
        __riscv_vse8_v_i8m1(out + off, __riscv_vncvt_x_x_w_i8m1(i16, vl), vl);
        off += vl;
    }
    for (size_t i = len; i < BlkLen; ++i) {
        out[i] = 0;
    }
    return scale;
}

// Store a quantized block into a row's data area in the chunk order described
// at CompInt8Geometry: each segment's first half, then (after the chunk's
// other first halves) its second half.
MLAS_FORCEINLINE void
StoreABlock(const CompInt8Geometry& Geom, size_t block, const int8_t* q, int8_t* data)
{
    const size_t SegsPerBlock = Geom.BlkLen / Geom.SegLen;
    for (size_t s = 0; s < SegsPerBlock; ++s) {
        const size_t seg = block * SegsPerBlock + s;
        const size_t off = Geom.SegmentOffset(seg);
        const size_t segs = Geom.SegsInChunk(seg / Geom.SegsPerChunk);
        std::memcpy(data + off, q + s * Geom.SegLen, Geom.SegHalf);
        std::memcpy(data + off + segs * Geom.SegHalf, q + s * Geom.SegLen + Geom.SegHalf, Geom.SegHalf);
    }
}

// 4-bit path A row: [BlockCountK float scales][BlockCountK * BlkLen int8 in chunk order];
// the same bytes as BlockCountK Q8 blocks, which is the driver's row stride.
void
RvvQuantizeARow_CompInt8_Impl(size_t BlkLen, const float* A, size_t CountK, std::byte* QuantA)
{
    const size_t BlockCountK = MlasDivRoundup(CountK, BlkLen);
    const CompInt8Geometry Geom(BlkLen, BlockCountK);
    float* scales = reinterpret_cast<float*>(QuantA);
    int8_t* data = reinterpret_cast<int8_t*>(QuantA + BlockCountK * sizeof(float));

    int8_t q[CompInt8MaxBlkLen];
    for (size_t b = 0; b < BlockCountK; ++b) {
        const size_t k0 = b * BlkLen;
        scales[b] = QuantizeABlock(A + k0, std::min(BlkLen, CountK - k0), BlkLen, q);
        StoreABlock(Geom, b, q, data);
    }
}

//
// CompInt8 tile helpers.
//
// The K-reduction of one block runs in int16 lanes: each chunk of the block
// is two 'vl'-element halves, multiplied int8 x int8 and accumulated with a
// widening multiply-add. The int16 partial is widened to float once per block
// and folded into a float vector accumulator with the block's combined scale.
// A single vfredusum per output then finishes the row, instead of one vwredsum
// per (row, block).
//
// int16 is safe because the operands are bounded: 4-bit B is centered to
// [-8, 7] so |a*b| <= 1016 and a whole block of up to 256 elements fits; 8-bit
// B is stored centered to [-128, 127] so |a*b| <= 16256 and one pair fits,
// which is exactly one chunk.
//
// Two tile shapes cover the two chunk regimes of CompInt8Geometry:
//  - BlkLen >= ChunkElems (one segment per chunk): MTILE x NTILE tiles share
//    the B unpack across rows and the A loads across columns, and fold with a
//    scalar scale.
//  - BlkLen < ChunkElems (several blocks per chunk): a 1 x 4 tile folds each
//    chunk with one masked vfmacc per block, the mask selecting the block's
//    lanes. That is what lets a wide register carry several small blocks.
// Tile slots past the valid row/column count alias slot 0 and are not stored,
// so one body serves the full tile and the remainder. A blocks are zero-padded
// to BlkLen by the quantizer, so a chunk can always run at its full width.
//

#define MLAS_UNROLL_LOOP _Pragma("GCC unroll 8")
#define MLAS_SCHED_BARRIER asm volatile("" ::: "memory");

MLAS_FORCEINLINE float
ReduceSumF32M2(vfloat32m2_t v, size_t vl)
{
    const vfloat32m1_t z = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(v, z, vl));
}

// acc += scale * (float)part, lane-wise.
#define QI8_FOLD(acc, part, scale, vl) \
    (acc) = __riscv_vfmacc_vf_f32m2_tu((acc), (scale), __riscv_vfwcvt_f_x_v_f32m2((part), (vl)), (vl))

// Multi-block fold. seg_id holds each lane's segment index (lane / SegHalf);
// the mask of segment g is one compare away, which is cheaper than holding
// eight masks: v0 is the only mask register, and moving a mask into it costs
// as much as the compare on this hardware.
#define QI8_SEG_ID(seg_id, SegHalf, vl) \
    const vuint16m1_t seg_id = __riscv_vsrl_vx_u16m1(__riscv_vid_v_u16m1((vl)), __builtin_ctzl(SegHalf), (vl))

// acc_c += (as[b0 + g] * bs_c[b0 + g]) * f_c on the lanes of segment g, for
// four columns at once: one mask per segment, four independent chains.
#define QI8_FOLD_SEG4(g, seg_id, A0, A1, A2, A3, F0, F1, F2, F3, S0, S1, S2, S3) \
    if (segs > (g)) {                                                            \
        const vbool16_t m_ = __riscv_vmseq_vx_u16m1_b16(seg_id, (g), vl);        \
        const float a_ = as[b0 + (g)];                                           \
        A0 = __riscv_vfmacc_vf_f32m2_tumu(m_, A0, a_ * S0[b0 + (g)], F0, vl);    \
        A1 = __riscv_vfmacc_vf_f32m2_tumu(m_, A1, a_ * S1[b0 + (g)], F1, vl);    \
        A2 = __riscv_vfmacc_vf_f32m2_tumu(m_, A2, a_ * S2[b0 + (g)], F2, vl);    \
        A3 = __riscv_vfmacc_vf_f32m2_tumu(m_, A3, a_ * S3[b0 + (g)], F3, vl);    \
    }
#define QI8_FOLD_SEGS4(seg_id, A0, A1, A2, A3, F0, F1, F2, F3, S0, S1, S2, S3) \
    QI8_FOLD_SEG4(0, seg_id, A0, A1, A2, A3, F0, F1, F2, F3, S0, S1, S2, S3)   \
    QI8_FOLD_SEG4(1, seg_id, A0, A1, A2, A3, F0, F1, F2, F3, S0, S1, S2, S3)   \
    QI8_FOLD_SEG4(2, seg_id, A0, A1, A2, A3, F0, F1, F2, F3, S0, S1, S2, S3)   \
    QI8_FOLD_SEG4(3, seg_id, A0, A1, A2, A3, F0, F1, F2, F3, S0, S1, S2, S3)   \
    QI8_FOLD_SEG4(4, seg_id, A0, A1, A2, A3, F0, F1, F2, F3, S0, S1, S2, S3)   \
    QI8_FOLD_SEG4(5, seg_id, A0, A1, A2, A3, F0, F1, F2, F3, S0, S1, S2, S3)   \
    QI8_FOLD_SEG4(6, seg_id, A0, A1, A2, A3, F0, F1, F2, F3, S0, S1, S2, S3)   \
    QI8_FOLD_SEG4(7, seg_id, A0, A1, A2, A3, F0, F1, F2, F3, S0, S1, S2, S3)

// v = per-lane 4-bit zero-point offsets for a chunk holding 'segs' blocks.
#define QI8_SEG_OFFSETS(v, offset, segs, seg_id, vl)                                                               \
    do {                                                                                                           \
        (v) = __riscv_vmv_v_x_i8mf2((offset)[0], (vl));                                                            \
        for (size_t g_ = 1; g_ < (segs); ++g_) {                                                                   \
            (v) = __riscv_vmerge_vxm_i8mf2((v), (offset)[g_], __riscv_vmseq_vx_u16m1_b16(seg_id, g_, (vl)), (vl)); \
        }                                                                                                          \
    } while (0)

// 4-bit A row accessors (see RvvQuantizeARow_CompInt8_Impl).
MLAS_FORCEINLINE const float*
QuantARowScales(const std::byte* row)
{
    return reinterpret_cast<const float*>(row);
}

MLAS_FORCEINLINE const int8_t*
QuantARowData(const std::byte* row, size_t BlockCountK)
{
    return reinterpret_cast<const int8_t*>(row + BlockCountK * sizeof(float));
}

// MTILE rows x NTILE columns of the SQ4 CompInt8 GEMM, BlkLen >= ChunkElems.
// Columns >= n_cols alias column 0.
template <bool HasZeroPoint, size_t MTILE, size_t NTILE>
MLAS_FORCEINLINE void
SQ4BitGemmKernel_CompInt8_Tile(
    const CompInt8Geometry& Geom,
    const std::byte* QuantA,  // first row of the tile
    size_t lda,
    const uint8_t* QuantBData,         // chunk 0 of the tile's first column
    size_t Width,                      // columns interleaved in the packed tile
    const float* QuantBScale,          // first column
    const std::byte* QuantBZeroPoint,  // first column, or nullptr
    size_t StrideQuantBZeroPoint,
    size_t n_cols,
    float* C,  // C[row 0][col 0]
    size_t ldc,
    const float* Bias  // Bias[col 0], or nullptr
)
{
    static_assert(MTILE == 1 || MTILE == 2, "unsupported MTILE");
    static_assert(NTILE == 4 || NTILE == 8, "unsupported NTILE");
    static_assert(MTILE * NTILE <= 8, "too many accumulators");
    const size_t BlockCountK = Geom.BlockCountK;
    const size_t SegHalf = Geom.SegHalf;  // == ChunkElems / 2 in this regime
    const size_t ChunksPerBlock = Geom.ChunksPerBlock;
    const size_t ChunkStride = Width * SegHalf;  // packed bytes per chunk of the tile
    const size_t accvl = __riscv_vsetvlmax_e32m2();
    const size_t partvl = __riscv_vsetvlmax_e16m1();

    const uint8_t* b_data[NTILE];  // adjacent columns of a chunk are SegHalf apart
    const float* b_scale[NTILE];
    const std::byte* b_zp[NTILE];
    for (size_t c = 0; c < NTILE; ++c) {
        const size_t cc = (c < n_cols) ? c : 0;
        b_data[c] = QuantBData + cc * SegHalf;
        b_scale[c] = QuantBScale + cc * BlockCountK;
        b_zp[c] = HasZeroPoint ? QuantBZeroPoint + cc * StrideQuantBZeroPoint : nullptr;
    }

    const float* as_row0 = QuantARowScales(QuantA);
    const int8_t* qa_row0 = QuantARowData(QuantA, BlockCountK);
    const float* as_row1 = as_row0;
    const int8_t* qa_row1 = qa_row0;
    if constexpr (MTILE > 1) {
        as_row1 = QuantARowScales(QuantA + lda);
        qa_row1 = QuantARowData(QuantA + lda, BlockCountK);
    }

    vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0.0f, accvl);
    vfloat32m2_t acc1 = acc0, acc2 = acc0, acc3 = acc0, acc4 = acc0, acc5 = acc0, acc6 = acc0, acc7 = acc0;

    for (size_t b = 0; b < BlockCountK; ++b) {
        const float as0 = as_row0[b];
        const float as1 = as_row1[b];

        int8_t offset[NTILE];
        float s0[NTILE], s1[NTILE];
        MLAS_UNROLL_LOOP
        for (size_t c = 0; c < NTILE; ++c) {
            offset[c] = static_cast<int8_t>(HasZeroPoint ? static_cast<int>(DequantOffset(b_zp[c], b, true)) : 8);
            s0[c] = as0 * b_scale[c][b];
            s1[c] = as1 * b_scale[c][b];
        }

        // int16 partials, one per tile slot, accumulated over the block's chunks.
        vint16m1_t p0 = __riscv_vmv_v_x_i16m1(0, partvl);
        vint16m1_t p1 = p0, p2 = p0, p3 = p0, p4 = p0, p5 = p0, p6 = p0, p7 = p0;

        for (size_t sub = 0; sub < ChunksPerBlock; ++sub) {
            const size_t chunk = b * ChunksPerBlock + sub;
            const size_t sb = chunk * ChunkStride;      // chunk offset within the packed tile
            const size_t sa = chunk * Geom.ChunkElems;  // chunk offset within the A row data
            for (size_t k = 0; k < SegHalf;) {
                const size_t vl = __riscv_vsetvl_e8mf2(SegHalf - k);

                const vint8mf2_t a0_lo = __riscv_vle8_v_i8mf2(qa_row0 + sa + k, vl);
                const vint8mf2_t a0_hi = __riscv_vle8_v_i8mf2(qa_row0 + sa + SegHalf + k, vl);
                vint8mf2_t a1_lo = a0_lo, a1_hi = a0_hi;
                if constexpr (MTILE > 1) {
                    a1_lo = __riscv_vle8_v_i8mf2(qa_row1 + sa + k, vl);
                    a1_hi = __riscv_vle8_v_i8mf2(qa_row1 + sa + SegHalf + k, vl);
                }

#define SQ4_COL(c, p_r0, p_r1)                                                                         \
    if constexpr ((c) < NTILE) {                                                                       \
        const vuint8mf2_t packed = __riscv_vle8_v_u8mf2(b_data[c] + sb + k, vl);                       \
        const vint8mf2_t b_lo = __riscv_vsub_vx_i8mf2(                                                 \
            __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(packed, 0x0F, vl)), offset[c], vl \
        );                                                                                             \
        const vint8mf2_t b_hi = __riscv_vsub_vx_i8mf2(                                                 \
            __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vsrl_vx_u8mf2(packed, 4, vl)), offset[c], vl    \
        );                                                                                             \
        p_r0 = __riscv_vwmacc_vv_i16m1_tu(p_r0, a0_lo, b_lo, vl);                                      \
        p_r0 = __riscv_vwmacc_vv_i16m1_tu(p_r0, a0_hi, b_hi, vl);                                      \
        if constexpr (MTILE > 1) {                                                                     \
            p_r1 = __riscv_vwmacc_vv_i16m1_tu(p_r1, a1_lo, b_lo, vl);                                  \
            p_r1 = __riscv_vwmacc_vv_i16m1_tu(p_r1, a1_hi, b_hi, vl);                                  \
        }                                                                                              \
    }

                if constexpr (MTILE == 1) {
                    SQ4_COL(0, p0, p0)
                    SQ4_COL(1, p1, p1) SQ4_COL(2, p2, p2) SQ4_COL(3, p3, p3)
                        SQ4_COL(4, p4, p4) SQ4_COL(5, p5, p5) SQ4_COL(6, p6, p6) SQ4_COL(7, p7, p7)
                } else {
                    SQ4_COL(0, p0, p4)
                    SQ4_COL(1, p1, p5) SQ4_COL(2, p2, p6) SQ4_COL(3, p3, p7)
                }
#undef SQ4_COL

                k += vl;
            }
        }

        if constexpr (MTILE == 1) {
            QI8_FOLD(acc0, p0, s0[0], partvl);
            QI8_FOLD(acc1, p1, s0[1], partvl);
            QI8_FOLD(acc2, p2, s0[2], partvl);
            QI8_FOLD(acc3, p3, s0[3], partvl);
            if constexpr (NTILE == 8) {
                QI8_FOLD(acc4, p4, s0[4], partvl);
                QI8_FOLD(acc5, p5, s0[5], partvl);
                QI8_FOLD(acc6, p6, s0[6], partvl);
                QI8_FOLD(acc7, p7, s0[7], partvl);
            }
        } else {
            QI8_FOLD(acc0, p0, s0[0], partvl);
            QI8_FOLD(acc1, p1, s0[1], partvl);
            QI8_FOLD(acc2, p2, s0[2], partvl);
            QI8_FOLD(acc3, p3, s0[3], partvl);
            QI8_FOLD(acc4, p4, s1[0], partvl);
            QI8_FOLD(acc5, p5, s1[1], partvl);
            QI8_FOLD(acc6, p6, s1[2], partvl);
            QI8_FOLD(acc7, p7, s1[3], partvl);
        }
    }

    auto store = [&](size_t r, size_t c, vfloat32m2_t acc) {
        if (c < n_cols) {
            C[r * ldc + c] = ReduceSumF32M2(acc, accvl) + (Bias ? Bias[c] : 0.0f);
        }
    };
    if constexpr (MTILE == 1) {
        store(0, 0, acc0);
        store(0, 1, acc1);
        store(0, 2, acc2);
        store(0, 3, acc3);
        if constexpr (NTILE == 8) {
            store(0, 4, acc4);
            store(0, 5, acc5);
            store(0, 6, acc6);
            store(0, 7, acc7);
        }
    } else {
        store(0, 0, acc0);
        store(0, 1, acc1);
        store(0, 2, acc2);
        store(0, 3, acc3);
        store(1, 0, acc4);
        store(1, 1, acc5);
        store(1, 2, acc6);
        store(1, 3, acc7);
    }
}

// One row x 8 columns (a whole packed tile) of the SQ4 CompInt8 GEMM,
// BlkLen < ChunkElems: every chunk holds SegsPerChunk blocks and is folded
// with one masked vfmacc per block and column. The tile is walked in one
// sequential pass; the columns are processed in two groups of four so the
// float partials fit the register file. Columns >= n_cols alias column 0.
template <bool HasZeroPoint>
MLAS_FORCEINLINE void
SQ4BitGemmKernel_CompInt8_MultiBlockTile(
    const CompInt8Geometry& Geom,
    const std::byte* QuantA,
    const uint8_t* QuantBData,         // chunk 0 of the packed tile
    size_t Width,                      // columns interleaved in the packed tile
    const float* QuantBScale,          // first column
    const std::byte* QuantBZeroPoint,  // first column, or nullptr
    size_t StrideQuantBZeroPoint,
    size_t n_cols,
    float* C,
    const float* Bias
)
{
    constexpr size_t NTILE = CompInt8ColTile;
    const size_t BlockCountK = Geom.BlockCountK;
    const size_t SegHalf = Geom.SegHalf;  // BlkLen / 2
    const size_t ChunkStride = Width * Geom.ChunkElems / 2;
    const size_t accvl = __riscv_vsetvlmax_e32m2();
    QI8_SEG_ID(seg_id, SegHalf, accvl);

    size_t b_col[NTILE];  // column index within the packed tile
    const float* bs[NTILE];
    const std::byte* b_zp[NTILE];
    for (size_t c = 0; c < NTILE; ++c) {
        const size_t cc = (c < n_cols) ? c : 0;
        b_col[c] = cc;
        bs[c] = QuantBScale + cc * BlockCountK;
        b_zp[c] = HasZeroPoint ? QuantBZeroPoint + cc * StrideQuantBZeroPoint : nullptr;
    }
    const float* as = QuantARowScales(QuantA);
    const int8_t* qa = QuantARowData(QuantA, BlockCountK);

    vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0.0f, accvl);
    vfloat32m2_t acc1 = acc0, acc2 = acc0, acc3 = acc0, acc4 = acc0, acc5 = acc0, acc6 = acc0, acc7 = acc0;

    for (size_t chunk = 0; chunk < Geom.ChunkCount; ++chunk) {
        const size_t segs = Geom.SegsInChunk(chunk);
        const size_t b0 = chunk * Geom.SegsPerChunk;  // first block of the chunk
        const size_t vl = __riscv_vsetvl_e8mf2(segs * SegHalf);
        const uint8_t* b_chunk = QuantBData + chunk * ChunkStride;
        const int8_t* a_chunk = qa + chunk * Geom.ChunkElems;

        const vint8mf2_t a_lo = __riscv_vle8_v_i8mf2(a_chunk, vl);
        const vint8mf2_t a_hi = __riscv_vle8_v_i8mf2(a_chunk + vl, vl);

        // Unscaled float partial of the chunk for column c.
#define SQ4_MBCOL(c, f)                                                                                      \
    vfloat32m2_t f;                                                                                          \
    {                                                                                                        \
        const vuint8mf2_t packed = __riscv_vle8_v_u8mf2(b_chunk + b_col[c] * vl, vl);                        \
        const vint8mf2_t n_lo = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(packed, 0x0F, vl)); \
        const vint8mf2_t n_hi = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vsrl_vx_u8mf2(packed, 4, vl));    \
        vint8mf2_t b_lo, b_hi;                                                                               \
        if constexpr (HasZeroPoint) {                                                                        \
            int8_t offset[8] = {};                                                                           \
            for (size_t g = 0; g < segs; ++g) {                                                              \
                offset[g] = static_cast<int8_t>(static_cast<int>(DequantOffset(b_zp[c], b0 + g, true)));     \
            }                                                                                                \
            vint8mf2_t offv;                                                                                 \
            QI8_SEG_OFFSETS(offv, offset, segs, seg_id, vl);                                                 \
            b_lo = __riscv_vsub_vv_i8mf2(n_lo, offv, vl);                                                    \
            b_hi = __riscv_vsub_vv_i8mf2(n_hi, offv, vl);                                                    \
        } else {                                                                                             \
            b_lo = __riscv_vsub_vx_i8mf2(n_lo, 8, vl);                                                       \
            b_hi = __riscv_vsub_vx_i8mf2(n_hi, 8, vl);                                                       \
        }                                                                                                    \
        vint16m1_t p = __riscv_vwmul_vv_i16m1(a_lo, b_lo, vl);                                               \
        p = __riscv_vwmacc_vv_i16m1(p, a_hi, b_hi, vl);                                                      \
        f = __riscv_vfwcvt_f_x_v_f32m2(p, vl);                                                               \
    }
        {
            SQ4_MBCOL(0, f0)
            SQ4_MBCOL(1, f1) SQ4_MBCOL(2, f2) SQ4_MBCOL(3, f3)
                QI8_FOLD_SEGS4(seg_id, acc0, acc1, acc2, acc3, f0, f1, f2, f3, bs[0], bs[1], bs[2], bs[3])
        }
        {
            SQ4_MBCOL(4, f4)
            SQ4_MBCOL(5, f5) SQ4_MBCOL(6, f6) SQ4_MBCOL(7, f7)
                QI8_FOLD_SEGS4(seg_id, acc4, acc5, acc6, acc7, f4, f5, f6, f7, bs[4], bs[5], bs[6], bs[7])
        }
#undef SQ4_MBCOL
    }

    auto store = [&](size_t c, vfloat32m2_t acc) {
        if (c < n_cols) {
            C[c] = ReduceSumF32M2(acc, accvl) + (Bias ? Bias[c] : 0.0f);
        }
    };
    store(0, acc0);
    store(1, acc1);
    store(2, acc2);
    store(3, acc3);
    store(4, acc4);
    store(5, acc5);
    store(6, acc6);
    store(7, acc7);
}

template <bool HasZeroPoint>
size_t
SQ4BitGemmKernel_CompInt8_Impl(
    size_t BlkLen,
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    size_t ldc,
    const float* Bias
)
{
    constexpr size_t BlkBitWidth = 4;
    MLAS_UNREFERENCED_PARAMETER(CountK);

    const CompInt8Geometry Geom(BlkLen, BlockCountK);
    const size_t lda = BlockCountK * Q8BlkSize(BlkLen);
    const size_t ldb = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);
    const uint8_t* b_data = reinterpret_cast<const uint8_t*>(QuantBData);

    // Walk the column tiles of the packed layout (see CompInt8ColTile). The
    // driver always hands us a tile-aligned start, and a tile narrower than
    // CompInt8ColTile can only be the last one of the matrix.
    for (size_t nn = 0; nn < CountN; nn += CompInt8ColTile) {
        const size_t width = std::min(CompInt8ColTile, CountN - nn);
        const uint8_t* b_tile = b_data + nn * ldb;
        const float* b_scale = QuantBScale + nn * BlockCountK;
        const std::byte* b_zp = HasZeroPoint ? QuantBZeroPoint + nn * StrideQuantBZeroPoint : nullptr;
        const float* bias = Bias ? Bias + nn : nullptr;

        if (Geom.SegsPerChunk > 1) {
            // Several blocks per chunk: the multi-block tile covers the whole
            // packed tile, one row at a time.
            for (size_t m = 0; m < CountM; ++m) {
                SQ4BitGemmKernel_CompInt8_MultiBlockTile<HasZeroPoint>(
                    Geom, QuantA + m * lda, b_tile, width, b_scale, b_zp, StrideQuantBZeroPoint,
                    width, C + m * ldc + nn, bias
                );
            }
            continue;
        }

        if (CountM == 1) {
            SQ4BitGemmKernel_CompInt8_Tile<HasZeroPoint, 1, CompInt8ColTile>(
                Geom, QuantA, lda, b_tile, width, b_scale, b_zp,
                StrideQuantBZeroPoint, width, C + nn, ldc, bias
            );
            continue;
        }

        // 2 x 4 sub-tiles over the tile's columns; the tile's B stays cached
        // across the row pairs.
        for (size_t c0 = 0; c0 < width; c0 += 4) {
            const size_t n_cols = std::min<size_t>(4, width - c0);
            const uint8_t* b_sub = b_tile + c0 * Geom.SegHalf;
            const float* sub_scale = b_scale + c0 * BlockCountK;
            const std::byte* sub_zp = HasZeroPoint ? b_zp + c0 * StrideQuantBZeroPoint : nullptr;
            const float* sub_bias = bias ? bias + c0 : nullptr;

            size_t m = 0;
            for (; m + 2 <= CountM; m += 2) {
                SQ4BitGemmKernel_CompInt8_Tile<HasZeroPoint, 2, 4>(
                    Geom, QuantA + m * lda, lda, b_sub, width, sub_scale, sub_zp,
                    StrideQuantBZeroPoint, n_cols, C + m * ldc + nn + c0, ldc, sub_bias
                );
            }
            if (m < CountM) {
                SQ4BitGemmKernel_CompInt8_Tile<HasZeroPoint, 1, 4>(
                    Geom, QuantA + m * lda, lda, b_sub, width, sub_scale, sub_zp,
                    StrideQuantBZeroPoint, n_cols, C + m * ldc + nn + c0, ldc, sub_bias
                );
            }
        }
    }

    return CountM;
}

//
// SQNBIT_CompInt8 kernels for 8-bit weights (BlkSum path).
//
// A is signed int8 (scale = amax/127); B is raw uint8 with a per-block zero
// point (default 128). Per block:
//   C += aScale*bScale*sum_i(qa_i * qbRaw_i)  -  ABlockSum * (bScale*bZeroPoint)
// which equals sum_i (aScale*qa_i) * (bScale*(qbRaw_i - bZeroPoint)) exactly.
// ABlockSum = aScale * sum_i(qa_i); QuantBBlkSum = bScale*bZeroPoint.
//

void
RvvQuantizeARowComputeBlkSum_CompInt8_Impl(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA,
    float* QuantAScale,
    float* AScaledBlkSum
)
{
    const size_t BlockCountK = MlasDivRoundup(CountK, BlkLen);
    const CompInt8Geometry Geom(BlkLen, BlockCountK);
    int8_t* qdata = reinterpret_cast<int8_t*>(QuantA);

    int8_t q[CompInt8MaxBlkLen];
    for (size_t b = 0; b < BlockCountK; ++b) {
        const size_t k0 = b * BlkLen;
        const size_t len = std::min(BlkLen, CountK - k0);
        const float scale = QuantizeABlock(A + k0, len, BlkLen, q);
        QuantAScale[b] = scale;

        vint32m1_t isum = __riscv_vmv_s_x_i32m1(0, 1);
        for (size_t off = 0; off < len;) {
            const size_t vl = __riscv_vsetvl_e8m1(len - off);
            const vint16m2_t w16 = __riscv_vsext_vf2_i16m2(__riscv_vle8_v_i8m1(q + off, vl), vl);
            isum = __riscv_vwredsum_vs_i16m2_i32m1(w16, isum, vl);
            off += vl;
        }
        AScaledBlkSum[b] = scale * static_cast<float>(__riscv_vmv_x_s_i32m1_i32(isum));

        StoreABlock(Geom, b, q, qdata);
    }
}

// MTILE rows x NTILE columns of the SQ8 BlkSum CompInt8 GEMM, BlkLen >= ChunkElems.
// Columns >= n_cols alias column 0.
template <size_t MTILE, size_t NTILE>
MLAS_FORCEINLINE void
SQ8BitGemmKernel_BlkSum_CompInt8_Tile(
    const CompInt8Geometry& Geom,
    const int8_t* a_data,  // first row
    const float* a_scale,  // first row
    const float* a_sum,    // first row
    size_t lda,
    const int8_t* b_data,  // chunk 0 of the tile's first column (centered int8)
    size_t Width,          // columns interleaved in the packed tile
    const float* b_scale,  // first column
    const float* b_sum,    // first column
    size_t n_cols,
    float* C,
    size_t ldc,
    const float* Bias
)
{
    static_assert(MTILE == 1 || MTILE == 2, "unsupported MTILE");
    static_assert(NTILE == 4 || NTILE == 8, "unsupported NTILE");
    static_assert(MTILE * NTILE <= 8, "too many accumulators");
    const size_t BlockCountK = Geom.BlockCountK;
    const size_t SegLen = Geom.SegLen;  // == ChunkElems in this regime
    const size_t SegHalf = Geom.SegHalf;
    const size_t ChunksPerBlock = Geom.ChunksPerBlock;
    const size_t ChunkStride = Width * SegLen;  // packed bytes per chunk of the tile
    const size_t accvl = __riscv_vsetvlmax_e32m2();

    const int8_t* qb[NTILE];  // adjacent columns of a chunk are SegLen apart
    const float* bsc[NTILE];
    const float* bsum[NTILE];
    for (size_t c = 0; c < NTILE; ++c) {
        const size_t cc = (c < n_cols) ? c : 0;
        qb[c] = b_data + cc * SegLen;
        bsc[c] = b_scale + cc * BlockCountK;
        bsum[c] = b_sum + cc * BlockCountK;
    }

    vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0.0f, accvl);
    vfloat32m2_t acc1 = acc0, acc2 = acc0, acc3 = acc0, acc4 = acc0, acc5 = acc0, acc6 = acc0, acc7 = acc0;

    for (size_t b = 0; b < BlockCountK; ++b) {
        const float as0 = a_scale[b];
        const float as1 = a_scale[(MTILE > 1 ? BlockCountK : 0) + b];

        float s0[NTILE], s1[NTILE];
        MLAS_UNROLL_LOOP
        for (size_t c = 0; c < NTILE; ++c) {
            s0[c] = as0 * bsc[c][b];
            s1[c] = as1 * bsc[c][b];
        }

        // One chunk (two halves of 'vl' elements) per int16 partial: the sum of
        // two 8-bit products is the int16 limit, so widen after every chunk.
        for (size_t sub = 0; sub < ChunksPerBlock; ++sub) {
            const size_t chunk = b * ChunksPerBlock + sub;
            const size_t sb = chunk * ChunkStride;      // chunk offset within the packed tile
            const size_t sa = chunk * Geom.ChunkElems;  // chunk offset within the A row
            for (size_t k = 0; k < SegHalf;) {
                const size_t vl = __riscv_vsetvl_e8mf2(SegHalf - k);

                const vint8mf2_t a0_lo = __riscv_vle8_v_i8mf2(a_data + sa + k, vl);
                const vint8mf2_t a0_hi = __riscv_vle8_v_i8mf2(a_data + sa + SegHalf + k, vl);
                vint8mf2_t a1_lo = a0_lo, a1_hi = a0_hi;
                if constexpr (MTILE > 1) {
                    a1_lo = __riscv_vle8_v_i8mf2(a_data + lda + sa + k, vl);
                    a1_hi = __riscv_vle8_v_i8mf2(a_data + lda + sa + SegHalf + k, vl);
                }

#define SQ8_COL(c, acc_r0, acc_r1)                                                  \
    if constexpr ((c) < NTILE) {                                                    \
        const vint8mf2_t b_lo = __riscv_vle8_v_i8mf2(qb[c] + sb + k, vl);           \
        const vint8mf2_t b_hi = __riscv_vle8_v_i8mf2(qb[c] + sb + SegHalf + k, vl); \
        {                                                                           \
            vint16m1_t p_ = __riscv_vwmul_vv_i16m1(a0_lo, b_lo, vl);                \
            p_ = __riscv_vwmacc_vv_i16m1(p_, a0_hi, b_hi, vl);                      \
            QI8_FOLD(acc_r0, p_, s0[c], vl);                                        \
        }                                                                           \
        if constexpr (MTILE > 1) {                                                  \
            vint16m1_t p_ = __riscv_vwmul_vv_i16m1(a1_lo, b_lo, vl);                \
            p_ = __riscv_vwmacc_vv_i16m1(p_, a1_hi, b_hi, vl);                      \
            QI8_FOLD(acc_r1, p_, s1[c], vl);                                        \
        }                                                                           \
    }

                // The barriers keep the compiler from hoisting every column's
                // loads and products to the top of the body, which would need
                // more than the 32 vector registers and spill an accumulator.
                if constexpr (MTILE == 1) {
                    SQ8_COL(0, acc0, acc0)
                    SQ8_COL(1, acc1, acc1)
                        MLAS_SCHED_BARRIER
                            SQ8_COL(2, acc2, acc2) SQ8_COL(3, acc3, acc3)
                                MLAS_SCHED_BARRIER
                                    SQ8_COL(4, acc4, acc4) SQ8_COL(5, acc5, acc5)
                                        MLAS_SCHED_BARRIER
                                            SQ8_COL(6, acc6, acc6) SQ8_COL(7, acc7, acc7)
                } else {
                    SQ8_COL(0, acc0, acc4)
                    SQ8_COL(1, acc1, acc5)
                        MLAS_SCHED_BARRIER
                            SQ8_COL(2, acc2, acc6) SQ8_COL(3, acc3, acc7)
                }
#undef SQ8_COL

                k += vl;
            }
        }
    }

    // Zero-point correction: C -= sum_b ABlockSum[b] * QuantBBlkSum[b], one
    // vector dot over the blocks per output.
    auto correction = [&](const float* asum, const float* bs) -> float {
        vfloat32m2_t v = __riscv_vfmv_v_f_f32m2(0.0f, accvl);
        for (size_t b = 0; b < BlockCountK;) {
            const size_t vl = __riscv_vsetvl_e32m2(BlockCountK - b);
            v = __riscv_vfmacc_vv_f32m2_tu(v, __riscv_vle32_v_f32m2(asum + b, vl), __riscv_vle32_v_f32m2(bs + b, vl), vl);
            b += vl;
        }
        return ReduceSumF32M2(v, accvl);
    };

    auto store = [&](size_t r, size_t c, vfloat32m2_t acc) {
        if (c < n_cols) {
            C[r * ldc + c] = ReduceSumF32M2(acc, accvl) - correction(a_sum + r * BlockCountK, bsum[c]) + (Bias ? Bias[c] : 0.0f);
        }
    };
    if constexpr (MTILE == 1) {
        store(0, 0, acc0);
        store(0, 1, acc1);
        store(0, 2, acc2);
        store(0, 3, acc3);
        if constexpr (NTILE == 8) {
            store(0, 4, acc4);
            store(0, 5, acc5);
            store(0, 6, acc6);
            store(0, 7, acc7);
        }
    } else {
        store(0, 0, acc0);
        store(0, 1, acc1);
        store(0, 2, acc2);
        store(0, 3, acc3);
        store(1, 0, acc4);
        store(1, 1, acc5);
        store(1, 2, acc6);
        store(1, 3, acc7);
    }
}

// One row x 8 columns (a whole packed tile) of the SQ8 BlkSum CompInt8 GEMM,
// BlkLen < ChunkElems: see the 4-bit multi-block tile. Columns >= n_cols
// alias column 0.
MLAS_FORCEINLINE void
SQ8BitGemmKernel_BlkSum_CompInt8_MultiBlockTile(
    const CompInt8Geometry& Geom,
    const int8_t* a_data,
    const float* a_scale,
    const float* a_sum,
    const int8_t* b_data,  // chunk 0 of the packed tile
    size_t Width,
    const float* b_scale,  // first column
    const float* b_sum,    // first column
    size_t n_cols,
    float* C,
    const float* Bias
)
{
    constexpr size_t NTILE = CompInt8ColTile;
    const size_t BlockCountK = Geom.BlockCountK;
    const size_t SegHalf = Geom.SegHalf;  // BlkLen / 2
    const size_t ChunkStride = Width * Geom.ChunkElems;
    const size_t accvl = __riscv_vsetvlmax_e32m2();
    QI8_SEG_ID(seg_id, SegHalf, accvl);

    const float* bs[NTILE];
    const float* bsum[NTILE];
    size_t b_col[NTILE];
    for (size_t c = 0; c < NTILE; ++c) {
        const size_t cc = (c < n_cols) ? c : 0;
        b_col[c] = cc;
        bs[c] = b_scale + cc * BlockCountK;
        bsum[c] = b_sum + cc * BlockCountK;
    }
    const float* as = a_scale;

    vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0.0f, accvl);
    vfloat32m2_t acc1 = acc0, acc2 = acc0, acc3 = acc0, acc4 = acc0, acc5 = acc0, acc6 = acc0, acc7 = acc0;

    for (size_t chunk = 0; chunk < Geom.ChunkCount; ++chunk) {
        const size_t segs = Geom.SegsInChunk(chunk);
        const size_t b0 = chunk * Geom.SegsPerChunk;
        const size_t vl = __riscv_vsetvl_e8mf2(segs * SegHalf);
        const int8_t* b_chunk = b_data + chunk * ChunkStride;
        const int8_t* a_chunk = a_data + chunk * Geom.ChunkElems;

        const vint8mf2_t a_lo = __riscv_vle8_v_i8mf2(a_chunk, vl);
        const vint8mf2_t a_hi = __riscv_vle8_v_i8mf2(a_chunk + vl, vl);

#define SQ8_MBCOL(c, f)                                            \
    vfloat32m2_t f;                                                \
    {                                                              \
        const int8_t* bc = b_chunk + b_col[c] * (2 * vl);          \
        const vint8mf2_t b_lo = __riscv_vle8_v_i8mf2(bc, vl);      \
        const vint8mf2_t b_hi = __riscv_vle8_v_i8mf2(bc + vl, vl); \
        vint16m1_t p = __riscv_vwmul_vv_i16m1(a_lo, b_lo, vl);     \
        p = __riscv_vwmacc_vv_i16m1(p, a_hi, b_hi, vl);            \
        f = __riscv_vfwcvt_f_x_v_f32m2(p, vl);                     \
    }
        {
            SQ8_MBCOL(0, f0)
            SQ8_MBCOL(1, f1) SQ8_MBCOL(2, f2) SQ8_MBCOL(3, f3)
                QI8_FOLD_SEGS4(seg_id, acc0, acc1, acc2, acc3, f0, f1, f2, f3, bs[0], bs[1], bs[2], bs[3])
        }
        {
            SQ8_MBCOL(4, f4)
            SQ8_MBCOL(5, f5) SQ8_MBCOL(6, f6) SQ8_MBCOL(7, f7)
                QI8_FOLD_SEGS4(seg_id, acc4, acc5, acc6, acc7, f4, f5, f6, f7, bs[4], bs[5], bs[6], bs[7])
        }
#undef SQ8_MBCOL
    }

    auto correction = [&](const float* bsc) -> float {
        vfloat32m2_t v = __riscv_vfmv_v_f_f32m2(0.0f, accvl);
        for (size_t b = 0; b < BlockCountK;) {
            const size_t vl = __riscv_vsetvl_e32m2(BlockCountK - b);
            v = __riscv_vfmacc_vv_f32m2_tu(v, __riscv_vle32_v_f32m2(a_sum + b, vl), __riscv_vle32_v_f32m2(bsc + b, vl), vl);
            b += vl;
        }
        return ReduceSumF32M2(v, accvl);
    };
    auto store = [&](size_t c, vfloat32m2_t acc) {
        if (c < n_cols) {
            C[c] = ReduceSumF32M2(acc, accvl) - correction(bsum[c]) + (Bias ? Bias[c] : 0.0f);
        }
    };
    store(0, acc0);
    store(1, acc1);
    store(2, acc2);
    store(3, acc3);
    store(4, acc4);
    store(5, acc5);
    store(6, acc6);
    store(7, acc7);
}

size_t
SQ8BitGemmKernel_BlkSum_CompInt8_Impl(
    size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum
)
{
    // int8(A) x int8(B - 128) block-sum kernel. Per block:
    //   C += aScale*bScale*sum_i(qa_i * qb_i)  -  ABlockSum * (bScale*(bZeroPoint - 128))
    MLAS_UNREFERENCED_PARAMETER(CountK);
    const CompInt8Geometry Geom(BlkLen, BlockCountK);
    const size_t lda = BlockCountK * BlkLen;  // int8 A data per row
    const size_t ldb = BlockCountK * BlkLen;  // int8 B data per column

    const int8_t* a_data = reinterpret_cast<const int8_t*>(QuantA);
    const int8_t* b_data = reinterpret_cast<const int8_t*>(QuantBData);

    // Walk the column tiles of the packed layout (see CompInt8ColTile).
    for (size_t nn = 0; nn < CountN; nn += CompInt8ColTile) {
        const size_t width = std::min(CompInt8ColTile, CountN - nn);
        const int8_t* b_tile = b_data + nn * ldb;
        const float* b_scale = QuantBScale + nn * BlockCountK;
        const float* b_sum = QuantBBlkSum + nn * BlockCountK;
        const float* bias = Bias ? Bias + nn : nullptr;

        if (Geom.SegsPerChunk > 1) {
            for (size_t m = 0; m < CountM; ++m) {
                SQ8BitGemmKernel_BlkSum_CompInt8_MultiBlockTile(
                    Geom, a_data + m * lda, QuantAScale + m * BlockCountK, ABlockSum + m * BlockCountK,
                    b_tile, width, b_scale, b_sum, width, C + m * ldc + nn, bias
                );
            }
            continue;
        }

        if (CountM == 1) {
            SQ8BitGemmKernel_BlkSum_CompInt8_Tile<1, CompInt8ColTile>(
                Geom, a_data, QuantAScale, ABlockSum, lda, b_tile, width,
                b_scale, b_sum, width, C + nn, ldc, bias
            );
            continue;
        }

        for (size_t c0 = 0; c0 < width; c0 += 4) {
            const size_t n_cols = std::min<size_t>(4, width - c0);
            const int8_t* b_sub = b_tile + c0 * Geom.SegLen;
            const float* sub_scale = b_scale + c0 * BlockCountK;
            const float* sub_sum = b_sum + c0 * BlockCountK;
            const float* sub_bias = bias ? bias + c0 : nullptr;

            size_t m = 0;
            for (; m + 2 <= CountM; m += 2) {
                SQ8BitGemmKernel_BlkSum_CompInt8_Tile<2, 4>(
                    Geom, a_data + m * lda, QuantAScale + m * BlockCountK,
                    ABlockSum + m * BlockCountK, lda, b_sub, width, sub_scale, sub_sum, n_cols,
                    C + m * ldc + nn + c0, ldc, sub_bias
                );
            }
            if (m < CountM) {
                SQ8BitGemmKernel_BlkSum_CompInt8_Tile<1, 4>(
                    Geom, a_data + m * lda, QuantAScale + m * BlockCountK,
                    ABlockSum + m * BlockCountK, lda, b_sub, width, sub_scale, sub_sum, n_cols,
                    C + m * ldc + nn + c0, ldc, sub_bias
                );
            }
        }
    }

    return CountM;
}

#undef QI8_FOLD
#undef QI8_SEG_ID
#undef QI8_FOLD_SEG4
#undef QI8_FOLD_SEGS4
#undef QI8_SEG_OFFSETS
#undef MLAS_UNROLL_LOOP
#undef MLAS_SCHED_BARRIER

#endif  // MLAS_USE_RVV

}  // namespace

#if defined(MLAS_USE_RVV)

void
RvvSQ4BitGemmM1Kernel_CompFp32(
    size_t BlkLen,
    const float* A,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias
)
{
    if (QuantBZeroPoint != nullptr) {
        SQ4BitGemmM1Kernel_CompFp32_Impl<true>(
            BlkLen, A, QuantBData, QuantBScale, QuantBZeroPoint, C, CountN, CountK, BlockCountK, Bias
        );
    } else {
        SQ4BitGemmM1Kernel_CompFp32_Impl<false>(
            BlkLen, A, QuantBData, QuantBScale, QuantBZeroPoint, C, CountN, CountK, BlockCountK, Bias
        );
    }
}

void
RvvSQ4BitBlkDequantBForSgemm_CompFp32(
    size_t BlkLen,
    float* FpData,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK
)
{
    if (QuantBZeroPoint != nullptr) {
        Q4BitBlkDequantBForSgemm_CompFp32_Impl<true>(
            BlkLen, FpData, QuantBData, QuantBScale, QuantBZeroPoint, CountN, CountK, BlockCountK
        );
    } else {
        Q4BitBlkDequantBForSgemm_CompFp32_Impl<false>(
            BlkLen, FpData, QuantBData, QuantBScale, QuantBZeroPoint, CountN, CountK, BlockCountK
        );
    }
}

void
RvvQuantizeARow_CompInt8(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA
)
{
    RvvQuantizeARow_CompInt8_Impl(BlkLen, A, CountK, QuantA);
}

size_t
RvvSQ4BitGemmKernel_CompInt8(
    size_t BlkLen,
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    size_t ldc,
    const float* Bias
)
{
    if (QuantBZeroPoint != nullptr) {
        return SQ4BitGemmKernel_CompInt8_Impl<true>(
            BlkLen, QuantA, QuantBData, QuantBScale, QuantBZeroPoint, C,
            CountM, CountN, CountK, BlockCountK, ldc, Bias
        );
    } else {
        return SQ4BitGemmKernel_CompInt8_Impl<false>(
            BlkLen, QuantA, QuantBData, QuantBScale, QuantBZeroPoint, C,
            CountM, CountN, CountK, BlockCountK, ldc, Bias
        );
    }
}

void
RvvQuantizeARowComputeBlkSum_CompInt8(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA,
    float* QuantAScale,
    float* AScaledGroupSum
)
{
    RvvQuantizeARowComputeBlkSum_CompInt8_Impl(BlkLen, A, CountK, QuantA, QuantAScale, AScaledGroupSum);
}

size_t
RvvSQ8BitGemmKernel_BlkSum_CompInt8(
    size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* /*QuantBZeroPoint*/,  // zero point folded into QuantBBlkSum
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum,
    const float* /*BlkUnsignedQuantAZeroPointCorrection*/  // unused on the signed-A path
)
{
    return SQ8BitGemmKernel_BlkSum_CompInt8_Impl(
        BlkLen, QuantA, QuantAScale, QuantBData, QuantBScale, C,
        CountM, CountN, CountK, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum
    );
}

#endif  // MLAS_USE_RVV

#if defined(MLAS_USE_RVV_ZVFH)
// Defined in hqnbitgemm_kernel_rvv.cpp (compiled with -march=rv64gcv_zvfh).
void
RvvHQ4BitBlkDequantBForHgemm_CompFp16(
    size_t BlkLen, MLAS_FP16* FpData, const std::byte* QuantBData, const MLAS_FP16* QuantBScale, const std::byte* QuantBZeroPoint, size_t CountN, size_t CountK, size_t BlockCountK
);
void
RvvHQ4BitGemmKernel_CompFp16(
    const MLAS_FP16* A, const MLAS_FP16* B, const MLAS_FP16* Bias, MLAS_FP16* C, size_t CountM, size_t CountN, size_t K, size_t lda, size_t ldb, size_t ldc
);
void
RvvHQ8BitBlkDequantBForHgemm_CompFp16(
    size_t BlkLen, MLAS_FP16* FpData, const std::byte* QuantBData, const MLAS_FP16* QuantBScale, const std::byte* QuantBZeroPoint, size_t CountN, size_t CountK, size_t BlockCountK
);
#endif

//
// RVV QNBit GEMM dispatch.
//
// Wires up the portable packing/workspace helpers (always) plus, under
// MLAS_USE_RVV, the SQNBIT_CompFp32 (4-bit) and SQNBIT_CompInt8 (4-bit and
// 8-bit) compute kernels, and, under MLAS_USE_RVV_ZVFH, the HQNBIT_CompFp16
// (4-bit and 8-bit) kernels. Compute-type/bit-width variants without an
// assignment below remain null, so MlasIsQNBitGemmAvailable() reports them
// unavailable and they fall back to the generic path.
//
const MLAS_QNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchRvv = []() {
    MLAS_QNBIT_GEMM_DISPATCH d;

    d.Q4BitGemmPackQuantBDataSize = RvvQ4BitGemmPackQuantBDataSize;
    d.SQ4BitGemmPackQuantBData = RvvSQ4BitGemmPackQuantBData;

    d.Q8BitGemmPackQuantBDataSize = RvvQ8BitGemmPackQuantBDataSize;
    d.SQ8BitGemmPackQuantBDataAndBlkSum = RvvSQ8BitGemmPackQuantBDataAndBlkSum;

    d.QNBitGemmPerGemmWorkspaceSize = RvvQNBitGemmPerGemmWorkspaceSize;
    d.QNBitGemmPerGemmWorkspaceAlignment = RvvQNBitGemmPerGemmWorkspaceAlignment;

#if defined(MLAS_USE_RVV)
    d.SQ4BitGemmM1Kernel_CompFp32 = RvvSQ4BitGemmM1Kernel_CompFp32;
    d.SQ4BitBlkDequantBForSgemm_CompFp32 = RvvSQ4BitBlkDequantBForSgemm_CompFp32;

    d.QuantizeARow_CompInt8 = RvvQuantizeARow_CompInt8;
    d.SQ4BitGemmKernel_CompInt8 = RvvSQ4BitGemmKernel_CompInt8;

    d.QuantizeARowComputeBlkSum_CompInt8 = RvvQuantizeARowComputeBlkSum_CompInt8;
    d.SQ8BitGemmKernel_BlkSum_CompInt8 = RvvSQ8BitGemmKernel_BlkSum_CompInt8;
#endif

#if defined(MLAS_USE_RVV_ZVFH)
    // HQNBIT_CompFp16 (4-bit weights, fp16 activations). The B-data packing is
    // type-agnostic, so it reuses the SubBlkLen=16 nibble pack.
    d.HQ4BitGemmPackQuantBData = RvvSQ4BitGemmPackQuantBData;
    d.HQ4BitBlkDequantBForHgemm_CompFp16 = RvvHQ4BitBlkDequantBForHgemm_CompFp16;
    d.HQ4BitGemmKernel_CompFp16 = RvvHQ4BitGemmKernel_CompFp16;

    // HQ8 (8-bit weights, fp16 activations) reuses the fp16 GEMM kernel above.
    d.HQ8BitGemmPackQuantBData = RvvHQ8BitGemmPackQuantBData;
    d.HQ8BitBlkDequantBForHgemm_CompFp16 = RvvHQ8BitBlkDequantBForHgemm_CompFp16;
#endif

    return d;
}();
