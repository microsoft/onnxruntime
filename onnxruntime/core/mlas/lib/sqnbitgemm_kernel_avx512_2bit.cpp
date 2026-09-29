/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx512_2bit.cpp

Abstract:

    Pack helpers and a scalar reference kernel for the block-group W2 layout.

    See sqnbitgemm_kernel_avx512_2bit.h for the layout description
    and the rationale (closing the W2-vs-W4 prefill gap by replacing the
    per-block broadcast + variable shift unpack with a single 64-byte load +
    four fixed-shift+mask pairs).

    This translation unit is scalar / portable. The vectorized inner loops
    that consume the block-group layout live in separate headers
    (sqnbitgemm_kernel_avx512_2bit_blklen{32,64,128}.h) and are wired into the
    AVX-512 and AVX-512-VNNI dispatch tables.

--*/

#include "sqnbitgemm_kernel_avx512_2bit.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>

#include "mlasi.h"
#include "qnbitgemm.h"

namespace onnxruntime {
namespace mlas {
namespace sq2bit_avx512 {

//
// Workspace / pack-buffer size for the block-group W2 path. Returns 0 if any
// of the configuration constraints is violated; the caller
// (MlasQNBitGemmPackQuantBDataSize) treats that as "unsupported" and falls
// back to LUT (if opted in and the shape is LUT-eligible) or to the fp32
// dequant + SGEMM path in MatMulNBits::ComputeBUnpacked.
//
// Constraints:
//   * BlkLen ∈ {32, 64, 128}
//   * ComputeType == SQNBIT_CompInt8
//
// K-tail handling: BlockCountK is rounded UP to a multiple of kBlockGroupBlks
// for the storage that the inner K-loop walks (PackedQuantBData,
// PackedQuantBScale). Padding slots hold zeroed weights and scales, so they
// contribute exactly 0 to the dot product. The BlkSum buffer is *physically*
// sized with the padded BlockCountK so its offset within the combined
// workspace is consistent with the caller's PackedQuantBDataStruct layout
// (which uses one BlockCountK value for the whole struct); the SGEMM
// correction step still reads only the logical BlockCountK entries from it.
//
// Storage matches the original W2 layout total bytes when BlockCountK is a
// multiple of 4. When not a multiple of 4, storage grows by at most 3 K-blocks
// per N-col (= up to 48 bytes per col -- negligible at production N values).
//
//   [PackedQuantBData]  N * BlockCountKPadded * kBlkBytes
//   [PackedQuantBScale] N * BlockCountKPadded * sizeof(float)
//   [QuantBBlkSum]      roundup_16(N) * BlockCountKPadded * 16 floats (only
//                       the first BlockCountK entries per N are populated)
//
size_t MLASCALL
Q2BitGemmPackQuantBDataSize_Avx512(
    size_t N,
    size_t K,
    size_t BlkLen,
    bool /* HasZeroPoint */,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /* BackendKernelSelectorConfig */
)
{
    if (ComputeType != SQNBIT_CompInt8) {
        return 0;
    }
    if (BlkLen != kBlkLen && BlkLen != kBlkLen128 && BlkLen != kBlkLen32) {
        return 0;
    }
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    if (BlockCountK == 0) {
        return 0;
    }
    const size_t BlockCountKPadded =
        MlasDivRoundup(BlockCountK, kBlockGroupBlks) * kBlockGroupBlks;

    // Per-block packed-data byte count: BlkLen / 4 (= kBlkBytes for BlkLen=64,
    // = kBlkBytes128 for BlkLen=128). Use the runtime BlkLen so the same
    // pack-size helper covers both kernel variants -- the BlkLen=64 storage
    // total when BlkLen=64 is bit-identical to the original computation.
    const size_t BlkBytes = BlkLen / kWeightsPerByte;

    // Use BlockCountKPadded for BlkSum sizing too. The actual SGEMM-correction
    // step only reads LOGICAL BlockCountK entries, but PackedQuantBDataStruct
    // is constructed by the caller with a single BlockCountK value that
    // controls BOTH the packed-B size and the BlkSum offset. If we sized
    // BlkSum at the logical BlockCountK while sizing packed-B at padded,
    // the struct's BlkSum pointer would land inside the packed-B region
    // (because the caller's struct uses one BlockCountK consistently). The
    // extra storage from padding the BlkSum is ~16 floats per N -- trivial.
    size_t PackedQuantBDataSize = N * BlockCountKPadded * BlkBytes;
    const size_t ScaleSize = N * BlockCountKPadded * sizeof(float);
    size_t BlkSumSize = MlasDivRoundup(N, 16) * BlockCountKPadded * 16 * sizeof(float);

    constexpr size_t kPackedQuantBDataAlignment = 64;
    PackedQuantBDataSize += kPackedQuantBDataAlignment - 1;

    constexpr size_t kBlkSumAlignment = MlasQNBitQuantBBlkSumAlignment();
    BlkSumSize += kBlkSumAlignment - 1;

    return PackedQuantBDataSize + ScaleSize + BlkSumSize;
}

//
// Pack quantized B data + scales + per-block sums for the block-group W2 path.
//
// PackedQuantBData layout (block-groups of 4 K-blocks, 64 bytes each):
//   The block-group at logical (n, blk_group=blk/4) lives at byte offset
//   PackedQuantBOffsetBytes_W2(n, blk_group, BlockGroupCountK, NMain).
//   Byte b within the block-group holds 2-bit weight b from each of the 4
//   constituent K-blocks at bit positions {0..1, 2..3, 4..5, 6..7}.
//
// PackedQuantBScale layout: one float per K-block, four floats per block-group,
// addressed by PackedQuantBScaleOffset_W2.
//
// QuantBBlkSum layout: the same width-16 row-major chunked layout used by the
// existing W2 path, so the SGEMM correction step (MlasGemmFloatKernel) can be
// shared verbatim with the existing kernel.
//
// Mirrors the SQ2BitGemmPackQuantBDataAndBlkSum_Scalar prepack 3-call pattern:
// ORT's matmul_nbits.cc invokes this function up to three times (B, scales, ZP).
// We write scales when scales arrive, then re-derive BlkSum whenever either
// scales or zero-points arrive, reading scales from the already-packed buffer.
//

// Forward declaration: BlkLen=128 variant lives further down in this TU.
static void
SQ2BitGemmPackQuantBDataAndBlkSum_BlkLen128_Scalar(
    size_t N, size_t K,
    const std::byte* QuantBDataBegin,
    const float* QuantBScaleBegin,
    const std::byte* QuantBZPBegin,
    PackedQuantBDataStruct<float, 2>& PackedQuantB,
    MLAS_THREADPOOL* ThreadPool);

// Forward declaration: BlkLen=32 variant lives further down in this TU.
static void
SQ2BitGemmPackQuantBDataAndBlkSum_BlkLen32_Scalar(
    size_t N, size_t K,
    const std::byte* QuantBDataBegin,
    const float* QuantBScaleBegin,
    const std::byte* QuantBZPBegin,
    PackedQuantBDataStruct<float, 2>& PackedQuantB,
    MLAS_THREADPOOL* ThreadPool);

void MLASCALL
SQ2BitGemmPackQuantBDataAndBlkSum_Scalar(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /* ComputeType */,
    const std::byte* QuantBDataBegin,
    const float* QuantBScaleBegin,
    bool /* HasZeroPoint */,
    const std::byte* QuantBZPBegin,
    PackedQuantBDataStruct<float, 2>& PackedQuantB,
    MLAS_THREADPOOL* ThreadPool,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /* BackendKernelSelectorConfig */
)
{
    // BlkLen=128 dispatches to its own parallel implementation below; the
    // BlkLen=64 path remains exactly as before for full bit-for-bit parity.
    if (BlkLen == kBlkLen128) {
        SQ2BitGemmPackQuantBDataAndBlkSum_BlkLen128_Scalar(
            N, K, QuantBDataBegin, QuantBScaleBegin, QuantBZPBegin,
            PackedQuantB, ThreadPool);
        return;
    }
    if (BlkLen == kBlkLen32) {
        SQ2BitGemmPackQuantBDataAndBlkSum_BlkLen32_Scalar(
            N, K, QuantBDataBegin, QuantBScaleBegin, QuantBZPBegin,
            PackedQuantB, ThreadPool);
        return;
    }
    assert(BlkLen == kBlkLen);
    if (BlkLen != kBlkLen) {
        return;
    }

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    if (BlockCountK == 0) {
        return;
    }

    // Pad BlockCountK up to a multiple of kBlockGroupBlks so the inner K-loop
    // can iterate whole block-groups uniformly. Padding slots store zeroed
    // weights / scales and contribute exactly 0 to the dot product.
    const size_t BlockGroupCountKPadded =
        MlasDivRoundup(BlockCountK, kBlockGroupBlks);
    const size_t BlockCountKPadded = BlockGroupCountKPadded * kBlockGroupBlks;
    const size_t NMain = (N / kNCols4) * kNCols4;

    // Zero source block used when packing a group whose K-range crosses the
    // logical BlockCountK boundary. We point to this static zero buffer in
    // the missing slots so the existing 4-block pack helper does the right
    // thing without any branching inside it.
    static const std::byte kZeroBlock[kBlkBytes] = {};

    // ----- B-data pack -----
    if (QuantBDataBegin != nullptr) {
        std::byte* PackedQuantBData = PackedQuantB.PackedQuantBData;
        const size_t Iterations = N * BlockGroupCountKPadded;
        MlasTrySimpleParallel(
            ThreadPool, static_cast<ptrdiff_t>(Iterations),
            [&](ptrdiff_t tid) {
                const size_t n = static_cast<size_t>(tid) / BlockGroupCountKPadded;
                const size_t blk_group = static_cast<size_t>(tid) % BlockGroupCountKPadded;
                const size_t blk0 = blk_group * kBlockGroupBlks;

                // Pick real source block pointers for slots that exist; the
                // static zero buffer for slots past the logical BlockCountK.
                auto src_for = [&](size_t blk) -> const std::byte* {
                    if (blk < BlockCountK) {
                        return QuantBDataBegin + (n * BlockCountK + blk) * kBlkBytes;
                    }
                    return kZeroBlock;
                };
                const std::byte* src_blk_0 = src_for(blk0 + 0);
                const std::byte* src_blk_1 = src_for(blk0 + 1);
                const std::byte* src_blk_2 = src_for(blk0 + 2);
                const std::byte* src_blk_3 = src_for(blk0 + 3);

                const size_t dst_offset =
                    PackedQuantBOffsetBytes_W2(n, blk_group, BlockGroupCountKPadded, NMain);
                PackBlockGroup_BlkLen64(src_blk_0, src_blk_1, src_blk_2, src_blk_3,
                                              PackedQuantBData + dst_offset);
            }
        );
    }

    // ----- Scales -----
    // Iterate over the PADDED block count so trailing padding slots get
    // explicit zero scales (otherwise they could hold uninitialised noise
    // and the kernel's K-loop would read those into the FMA).
    if (QuantBScaleBegin != nullptr) {
        float* PackedScales = PackedQuantB.PackedQuantBScale;
        const size_t Iterations = N * BlockCountKPadded;
        MlasTrySimpleParallel(
            ThreadPool, static_cast<ptrdiff_t>(Iterations),
            [&](ptrdiff_t tid) {
                const size_t n = static_cast<size_t>(tid) / BlockCountKPadded;
                const size_t blk = static_cast<size_t>(tid) % BlockCountKPadded;
                const float scale = (blk < BlockCountK)
                    ? QuantBScaleBegin[n * BlockCountK + blk]
                    : 0.0f;
                PackedScales[PackedQuantBScaleOffset_W2(n, blk, BlockCountKPadded, NMain)] = scale;
            }
        );
    }

    // ----- BlkSum (recomputed whenever scales or ZPs arrive) -----
    // BlkSum is consumed by the SGEMM correction step (MlasGemmFloatKernel),
    // which is called outside the inner K-loop with the LOGICAL BlockCountK
    // and the per-row ABlockSum the dispatcher produced for that logical K.
    // We therefore only need to fill the first BlockCountK entries; the buffer
    // is sized at MlasDivRoundup(N, 16) * BlockCountK * 16 floats (logical).
    if (QuantBScaleBegin != nullptr || QuantBZPBegin != nullptr) {
        const float* PackedScales = PackedQuantB.PackedQuantBScale;
        float* BlkSum = PackedQuantB.QuantBBlkSum;
        const size_t ZPCountK = MlasDivRoundup(BlockCountK, 4);
        const size_t Iterations = N * BlockCountK;
        MlasTrySimpleParallel(
            ThreadPool, static_cast<ptrdiff_t>(Iterations),
            [&](ptrdiff_t tid) {
                const size_t n = static_cast<size_t>(tid) / BlockCountK;
                const size_t blk = static_cast<size_t>(tid) % BlockCountK;
                const float scale =
                    PackedScales[PackedQuantBScaleOffset_W2(n, blk, BlockCountKPadded, NMain)];

                uint8_t zp = kDefaultSymmetricZeroPoint2Bit;
                if (QuantBZPBegin != nullptr) {
                    const size_t zp_byte_idx = n * ZPCountK + (blk / 4);
                    const size_t zp_bit_off = (blk % 4) * 2;
                    zp = static_cast<uint8_t>(
                        (static_cast<uint8_t>(QuantBZPBegin[zp_byte_idx]) >> zp_bit_off) & 0x03u);
                }

                const size_t blksum_offset = ((n / 16) * BlockCountK + blk) * 16 + (n % 16);
                BlkSum[blksum_offset] = -scale * static_cast<float>(zp);
            }
        );
    }
}

//
// Scalar reference kernel that consumes the block-group packed layout.
// Same math as the existing reference kernel; differs only in how it walks
// PackedQuantBData (block-group-major) and PackedQuantBScale (block-group-major).
//
// This is the correctness oracle for the SIMD block-group kernel coming in
// SIMD path. It also lets us validate the pack layout end-to-end via the
// existing MlasQNBitGemmBatch dispatch path once we wire it up.
//

// Forward declaration: BlkLen=128 variant lives further down in this TU.
static size_t
SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen128_Scalar(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum);

// Forward declaration: BlkLen=32 variant lives further down in this TU.
static size_t
SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen32_Scalar(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum);

size_t MLASCALL
SQ2BitGemmKernel_BlkSum_CompInt8_Scalar(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* /* QuantBZeroPoint */,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t /* CountK */,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum
)
{
    if (BlkLen == kBlkLen128) {
        return SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen128_Scalar(
            QuantA, QuantAScale, QuantBData, QuantBScale,
            C, CountM, CountN, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
    }
    if (BlkLen == kBlkLen32) {
        return SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen32_Scalar(
            QuantA, QuantAScale, QuantBData, QuantBScale,
            C, CountM, CountN, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
    }
    if (BlkLen != kBlkLen) {
        return 0;
    }
    if (BlockCountK == 0) {
        return 0;
    }

    // PackedQuantBData and PackedQuantBScale are addressed via padded counts
    // (K-tail handling -- see Q2BitGemmPackQuantBDataSize_Avx512). The K
    // dot-product loop itself iterates only LOGICAL BlockCountK steps because
    // A is unpadded; the kernel never reads past the real A rows.
    const size_t BlockGroupCountKPadded =
        MlasDivRoundup(BlockCountK, kBlockGroupBlks);
    const size_t BlockCountKPadded = BlockGroupCountKPadded * kBlockGroupBlks;

    // The kernel is called by SQ2BitGemm_CompInt8 with the full CountN range; that
    // function selects an N-tile boundary (kNCols4) up the stack. For a scalar
    // reference path we don't depend on the 4-N-col grouping, but we DO need to
    // index PackedQuantBData/PackedQuantBScale via the block-group offset helpers
    // so we read the right bytes regardless of caller tile choice.
    //
    // CountN may not be a multiple of kNCols4 in the tail case. Detect that and
    // fall back to plain column-major for the tail cols (the layout helpers
    // already encode this rule).
    const size_t NMainLocal = (CountN / kNCols4) * kNCols4;

    const size_t lda = BlockCountK * kBlkLen;  // bytes per A row (int8)
    const size_t lda_scale = BlockCountK;            // floats per A scale row

    for (size_t m = 0; m < CountM; ++m) {
        const int8_t* a_row = reinterpret_cast<const int8_t*>(QuantA + m * lda);
        const float* a_scale_row = QuantAScale + m * lda_scale;
        const float* a_blksum_row = ABlockSum + m * lda_scale;
        float* c_row = C + m * ldc;

        for (size_t n = 0; n < CountN; ++n) {
            float acc = (Bias != nullptr) ? Bias[n] : 0.0f;

            for (size_t blk = 0; blk < BlockCountK; ++blk) {
                // Pull the block-group this K-block belongs to and unpack only the
                // slot we need (block_in_group = blk % 4 selects the 2-bit field).
                const size_t blk_group = blk / kBlockGroupBlks;
                const size_t blk_in_group = blk % kBlockGroupBlks;
                const size_t block_group_offset =
                    PackedQuantBOffsetBytes_W2(n, blk_group, BlockGroupCountKPadded, NMainLocal);
                const std::byte* block_group = QuantBData + block_group_offset;

                uint8_t b_unpacked[kBlkLen];
                for (size_t i = 0; i < kBlkLen; ++i) {
                    const uint8_t byte = static_cast<uint8_t>(block_group[i]);
                    b_unpacked[i] = static_cast<uint8_t>((byte >> (2 * blk_in_group)) & 0x03u);
                }

                const int8_t* a_blk = a_row + blk * kBlkLen;
                int32_t dot = 0;
                for (size_t i = 0; i < kBlkLen; ++i) {
                    dot += static_cast<int32_t>(a_blk[i]) * static_cast<int32_t>(b_unpacked[i]);
                }

                const float b_scale =
                    QuantBScale[PackedQuantBScaleOffset_W2(n, blk, BlockCountKPadded, NMainLocal)];
                acc += a_scale_row[blk] * b_scale * static_cast<float>(dot);

                // The width-16 row-major BlkSum layout is column-major in n
                // (one float per (n, blk)); same as the existing W2 path.
                const size_t blksum_offset = ((n / 16) * BlockCountK + blk) * 16 + (n % 16);
                acc += a_blksum_row[blk] * QuantBBlkSum[blksum_offset];
            }

            c_row[n] = acc;
        }
    }

    return CountM;
}

// -----------------------------------------------------------------------------
// BlkLen=128 variants of pack-and-blksum and the scalar oracle.
//
// These are direct ports of the BlkLen=64 variants with three substitutions:
//   * kBlkBytes              -> kBlkBytes128             (32 bytes per block)
//   * kBlkLen                -> kBlkLen128               (128 weights per block)
//   * PackBlockGroup_BlkLen64        -> PackBlockGroup_BlkLen128
//   * PackedQuantBOffsetBytes_W2     -> PackedQuantBOffsetBytes_W2_BlkLen128
//
// PackedQuantBScaleOffset_W2 is reused as-is (scale layout is BlkLen-invariant).
// The N-major / 4-col-grouped layout, the K-tail rounding rule, and the
// SGEMM correction step (width-16 BlkSum) are identical to BlkLen=64.
// -----------------------------------------------------------------------------

static void
SQ2BitGemmPackQuantBDataAndBlkSum_BlkLen128_Scalar(
    size_t N,
    size_t K,
    const std::byte* QuantBDataBegin,
    const float* QuantBScaleBegin,
    const std::byte* QuantBZPBegin,
    PackedQuantBDataStruct<float, 2>& PackedQuantB,
    MLAS_THREADPOOL* ThreadPool)
{
    const size_t BlockCountK = MlasDivRoundup(K, kBlkLen128);
    if (BlockCountK == 0) {
        return;
    }

    const size_t BlockGroupCountKPadded =
        MlasDivRoundup(BlockCountK, kBlockGroupBlks);
    const size_t BlockCountKPadded = BlockGroupCountKPadded * kBlockGroupBlks;
    const size_t NMain = (N / kNCols4) * kNCols4;

    static const std::byte kZeroBlock128[kBlkBytes128] = {};

    // ----- B-data pack -----
    if (QuantBDataBegin != nullptr) {
        std::byte* PackedQuantBData = PackedQuantB.PackedQuantBData;
        const size_t Iterations = N * BlockGroupCountKPadded;
        MlasTrySimpleParallel(
            ThreadPool, static_cast<ptrdiff_t>(Iterations),
            [&](ptrdiff_t tid) {
                const size_t n = static_cast<size_t>(tid) / BlockGroupCountKPadded;
                const size_t blk_group = static_cast<size_t>(tid) % BlockGroupCountKPadded;
                const size_t blk0 = blk_group * kBlockGroupBlks;

                auto src_for = [&](size_t blk) -> const std::byte* {
                    if (blk < BlockCountK) {
                        return QuantBDataBegin + (n * BlockCountK + blk) * kBlkBytes128;
                    }
                    return kZeroBlock128;
                };
                const std::byte* src_blk_0 = src_for(blk0 + 0);
                const std::byte* src_blk_1 = src_for(blk0 + 1);
                const std::byte* src_blk_2 = src_for(blk0 + 2);
                const std::byte* src_blk_3 = src_for(blk0 + 3);

                const size_t dst_offset =
                    PackedQuantBOffsetBytes_W2_BlkLen128(n, blk_group, BlockGroupCountKPadded, NMain);
                PackBlockGroup_BlkLen128(src_blk_0, src_blk_1, src_blk_2, src_blk_3,
                                         PackedQuantBData + dst_offset);
            }
        );
    }

    // ----- Scales (same layout as BlkLen=64; reuse PackedQuantBScaleOffset_W2) -----
    if (QuantBScaleBegin != nullptr) {
        float* PackedScales = PackedQuantB.PackedQuantBScale;
        const size_t Iterations = N * BlockCountKPadded;
        MlasTrySimpleParallel(
            ThreadPool, static_cast<ptrdiff_t>(Iterations),
            [&](ptrdiff_t tid) {
                const size_t n = static_cast<size_t>(tid) / BlockCountKPadded;
                const size_t blk = static_cast<size_t>(tid) % BlockCountKPadded;
                const float scale = (blk < BlockCountK)
                    ? QuantBScaleBegin[n * BlockCountK + blk]
                    : 0.0f;
                PackedScales[PackedQuantBScaleOffset_W2(n, blk, BlockCountKPadded, NMain)] = scale;
            }
        );
    }

    // ----- BlkSum (recomputed whenever scales or ZPs arrive) -----
    if (QuantBScaleBegin != nullptr || QuantBZPBegin != nullptr) {
        const float* PackedScales = PackedQuantB.PackedQuantBScale;
        float* BlkSum = PackedQuantB.QuantBBlkSum;
        const size_t ZPCountK = MlasDivRoundup(BlockCountK, 4);
        const size_t Iterations = N * BlockCountK;
        MlasTrySimpleParallel(
            ThreadPool, static_cast<ptrdiff_t>(Iterations),
            [&](ptrdiff_t tid) {
                const size_t n = static_cast<size_t>(tid) / BlockCountK;
                const size_t blk = static_cast<size_t>(tid) % BlockCountK;
                const float scale =
                    PackedScales[PackedQuantBScaleOffset_W2(n, blk, BlockCountKPadded, NMain)];

                uint8_t zp = kDefaultSymmetricZeroPoint2Bit;
                if (QuantBZPBegin != nullptr) {
                    const size_t zp_byte_idx = n * ZPCountK + (blk / 4);
                    const size_t zp_bit_off = (blk % 4) * 2;
                    zp = static_cast<uint8_t>(
                        (static_cast<uint8_t>(QuantBZPBegin[zp_byte_idx]) >> zp_bit_off) & 0x03u);
                }

                const size_t blksum_offset = ((n / 16) * BlockCountK + blk) * 16 + (n % 16);
                BlkSum[blksum_offset] = -scale * static_cast<float>(zp);
            }
        );
    }
}

static size_t
SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen128_Scalar(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum)
{
    if (BlockCountK == 0) {
        return 0;
    }

    const size_t BlockGroupCountKPadded =
        MlasDivRoundup(BlockCountK, kBlockGroupBlks);
    const size_t BlockCountKPadded = BlockGroupCountKPadded * kBlockGroupBlks;

    const size_t NMainLocal = (CountN / kNCols4) * kNCols4;

    const size_t lda = BlockCountK * kBlkLen128;       // bytes per A row (int8)
    const size_t lda_scale = BlockCountK;              // floats per A scale row

    for (size_t m = 0; m < CountM; ++m) {
        const int8_t* a_row = reinterpret_cast<const int8_t*>(QuantA + m * lda);
        const float* a_scale_row = QuantAScale + m * lda_scale;
        const float* a_blksum_row = ABlockSum + m * lda_scale;
        float* c_row = C + m * ldc;

        for (size_t n = 0; n < CountN; ++n) {
            float acc = (Bias != nullptr) ? Bias[n] : 0.0f;

            for (size_t blk = 0; blk < BlockCountK; ++blk) {
                const size_t blk_group = blk / kBlockGroupBlks;
                const size_t blk_in_group = blk % kBlockGroupBlks;
                const size_t block_group_offset =
                    PackedQuantBOffsetBytes_W2_BlkLen128(n, blk_group, BlockGroupCountKPadded, NMainLocal);
                const std::byte* block_group = QuantBData + block_group_offset;

                uint8_t b_unpacked[kBlkLen128];
                for (size_t i = 0; i < kBlkLen128; ++i) {
                    const uint8_t byte = static_cast<uint8_t>(block_group[i]);
                    b_unpacked[i] = static_cast<uint8_t>((byte >> (2 * blk_in_group)) & 0x03u);
                }

                const int8_t* a_blk = a_row + blk * kBlkLen128;
                int32_t dot = 0;
                for (size_t i = 0; i < kBlkLen128; ++i) {
                    dot += static_cast<int32_t>(a_blk[i]) * static_cast<int32_t>(b_unpacked[i]);
                }

                const float b_scale =
                    QuantBScale[PackedQuantBScaleOffset_W2(n, blk, BlockCountKPadded, NMainLocal)];
                acc += a_scale_row[blk] * b_scale * static_cast<float>(dot);

                const size_t blksum_offset = ((n / 16) * BlockCountK + blk) * 16 + (n % 16);
                acc += a_blksum_row[blk] * QuantBBlkSum[blksum_offset];
            }

            c_row[n] = acc;
        }
    }

    return CountM;
}

// -----------------------------------------------------------------------------
// BlkLen=32 variants of pack-and-blksum and the scalar oracle. Same pattern
// as the BlkLen=128 variants above (which are themselves direct ports of the
// BlkLen=64 implementations). Only the BlkLen-specific constants and the
// BlkLen=32 pack/offset helpers differ.
// -----------------------------------------------------------------------------

static void
SQ2BitGemmPackQuantBDataAndBlkSum_BlkLen32_Scalar(
    size_t N,
    size_t K,
    const std::byte* QuantBDataBegin,
    const float* QuantBScaleBegin,
    const std::byte* QuantBZPBegin,
    PackedQuantBDataStruct<float, 2>& PackedQuantB,
    MLAS_THREADPOOL* ThreadPool)
{
    const size_t BlockCountK = MlasDivRoundup(K, kBlkLen32);
    if (BlockCountK == 0) {
        return;
    }

    const size_t BlockGroupCountKPadded =
        MlasDivRoundup(BlockCountK, kBlockGroupBlks);
    const size_t BlockCountKPadded = BlockGroupCountKPadded * kBlockGroupBlks;
    const size_t NMain = (N / kNCols4) * kNCols4;

    static const std::byte kZeroBlock32[kBlkBytes32] = {};

    // ----- B-data pack -----
    if (QuantBDataBegin != nullptr) {
        std::byte* PackedQuantBData = PackedQuantB.PackedQuantBData;
        const size_t Iterations = N * BlockGroupCountKPadded;
        MlasTrySimpleParallel(
            ThreadPool, static_cast<ptrdiff_t>(Iterations),
            [&](ptrdiff_t tid) {
                const size_t n = static_cast<size_t>(tid) / BlockGroupCountKPadded;
                const size_t blk_group = static_cast<size_t>(tid) % BlockGroupCountKPadded;
                const size_t blk0 = blk_group * kBlockGroupBlks;

                auto src_for = [&](size_t blk) -> const std::byte* {
                    if (blk < BlockCountK) {
                        return QuantBDataBegin + (n * BlockCountK + blk) * kBlkBytes32;
                    }
                    return kZeroBlock32;
                };
                const std::byte* src_blk_0 = src_for(blk0 + 0);
                const std::byte* src_blk_1 = src_for(blk0 + 1);
                const std::byte* src_blk_2 = src_for(blk0 + 2);
                const std::byte* src_blk_3 = src_for(blk0 + 3);

                const size_t dst_offset =
                    PackedQuantBOffsetBytes_W2_BlkLen32(n, blk_group, BlockGroupCountKPadded, NMain);
                PackBlockGroup_BlkLen32(src_blk_0, src_blk_1, src_blk_2, src_blk_3,
                                        PackedQuantBData + dst_offset);
            }
        );
    }

    // ----- Scales (same layout as BlkLen=64; reuse PackedQuantBScaleOffset_W2) -----
    if (QuantBScaleBegin != nullptr) {
        float* PackedScales = PackedQuantB.PackedQuantBScale;
        const size_t Iterations = N * BlockCountKPadded;
        MlasTrySimpleParallel(
            ThreadPool, static_cast<ptrdiff_t>(Iterations),
            [&](ptrdiff_t tid) {
                const size_t n = static_cast<size_t>(tid) / BlockCountKPadded;
                const size_t blk = static_cast<size_t>(tid) % BlockCountKPadded;
                const float scale = (blk < BlockCountK)
                    ? QuantBScaleBegin[n * BlockCountK + blk]
                    : 0.0f;
                PackedScales[PackedQuantBScaleOffset_W2(n, blk, BlockCountKPadded, NMain)] = scale;
            }
        );
    }

    // ----- BlkSum (recomputed whenever scales or ZPs arrive) -----
    if (QuantBScaleBegin != nullptr || QuantBZPBegin != nullptr) {
        const float* PackedScales = PackedQuantB.PackedQuantBScale;
        float* BlkSum = PackedQuantB.QuantBBlkSum;
        const size_t ZPCountK = MlasDivRoundup(BlockCountK, 4);
        const size_t Iterations = N * BlockCountK;
        MlasTrySimpleParallel(
            ThreadPool, static_cast<ptrdiff_t>(Iterations),
            [&](ptrdiff_t tid) {
                const size_t n = static_cast<size_t>(tid) / BlockCountK;
                const size_t blk = static_cast<size_t>(tid) % BlockCountK;
                const float scale =
                    PackedScales[PackedQuantBScaleOffset_W2(n, blk, BlockCountKPadded, NMain)];

                uint8_t zp = kDefaultSymmetricZeroPoint2Bit;
                if (QuantBZPBegin != nullptr) {
                    const size_t zp_byte_idx = n * ZPCountK + (blk / 4);
                    const size_t zp_bit_off = (blk % 4) * 2;
                    zp = static_cast<uint8_t>(
                        (static_cast<uint8_t>(QuantBZPBegin[zp_byte_idx]) >> zp_bit_off) & 0x03u);
                }

                const size_t blksum_offset = ((n / 16) * BlockCountK + blk) * 16 + (n % 16);
                BlkSum[blksum_offset] = -scale * static_cast<float>(zp);
            }
        );
    }
}

static size_t
SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen32_Scalar(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum)
{
    if (BlockCountK == 0) {
        return 0;
    }

    const size_t BlockGroupCountKPadded =
        MlasDivRoundup(BlockCountK, kBlockGroupBlks);
    const size_t BlockCountKPadded = BlockGroupCountKPadded * kBlockGroupBlks;

    const size_t NMainLocal = (CountN / kNCols4) * kNCols4;

    const size_t lda = BlockCountK * kBlkLen32;        // bytes per A row (int8)
    const size_t lda_scale = BlockCountK;              // floats per A scale row

    for (size_t m = 0; m < CountM; ++m) {
        const int8_t* a_row = reinterpret_cast<const int8_t*>(QuantA + m * lda);
        const float* a_scale_row = QuantAScale + m * lda_scale;
        const float* a_blksum_row = ABlockSum + m * lda_scale;
        float* c_row = C + m * ldc;

        for (size_t n = 0; n < CountN; ++n) {
            float acc = (Bias != nullptr) ? Bias[n] : 0.0f;

            for (size_t blk = 0; blk < BlockCountK; ++blk) {
                const size_t blk_group = blk / kBlockGroupBlks;
                const size_t blk_in_group = blk % kBlockGroupBlks;
                const size_t block_group_offset =
                    PackedQuantBOffsetBytes_W2_BlkLen32(n, blk_group, BlockGroupCountKPadded, NMainLocal);
                const std::byte* block_group = QuantBData + block_group_offset;

                uint8_t b_unpacked[kBlkLen32];
                for (size_t i = 0; i < kBlkLen32; ++i) {
                    const uint8_t byte = static_cast<uint8_t>(block_group[i]);
                    b_unpacked[i] = static_cast<uint8_t>((byte >> (2 * blk_in_group)) & 0x03u);
                }

                const int8_t* a_blk = a_row + blk * kBlkLen32;
                int32_t dot = 0;
                for (size_t i = 0; i < kBlkLen32; ++i) {
                    dot += static_cast<int32_t>(a_blk[i]) * static_cast<int32_t>(b_unpacked[i]);
                }

                const float b_scale =
                    QuantBScale[PackedQuantBScaleOffset_W2(n, blk, BlockCountKPadded, NMainLocal)];
                acc += a_scale_row[blk] * b_scale * static_cast<float>(dot);

                const size_t blksum_offset = ((n / 16) * BlockCountK + blk) * 16 + (n % 16);
                acc += a_blksum_row[blk] * QuantBBlkSum[blksum_offset];
            }

            c_row[n] = acc;
        }
    }

    return CountM;
}

}  // namespace sq2bit_avx512
}  // namespace mlas
}  // namespace onnxruntime
