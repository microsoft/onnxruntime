/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx512_2bit.h

Abstract:

    Pack-time helpers and scalar reference routines for the block-group W2 layout: groups of 4 K-blocks share a single 64-byte
    packed buffer that allows the AVX-512 unpack to be one ZMM load plus
    four fixed `vpsrlw+vpand` pairs (instead of the current per-block
    broadcast + variable shift).

    Layout summary (BlkLen ∈ {32, 64, 128}):

      * Each "block-group" packs FOUR consecutive K-blocks.
      * Total storage per block-group = kBlkBytes * 4 bytes (kBlkBytes is
        BlkLen-dependent: 8 bytes at BlkLen=32, 16 bytes at BlkLen=64, 32
        bytes at BlkLen=128). Identical to 4 separately-packed blocks under
        the per-block scheme, just relaid out so all four blocks are reachable
        with one ZMM load.
      * Byte b of the block-group holds:
          bits[0..1] = block_0.weight[b]
          bits[2..3] = block_1.weight[b]
          bits[4..5] = block_2.weight[b]
          bits[6..7] = block_3.weight[b]
      * The N-dimension uses the same 4-col-grouped layout as the existing
        W2 kernel (kNCols4 = 4), so the main NMain region groups 4 N-cols
        per "row" of block-groups.

    Restrictions:

      * BlkLen ∈ {32, 64, 128}.
      * No constraint on BlockCountK. The pack-size helper rounds BlockCountK
        up to a multiple of kBlockGroupBlks = 4 internally, and the SIMD
        K-loop processes the padded blocks via the 4-block accumulator with
        zero ZMM for the missing A blocks (K-tail handler).
      * Typical W2 production K dimensions (e.g. 384, 1024, 4096) at
        BlkLen=64 are all multiples of 256 (= kBlockGroupBlks * 64) and so
        do not exercise the K-tail handler; smaller K (or BlkLen=32 / 128
        shapes whose K is not a multiple of kBlockGroupBlks * BlkLen) take
        the K-tail path and are exercised by the unit tests.

--*/

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "mlas.h"
#include "mlas_qnbit.h"

template <typename T, int BlkBitWidth>
struct PackedQuantBDataStruct;  // fwd decl, defined in qnbitgemm.h

struct MLAS_BACKEND_KERNEL_SELECTOR_CONFIG;

namespace onnxruntime {
namespace mlas {
namespace sq2bit_avx512 {

// -----------------------------------------------------------------------------
// Block / block-group constants (BlkLen=64 W2 native path).
// -----------------------------------------------------------------------------

// Each 2-bit weight occupies 2 bits; one byte holds 4 weights.
constexpr size_t kWeightsPerByte = 4;

// Block constants for the BlkLen=64 variant.
constexpr size_t kBlkLen = 64;
constexpr size_t kBlkBytes = kBlkLen / kWeightsPerByte;  // 16 packed src bytes per block

// Default zero point used when the input is symmetric (no zero-point tensor).
// For 2-bit unsigned values in [0, 3], the symmetric mid-point is 2.
constexpr uint8_t kDefaultSymmetricZeroPoint2Bit = 2;

// Tile shape used by the SIMD kernel; the pack layout is keyed off these.
constexpr size_t kNCols4 = 4;

// block-group constants. 4 consecutive K-blocks share a single 64-byte buffer
// so the SIMD unpack is one ZMM load + four fixed shift/mask pairs.
constexpr size_t kBlockGroupBlks = 4;                                // 4 K-blocks per group
constexpr size_t kBlockGroupBytes = kBlockGroupBlks * kBlkBytes;     // 64 bytes
constexpr size_t kBlockGroupWeights = kBlockGroupBlks * kBlkLen;     // 256 weights

// Effective (padded) BlockCountK of the packed layout: the dispatch tables
// report the per-column K stride rounded up to a whole block-group.
constexpr size_t
Q2BitGemmEffectiveBlockCountK(size_t BlockCountK)
{
    return ((BlockCountK + kBlockGroupBlks - 1) / kBlockGroupBlks) * kBlockGroupBlks;
}

//
// Extract a single 2-bit weight from a standard ONNX MatMulNBits packed byte
// stream. `src` is the start of one block (kBlkBytes bytes). `i` is the
// in-block weight index in [0, kBlkLen).
//
inline uint8_t
ExtractSrcWeight(const std::byte* src, size_t i)
{
    const size_t byte_idx = i / kWeightsPerByte;
    const size_t bit_off = (i % kWeightsPerByte) * 2;
    return static_cast<uint8_t>(
        (static_cast<uint8_t>(src[byte_idx]) >> bit_off) & 0x03u
    );
}

//
// Pack 4 consecutive K-blocks (4 * 16 = 64 source bytes in standard ONNX
// layout) into a 64-byte block-group. Pure permutation of the 256 2-bit
// elements; bit-identical round-trip with UnPackBlockGroup_BlkLen64_Reference.
//
//   src_byte[i] holds val[4i .. 4i+3] at bit positions {0..1, 2..3, 4..5, 6..7}.
//   dst_byte[i] holds block0.weight[i], block1.weight[i], block2.weight[i],
//                    block3.weight[i] at the same bit positions.
//
inline void
PackBlockGroup_BlkLen64(const std::byte* src_block_0,
                         const std::byte* src_block_1,
                         const std::byte* src_block_2,
                         const std::byte* src_block_3,
                         std::byte* dst)
{
    for (size_t i = 0; i < kBlkLen; ++i) {
        const uint8_t v0 = ExtractSrcWeight(src_block_0, i);
        const uint8_t v1 = ExtractSrcWeight(src_block_1, i);
        const uint8_t v2 = ExtractSrcWeight(src_block_2, i);
        const uint8_t v3 = ExtractSrcWeight(src_block_3, i);
        dst[i] = static_cast<std::byte>(
            static_cast<uint8_t>(v0 | (v1 << 2) | (v2 << 4) | (v3 << 6))
        );
    }
}

//
// Reference unpack of one 64-byte block-group back into 4 K-blocks worth of
// natural-order uint8 weights ([0, 3]). Written from the documented layout
// rule -- intentionally independent of PackBlockGroup_BlkLen64 so it can
// serve as a round-trip oracle.
//
inline void
UnPackBlockGroup_BlkLen64_Reference(const std::byte* packed,
                                     uint8_t out_block_0[kBlkLen],
                                     uint8_t out_block_1[kBlkLen],
                                     uint8_t out_block_2[kBlkLen],
                                     uint8_t out_block_3[kBlkLen])
{
    for (size_t i = 0; i < kBlkLen; ++i) {
        const uint8_t b = static_cast<uint8_t>(packed[i]);
        out_block_0[i] = static_cast<uint8_t>((b >> 0) & 0x03u);
        out_block_1[i] = static_cast<uint8_t>((b >> 2) & 0x03u);
        out_block_2[i] = static_cast<uint8_t>((b >> 4) & 0x03u);
        out_block_3[i] = static_cast<uint8_t>((b >> 6) & 0x03u);
    }
}

//
// Extract all 64 weights from a standard ONNX MatMulNBits source block into
// natural order. Used by tests as the "expected" sequence for round-tripping.
//
inline void
UnpackSourceBlock_BlkLen64_Reference(const std::byte* src, uint8_t out[kBlkLen])
{
    for (size_t i = 0; i < kBlkLen; ++i) {
        out[i] = ExtractSrcWeight(src, i);
    }
}

// -----------------------------------------------------------------------------
// BlkLen=128 constants and pack helpers (parallel to the BlkLen=64 set above).
// The block-group still aggregates 4 K-blocks; only the per-block width
// (and hence the per-group byte count) changes. The N-major / 4-col-grouped
// layout structure, the K-tail rounding rule, and the SGEMM correction step
// are identical to BlkLen=64.
// -----------------------------------------------------------------------------

constexpr size_t kBlkLen128 = 128;
constexpr size_t kBlkBytes128 = kBlkLen128 / kWeightsPerByte;             // 32 packed src bytes per block
constexpr size_t kBlockGroupBytes128 = kBlockGroupBlks * kBlkBytes128;    // 128 bytes per block-group
constexpr size_t kBlockGroupWeights128 = kBlockGroupBlks * kBlkLen128;    // 512 weights per block-group

//
// Pack 4 consecutive K-blocks (4 * 32 = 128 source bytes in standard ONNX
// layout) into a 128-byte block-group. Identical bit-layout rule as the
// BlkLen=64 variant: byte b of the destination holds bits[0..1] of block_0's
// weight[b], bits[2..3] of block_1's weight[b], etc. Only the byte count
// (= kBlkLen128 = 128) differs.
//
inline void
PackBlockGroup_BlkLen128(const std::byte* src_block_0,
                         const std::byte* src_block_1,
                         const std::byte* src_block_2,
                         const std::byte* src_block_3,
                         std::byte* dst)
{
    for (size_t i = 0; i < kBlkLen128; ++i) {
        const uint8_t v0 = ExtractSrcWeight(src_block_0, i);
        const uint8_t v1 = ExtractSrcWeight(src_block_1, i);
        const uint8_t v2 = ExtractSrcWeight(src_block_2, i);
        const uint8_t v3 = ExtractSrcWeight(src_block_3, i);
        dst[i] = static_cast<std::byte>(
            static_cast<uint8_t>(v0 | (v1 << 2) | (v2 << 4) | (v3 << 6))
        );
    }
}

//
// Reference unpack of one 128-byte block-group back into 4 K-blocks worth of
// natural-order uint8 weights ([0, 3]). Independent of PackBlockGroup_BlkLen128
// so it can serve as a round-trip oracle.
//
inline void
UnPackBlockGroup_BlkLen128_Reference(const std::byte* packed,
                                     uint8_t out_block_0[kBlkLen128],
                                     uint8_t out_block_1[kBlkLen128],
                                     uint8_t out_block_2[kBlkLen128],
                                     uint8_t out_block_3[kBlkLen128])
{
    for (size_t i = 0; i < kBlkLen128; ++i) {
        const uint8_t b = static_cast<uint8_t>(packed[i]);
        out_block_0[i] = static_cast<uint8_t>((b >> 0) & 0x03u);
        out_block_1[i] = static_cast<uint8_t>((b >> 2) & 0x03u);
        out_block_2[i] = static_cast<uint8_t>((b >> 4) & 0x03u);
        out_block_3[i] = static_cast<uint8_t>((b >> 6) & 0x03u);
    }
}

//
// Byte offset for the BlkLen=128 packed B-data buffer. Same shape as
// PackedQuantBOffsetBytes_W2 (the BlkLen=64 variant just above) but using
// kBlockGroupBytes128 (= 128) per slot instead of kBlockGroupBytes (= 64).
// The N-major / 4-col-grouped layout is identical so the dispatcher's
// per-N-tile pointer arithmetic and the N-tail handling are reusable.
//
inline size_t
PackedQuantBOffsetBytes_W2_BlkLen128(size_t n, size_t blk_group,
                                     size_t BlockGroupCountKPadded, size_t NMain)
{
    if (n < NMain) {
        const size_t g = n / kNCols4;
        const size_t c = n % kNCols4;
        const size_t per_group_bytes = BlockGroupCountKPadded * kNCols4 * kBlockGroupBytes128;
        return g * per_group_bytes
             + blk_group * (kNCols4 * kBlockGroupBytes128)
             + c * kBlockGroupBytes128;
    }
    return (n * BlockGroupCountKPadded + blk_group) * kBlockGroupBytes128;
}

//
// NOTE on scale offsets:
// PackedQuantBScaleOffset_W2 is BlkLen-independent (one float per K-block,
// the layout depends only on kBlockGroupBlks and kNCols4). The BlkLen=128
// path reuses it verbatim; no PackedQuantBScaleOffset_W2_BlkLen128 needed.
//

// -----------------------------------------------------------------------------
// BlkLen=32 constants and pack helpers (parallel to the BlkLen=64/128 sets).
// 4 K-blocks * 32 weights = 128 weights per group = 32 packed bytes per group
// (fits in one YMM). Same N-major / 4-col-grouped layout. Same K-tail rounding
// rule (round BlockCountK up to a multiple of kBlockGroupBlks=4).
// -----------------------------------------------------------------------------

constexpr size_t kBlkLen32 = 32;
constexpr size_t kBlkBytes32 = kBlkLen32 / kWeightsPerByte;             // 8 packed src bytes per block
constexpr size_t kBlockGroupBytes32 = kBlockGroupBlks * kBlkBytes32;    // 32 bytes per block-group
constexpr size_t kBlockGroupWeights32 = kBlockGroupBlks * kBlkLen32;    // 128 weights per block-group

inline void
PackBlockGroup_BlkLen32(const std::byte* src_block_0,
                        const std::byte* src_block_1,
                        const std::byte* src_block_2,
                        const std::byte* src_block_3,
                        std::byte* dst)
{
    for (size_t i = 0; i < kBlkLen32; ++i) {
        const uint8_t v0 = ExtractSrcWeight(src_block_0, i);
        const uint8_t v1 = ExtractSrcWeight(src_block_1, i);
        const uint8_t v2 = ExtractSrcWeight(src_block_2, i);
        const uint8_t v3 = ExtractSrcWeight(src_block_3, i);
        dst[i] = static_cast<std::byte>(
            static_cast<uint8_t>(v0 | (v1 << 2) | (v2 << 4) | (v3 << 6))
        );
    }
}

inline void
UnPackBlockGroup_BlkLen32_Reference(const std::byte* packed,
                                    uint8_t out_block_0[kBlkLen32],
                                    uint8_t out_block_1[kBlkLen32],
                                    uint8_t out_block_2[kBlkLen32],
                                    uint8_t out_block_3[kBlkLen32])
{
    for (size_t i = 0; i < kBlkLen32; ++i) {
        const uint8_t b = static_cast<uint8_t>(packed[i]);
        out_block_0[i] = static_cast<uint8_t>((b >> 0) & 0x03u);
        out_block_1[i] = static_cast<uint8_t>((b >> 2) & 0x03u);
        out_block_2[i] = static_cast<uint8_t>((b >> 4) & 0x03u);
        out_block_3[i] = static_cast<uint8_t>((b >> 6) & 0x03u);
    }
}

inline size_t
PackedQuantBOffsetBytes_W2_BlkLen32(size_t n, size_t blk_group,
                                    size_t BlockGroupCountKPadded, size_t NMain)
{
    if (n < NMain) {
        const size_t g = n / kNCols4;
        const size_t c = n % kNCols4;
        const size_t per_group_bytes = BlockGroupCountKPadded * kNCols4 * kBlockGroupBytes32;
        return g * per_group_bytes
             + blk_group * (kNCols4 * kBlockGroupBytes32)
             + c * kBlockGroupBytes32;
    }
    return (n * BlockGroupCountKPadded + blk_group) * kBlockGroupBytes32;
}

// -----------------------------------------------------------------------------
// block-group packed-data layout.
//
//   Main region (n < NMain = floor(N / kNCols4) * kNCols4):
//     4-N-col groups of g = n / 4, col within group c = n % 4.
//     K-block-group index s = blk / 4 (s in [0, BlockCountK / 4)).
//     Within a group, block-groups run consecutively across the 4 cols,
//     so each (s, group) slot is a contiguous (kNCols4 * kBlockGroupBytes)
//     = 256 byte chunk.
//
//   Tail region (n >= NMain): plain column-major block-groups, identical
//   shape to the main region but flat in N.
//
// K-tail handling (BlockCountK not a multiple of kBlockGroupBlks):
//   The pack helpers round BlockCountK up to a multiple of kBlockGroupBlks
//   (= 4) for storage purposes -- the padding 1-3 blocks at the trailing
//   block-group contain zeroed weight bytes and zeroed scales, so they
//   contribute 0 to the integer dot product and 0 to the BlkSum correction.
//   This lets the SIMD kernel iterate the block-group K-loop without a
//   special tail handler for B, and avoids dual packing layouts. Storage
//   waste is at most (kBlockGroupBlks - 1) blocks per N-col, i.e. <= 48
//   bytes per col -- negligible at any realistic N.
//
//   Conventions used by the offset helpers below:
//     * `BlockGroupCountKPadded = ceil(BlockCountK / kBlockGroupBlks)` is
//       the number of block-groups the kernel actually iterates.
//     * `BlockCountKPadded = BlockGroupCountKPadded * kBlockGroupBlks` is
//       the K-block count used to address the scale buffer.
//     * Callers must pass `BlockGroupCountKPadded` and `BlockCountKPadded`
//       to these helpers; the original logical BlockCountK is only used
//       for sizing the BlkSum buffer (which is consumed by the SGEMM
//       correction step, not the inner K-loop).
//
// Caller-side constraints: BlkLen == 64; BlockCountK >= 1 (any K, padded
// internally to a multiple of kBlockGroupBlks).
// -----------------------------------------------------------------------------

inline size_t
PackedQuantBOffsetBytes_W2(size_t n, size_t blk_group,
                                      size_t BlockGroupCountKPadded, size_t NMain)
{
    if (n < NMain) {
        const size_t g = n / kNCols4;
        const size_t c = n % kNCols4;
        const size_t per_group_bytes = BlockGroupCountKPadded * kNCols4 * kBlockGroupBytes;
        return g * per_group_bytes
             + blk_group * (kNCols4 * kBlockGroupBytes)
             + c * kBlockGroupBytes;
    }
    return (n * BlockGroupCountKPadded + blk_group) * kBlockGroupBytes;
}

//
// Float offset into the packed B-scale buffer for a logical (n, blk) cell.
// Scales remain per-block (one float per K-block), 4 per group. Caller
// passes BlockCountKPadded (= BlockGroupCountKPadded * kBlockGroupBlks);
// scale slots in [BlockCountK, BlockCountKPadded) contain zeros so the
// kernel can index uniformly.
//
inline size_t
PackedQuantBScaleOffset_W2(size_t n, size_t blk,
                                      size_t BlockCountKPadded, size_t NMain)
{
    const size_t BlockGroupCountKPadded = BlockCountKPadded / kBlockGroupBlks;
    const size_t blk_group = blk / kBlockGroupBlks;
    const size_t blk_in_group = blk % kBlockGroupBlks;
    if (n < NMain) {
        const size_t g = n / kNCols4;
        const size_t c = n % kNCols4;
        const size_t per_group_scales = BlockGroupCountKPadded * kNCols4 * kBlockGroupBlks;
        return g * per_group_scales
             + blk_group * (kNCols4 * kBlockGroupBlks)
             + c * kBlockGroupBlks
             + blk_in_group;
    }
    return n * BlockCountKPadded + blk;
}

//
// Reference (scalar) entry points -- defined in sqnbitgemm_kernel_avx512_2bit.cpp.
//
// These cover Pack helpers and scalar oracle. They are
// reachable from unit tests via direct linkage; production dispatch wiring
// is performed by the platform dispatcher.
//

size_t MLASCALL
Q2BitGemmPackQuantBDataSize_Avx512(
    size_t N,
    size_t K,
    size_t BlkLen,
    bool HasZeroPoint,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
);

void MLASCALL
SQ2BitGemmPackQuantBDataAndBlkSum_Scalar(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const std::byte* QuantBDataBegin,
    const float* QuantBScaleBegin,
    bool HasZeroPoint,
    const std::byte* QuantBZPBegin,
    PackedQuantBDataStruct<float, 2>& PackedQuantB,
    MLAS_THREADPOOL* ThreadPool,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
);

size_t MLASCALL
SQ2BitGemmKernel_BlkSum_CompInt8_Scalar(
    size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum
);

//
// BlkLen-routing wrappers for the W2 CompInt8 dispatch entries.
// Production code calls these via the MLAS dispatch table
// (MlasSQNBitGemmDispatchAvx512 / Avx512vnni); tests call them directly
// via the namespace. The caller MUST verify
// GetMlasPlatform().Avx512Supported_ (and, for the VNNI variant, that the
// active dispatch is the AVX-512-VNNI one) before invoking these symbols.
//
size_t MLASCALL
SQ2BitGemmKernel_BlkSum_CompInt8_Avx512_Dispatch(
    size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum
);

size_t MLASCALL
SQ2BitGemmKernel_BlkSum_CompInt8_Avx512Vnni_Dispatch(
    size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum
);

}  // namespace sq2bit_avx512
}  // namespace mlas
}  // namespace onnxruntime
