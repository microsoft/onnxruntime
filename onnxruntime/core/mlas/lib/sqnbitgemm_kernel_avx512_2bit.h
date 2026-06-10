/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx512_2bit.h

Abstract:

    Pack-time helpers and reference (scalar) routines for the 2-bit, BlkLen=64
    AVX-512 weight GEMM path (SQNBIT_CompInt8, BlkBitWidth=2). The vectorized
    kernels (AVX-512 and AVX-512-VNNI variants) live in
    sqnbitgemm_kernel_avx512vnni_2bit_blklen64.h.

    This header is scalar / header-only and contains:

      * The packed-block layout used by the AVX-512 dequant inner loop.
      * A pack routine that converts standard ONNX MatMulNBits 2-bit input data
        into the packed layout.
      * A reference unpack routine (independent of the pack code) that
        materialises one packed block back into 64 individual int8 values.

    The packed layout is designed so that the AVX-512BW dequant is one
    128-bit broadcast plus a per-lane variable shift:

        __m128i p   = _mm_loadu_si128(packed);               // 16 bytes
        __m512i p4  = _mm512_broadcast_i32x4(p);             // 4 lanes
        __m512i sh  = _mm512_set_epi32(6,6,6,6, 4,4,4,4,
                                       2,2,2,2, 0,0,0,0);
        __m512i v   = _mm512_srlv_epi32(p4, sh);             // per-lane shift
        v           = _mm512_and_si512(v, _mm512_set1_epi8(0x03));

    With this layout the resulting ZMM holds weights 0..63 in their natural
    order.

    NOTE: This file intentionally restricts itself to BlkLen == 64. Other
    BlkLens fall through to the existing LUT path until later phases.

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

// Each 2-bit weight occupies 2 bits; one byte holds 4 weights.
constexpr size_t kWeightsPerByte = 4;

// Block constants for the BlkLen=64 variant.
constexpr size_t kBlkLen = 64;
constexpr size_t kBlkBytes = kBlkLen / kWeightsPerByte;  // 16 packed src bytes per block
constexpr size_t kPackedBlkBytes = kBlkBytes;             // packing is in-place: 16 -> 16

// Default zero point used when the input is symmetric (no zero-point tensor).
// For 2-bit unsigned values in [0, 3], the symmetric mid-point is 2.
constexpr uint8_t kDefaultSymmetricZeroPoint2Bit = 2;

// -----------------------------------------------------------------------------
// Tile shape (must match the AVX-512-VNNI kernel in
// sqnbitgemm_kernel_avx512vnni_2bit_blklen64.h). The pack layout below is
// keyed off these values: the main NMain = floor(N / kNCols4) * kNCols4 cols
// are stored in a 4-col-grouped + 2-K-block-paired arrangement that lets the
// kernel's R2xC4 hot loop read each tile of B as a single contiguous stream,
// matching W4's pack layout in PackQuantB (sqnbitgemm_kernel_avx_common.h).
// -----------------------------------------------------------------------------

constexpr size_t kNCols4 = 4;
constexpr size_t kPerAccuBlk2 = 2;

//
// Byte offset into the packed B-data buffer for a logical (n, blk) cell.
//
//   Main region (n < NMain = floor(N/kNCols4)*kNCols4):
//     4-N-col group g = n / 4, col within group c = n % 4.
//     K-block-pair p = blk / 2, block within pair = blk % 2.
//     Pair slots run first, then a single-block trailing slot when
//     BlockCountK is odd.
//
//   Tail region (n >= NMain): plain column-major. The tail base lies
//   exactly at NMain * BlockCountK * kBlkBytes, so the dispatcher's
//   `multipleCols * ColStrideBytes` offset (where ColStrideBytes is
//   BlockCountK * kBlkBytes) is unchanged from the column-major layout.
//
inline size_t
PackedQuantBOffsetBytes_W2(size_t n, size_t blk, size_t BlockCountK, size_t NMain)
{
    if (n < NMain) {
        const size_t g = n / kNCols4;
        const size_t c = n % kNCols4;
        const size_t per_group_bytes = BlockCountK * kNCols4 * kBlkBytes;
        const size_t pair_idx = blk / kPerAccuBlk2;
        const size_t blk_in_pair = blk % kPerAccuBlk2;
        const size_t full_pairs = BlockCountK / kPerAccuBlk2;
        if (pair_idx < full_pairs) {
            return g * per_group_bytes
                 + pair_idx * (kNCols4 * kPerAccuBlk2 * kBlkBytes)
                 + c * (kPerAccuBlk2 * kBlkBytes)
                 + blk_in_pair * kBlkBytes;
        }
        return g * per_group_bytes
             + full_pairs * (kNCols4 * kPerAccuBlk2 * kBlkBytes)
             + c * kBlkBytes;
    }
    return (n * BlockCountK + blk) * kBlkBytes;
}

//
// Float offset into the packed B-scale buffer for a logical (n, blk) cell.
// Same grouping rule as the B-data, two scales per pair, one scale per
// single-block trailing slot.
//
inline size_t
PackedQuantBScaleOffset_W2(size_t n, size_t blk, size_t BlockCountK, size_t NMain)
{
    if (n < NMain) {
        const size_t g = n / kNCols4;
        const size_t c = n % kNCols4;
        const size_t per_group_scales = BlockCountK * kNCols4;
        const size_t pair_idx = blk / kPerAccuBlk2;
        const size_t blk_in_pair = blk % kPerAccuBlk2;
        const size_t full_pairs = BlockCountK / kPerAccuBlk2;
        if (pair_idx < full_pairs) {
            return g * per_group_scales
                 + pair_idx * (kNCols4 * kPerAccuBlk2)
                 + c * kPerAccuBlk2
                 + blk_in_pair;
        }
        return g * per_group_scales
             + full_pairs * (kNCols4 * kPerAccuBlk2)
             + c;
    }
    return n * BlockCountK + blk;
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
// Pack one source block (16 bytes = 64 2-bit weights, standard ONNX layout)
// into the destination layout described at the top of this file.
//
//   src_byte[i] holds val[4i .. 4i+3] at bit positions {0..1, 2..3, 4..5, 6..7}.
//   dst_byte[i] holds val[i], val[i+16], val[i+32], val[i+48] at the same bit
//                positions.
//
// This is a pure permutation of the 64 2-bit elements; the bit width and
// count are preserved (16 bytes in, 16 bytes out).
//
inline void
PackBlock_BlkLen64(const std::byte* src, std::byte* dst)
{
    for (size_t i = 0; i < kBlkBytes; ++i) {
        const uint8_t v0 = ExtractSrcWeight(src, i + 0);
        const uint8_t v1 = ExtractSrcWeight(src, i + 16);
        const uint8_t v2 = ExtractSrcWeight(src, i + 32);
        const uint8_t v3 = ExtractSrcWeight(src, i + 48);
        dst[i] = static_cast<std::byte>(
            static_cast<uint8_t>(v0 | (v1 << 2) | (v2 << 4) | (v3 << 6))
        );
    }
}

//
// Reference unpack of one packed block back into 64 int8 values in natural
// order ([val0, val1, ..., val63]).
//
// This routine is intentionally written without reference to PackBlock_BlkLen64
// so that it can serve as an independent oracle for round-trip tests:
// it simply applies the documented dst_byte layout rule in reverse.
//
inline void
UnpackBlock_BlkLen64_Reference(const std::byte* packed, uint8_t out[kBlkLen])
{
    for (size_t i = 0; i < kBlkBytes; ++i) {
        const uint8_t b = static_cast<uint8_t>(packed[i]);
        out[i + 0]  = static_cast<uint8_t>((b >> 0) & 0x03u);
        out[i + 16] = static_cast<uint8_t>((b >> 2) & 0x03u);
        out[i + 32] = static_cast<uint8_t>((b >> 4) & 0x03u);
        out[i + 48] = static_cast<uint8_t>((b >> 6) & 0x03u);
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
// EXPERIMENTAL: super-block packing for fast unpack
// -----------------------------------------------------------------------------
//
// The existing per-block pack (PackBlock_BlkLen64) stores 16 bytes per K-block
// such that each byte holds 4 weights from 4 different "rows" of the block.
// Unpacking that layout into a ZMM of 64 natural-order weights requires a
// broadcast + per-dword variable shift (`vpsrlvd`, ~3c latency, 1c throughput
// on Zen5) -- a cost we measured at ~30-35% of total W2 kernel time.
//
// The super-block layout below groups FOUR consecutive K-blocks together into
// a single 64-byte buffer (= 4 * 16 bytes, same total storage) such that
// byte[i] (i = 0..63) holds:
//   bits[0..1] = block_0.weight[i]
//   bits[2..3] = block_1.weight[i]
//   bits[4..5] = block_2.weight[i]
//   bits[6..7] = block_3.weight[i]
//
// Unpack with a single ZMM load and four fixed-shift+mask pairs:
//
//   __m512i super = _mm512_loadu_si512(packed);
//   __m512i mask  = _mm512_set1_epi8(0x03);
//   __m512i bv0   = _mm512_and_si512(super, mask);
//   __m512i bv1   = _mm512_and_si512(_mm512_srli_epi16(super, 2), mask);
//   __m512i bv2   = _mm512_and_si512(_mm512_srli_epi16(super, 4), mask);
//   __m512i bv3   = _mm512_and_si512(_mm512_srli_epi16(super, 6), mask);
//
// Each shift+mask is ~2c (1c srli_epi16 + 1c andd) and the four chains are
// fully independent, so the critical path is ~4c for ALL four blocks combined
// vs the current ~20c (4 broadcasts * ~5c each, partially overlapped).
//
// Note on `_mm512_srli_epi16`: it shifts each 16-bit lane by N. For a byte
// that is the LOW byte of a 16-bit lane, the shift pulls in bits from the
// adjacent HIGH byte. The subsequent AND with 0x03 discards those leaked
// bits, leaving the correct per-byte result. For the HIGH byte, zeros are
// shifted in from the top, which is what we want.

constexpr size_t kSuperBlockBlks = 4;                                // 4 K-blocks per super
constexpr size_t kSuperBlockBytes = kSuperBlockBlks * kBlkBytes;     // 64 bytes
constexpr size_t kSuperBlockWeights = kSuperBlockBlks * kBlkLen;     // 256 weights

//
// Pack 4 consecutive K-blocks (4 * 16 = 64 source bytes in standard ONNX
// layout) into a 64-byte super-block. Pure permutation of the 256 2-bit
// elements; bit-identical round-trip with UnpackSuperBlock4_BlkLen64_Reference.
//
inline void
PackSuperBlock4_BlkLen64(const std::byte* src_block_0,
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
// Reference unpack of one 64-byte super-block back into 4 K-blocks worth of
// natural-order uint8 weights ([0, 3]). Written from the documented layout
// rule -- intentionally independent of PackSuperBlock4_BlkLen64 so it can
// serve as a round-trip oracle.
//
inline void
UnpackSuperBlock4_BlkLen64_Reference(const std::byte* packed,
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
// Reference / dispatch entry points defined in sqnbitgemm_kernel_avx512_2bit.cpp.
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
// Unit-test forwarders for the two AVX-512 vectorized kernel variants. These
// are non-inline symbols defined in `sqnbitgemm_kernel_avx512vnni.cpp` (VNNI)
// and `sqnbitgemm_kernel_avx512.cpp` (non-VNNI), each of which is compiled
// with the appropriate ISA flags. The test TU (which is NOT compiled with
// AVX-512 flags) calls these by `extern` linkage to exercise both kernels
// independently of the platform dispatcher.
//
// Callers MUST gate on `GetMlasPlatform().Avx512Supported_` before invoking
// these symbols, since they execute AVX-512BW (and, for the VNNI variant,
// AVX-512-VNNI) instructions.
//
size_t MLASCALL
SQ2BitGemmKernel_BlkSum_CompInt8_Avx512_TestEntry(
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
SQ2BitGemmKernel_BlkSum_CompInt8_Avx512Vnni_TestEntry(
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
