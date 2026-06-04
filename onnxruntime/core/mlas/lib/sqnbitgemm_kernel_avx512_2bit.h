/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx512_2bit.h

Abstract:

    Pack-time helpers and reference (scalar) routines for the 2-bit, BlkLen=64
    AVX-512-VNNI weight GEMM path (SQNBIT_CompInt8, BlkBitWidth=2).

    This header is currently scalar / header-only and contains the pieces that
    the Phase 2a round-trip unit tests exercise:

      * The packed-block layout used by the (future) AVX-512 dequant inner loop.
      * A pack routine that converts standard ONNX MatMulNBits 2-bit input data
        into the packed layout.
      * A reference unpack routine (independent of the pack code) that
        materialises one packed block back into 64 individual int8 values.

    The packed layout is designed so that the AVX-512BW dequant in Phase 3 is
    one 128-bit broadcast plus a per-lane variable shift:

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

//
// Phase 2b reference / dispatch entry points.
// Defined in sqnbitgemm_kernel_avx512_2bit.cpp; registered into the
// AVX-512 / AVX-512-VNNI dispatch tables.
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

}  // namespace sq2bit_avx512
}  // namespace mlas
}  // namespace onnxruntime
