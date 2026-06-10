/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx512_2bit_superblock.h

Abstract:

    Pack-time helpers and scalar reference routines for the EXPERIMENTAL
    super-block W2 layout: groups of 4 K-blocks share a single 64-byte
    packed buffer that allows the AVX-512 unpack to be one ZMM load plus
    four fixed `vpsrlw+vpand` pairs (instead of the current per-block
    broadcast + variable shift).

    Layout summary (BlkLen=64 only):

      * Each "super-block" packs FOUR consecutive K-blocks (256 weights total).
      * Total storage per super-block = kBlkBytes * 4 = 64 bytes (identical to
        4 separately-packed blocks under the current scheme).
      * Byte b of the super-block holds:
          bits[0..1] = block_0.weight[b]
          bits[2..3] = block_1.weight[b]
          bits[4..5] = block_2.weight[b]
          bits[6..7] = block_3.weight[b]
      * The N-dimension uses the same 4-col-grouped layout as the existing
        W2 kernel (kNCols4 = 4), so the main NMain region groups 4 N-cols
        per "row" of super-blocks.

    Restrictions:

      * BlkLen == 64 only.
      * BlockCountK must be a multiple of kSuperBlockBlks = 4. The prototype
        path returns 0 from the pack-size helper for non-multiples; the
        caller falls back to the existing W2 path.
      * The customer model's K dimensions (384, 1024, 4096) are all multiples
        of 256 (= 4 * 64), so all customer shapes satisfy this constraint.

--*/

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "mlas.h"
#include "mlas_qnbit.h"

#include "sqnbitgemm_kernel_avx512_2bit.h"  // Re-uses kBlkLen, kBlkBytes, kNCols4, etc.

template <typename T, int BlkBitWidth>
struct PackedQuantBDataStruct;  // fwd decl, defined in qnbitgemm.h

struct MLAS_BACKEND_KERNEL_SELECTOR_CONFIG;

namespace onnxruntime {
namespace mlas {
namespace sq2bit_avx512_super {

using ::onnxruntime::mlas::sq2bit_avx512::kBlkBytes;
using ::onnxruntime::mlas::sq2bit_avx512::kBlkLen;
using ::onnxruntime::mlas::sq2bit_avx512::kDefaultSymmetricZeroPoint2Bit;
using ::onnxruntime::mlas::sq2bit_avx512::kNCols4;
using ::onnxruntime::mlas::sq2bit_avx512::kSuperBlockBytes;
using ::onnxruntime::mlas::sq2bit_avx512::kSuperBlockBlks;
using ::onnxruntime::mlas::sq2bit_avx512::kWeightsPerByte;

// -----------------------------------------------------------------------------
// Super-block packed-data layout.
//
//   Main region (n < NMain = floor(N / kNCols4) * kNCols4):
//     4-N-col groups of g = n / 4, col within group c = n % 4.
//     K-super-block index s = blk / 4 (s in [0, BlockCountK / 4)).
//     Within a group, super-blocks run consecutively across the 4 cols,
//     so each (s, group) slot is a contiguous (kNCols4 * kSuperBlockBytes)
//     = 256 byte chunk.
//
//   Tail region (n >= NMain): plain column-major super-blocks, identical
//   shape to the main region but flat in N.
//
// K-tail handling (BlockCountK not a multiple of kSuperBlockBlks):
//   The pack helpers round BlockCountK up to a multiple of kSuperBlockBlks
//   (= 4) for storage purposes -- the padding 1-3 blocks at the trailing
//   super-block contain zeroed weight bytes and zeroed scales, so they
//   contribute 0 to the integer dot product and 0 to the BlkSum correction.
//   This lets the SIMD kernel iterate the super-block K-loop without a
//   special tail handler for B, and avoids dual packing layouts. Storage
//   waste is at most (kSuperBlockBlks - 1) blocks per N-col, i.e. <= 48
//   bytes per col -- negligible at any realistic N.
//
//   Conventions used by the offset helpers below:
//     * `SuperBlockCountKPadded = ceil(BlockCountK / kSuperBlockBlks)` is
//       the number of super-blocks the kernel actually iterates.
//     * `BlockCountKPadded = SuperBlockCountKPadded * kSuperBlockBlks` is
//       the K-block count used to address the scale buffer.
//     * Callers must pass `SuperBlockCountKPadded` and `BlockCountKPadded`
//       to these helpers; the original logical BlockCountK is only used
//       for sizing the BlkSum buffer (which is consumed by the SGEMM
//       correction step, not the inner K-loop).
//
// Caller-side constraints: BlkLen == 64; BlockCountK >= 1 (any K, padded
// internally to a multiple of kSuperBlockBlks).
// -----------------------------------------------------------------------------

inline size_t
PackedQuantBOffsetBytes_W2_SuperBlock(size_t n, size_t blk_super,
                                      size_t SuperBlockCountKPadded, size_t NMain)
{
    if (n < NMain) {
        const size_t g = n / kNCols4;
        const size_t c = n % kNCols4;
        const size_t per_group_bytes = SuperBlockCountKPadded * kNCols4 * kSuperBlockBytes;
        return g * per_group_bytes
             + blk_super * (kNCols4 * kSuperBlockBytes)
             + c * kSuperBlockBytes;
    }
    return (n * SuperBlockCountKPadded + blk_super) * kSuperBlockBytes;
}

//
// Float offset into the packed B-scale buffer for a logical (n, blk) cell.
// Scales remain per-block (one float per K-block), 4 per super. Caller
// passes BlockCountKPadded (= SuperBlockCountKPadded * kSuperBlockBlks);
// scale slots in [BlockCountK, BlockCountKPadded) contain zeros so the
// kernel can index uniformly.
//
inline size_t
PackedQuantBScaleOffset_W2_SuperBlock(size_t n, size_t blk,
                                      size_t BlockCountKPadded, size_t NMain)
{
    const size_t SuperBlockCountKPadded = BlockCountKPadded / kSuperBlockBlks;
    const size_t blk_super = blk / kSuperBlockBlks;
    const size_t blk_in_super = blk % kSuperBlockBlks;
    if (n < NMain) {
        const size_t g = n / kNCols4;
        const size_t c = n % kNCols4;
        const size_t per_group_scales = SuperBlockCountKPadded * kNCols4 * kSuperBlockBlks;
        return g * per_group_scales
             + blk_super * (kNCols4 * kSuperBlockBlks)
             + c * kSuperBlockBlks
             + blk_in_super;
    }
    return n * BlockCountKPadded + blk;
}

//
// Reference (scalar) entry points -- defined in sqnbitgemm_kernel_avx512_2bit_superblock.cpp.
//
// These cover Phase 2 of the super-block prototype: pack + scalar GEMM oracle
// against which the SIMD inner loop (Phase 3) will be validated. They are
// reachable from unit tests via direct linkage; production dispatch wiring
// happens in Phase 4 after the SIMD path is correctness-clean.
//

size_t MLASCALL
Q2BitGemmPackQuantBDataSize_SuperBlock(
    size_t N,
    size_t K,
    size_t BlkLen,
    bool HasZeroPoint,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
);

void MLASCALL
SQ2BitGemmPackQuantBDataAndBlkSum_SuperBlockScalar(
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
SQ2BitGemmKernel_BlkSum_CompInt8_SuperBlockScalar(
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
// Unit-test forwarders for the AVX-512 SIMD super-block kernels. Same gating
// rules as the existing W2 test entries: the caller MUST verify
// GetMlasPlatform().Avx512Supported_ (and, for the VNNI variant, that the
// active dispatch is the AVX-512-VNNI one) before invoking these symbols.
//
size_t MLASCALL
SQ2BitGemmKernel_BlkSum_CompInt8_Super_Avx512_TestEntry(
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
SQ2BitGemmKernel_BlkSum_CompInt8_Super_Avx512Vnni_TestEntry(
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

}  // namespace sq2bit_avx512_super
}  // namespace mlas
}  // namespace onnxruntime
