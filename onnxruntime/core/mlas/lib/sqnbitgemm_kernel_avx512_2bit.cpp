/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx512_2bit.cpp

Abstract:

    Reference implementation and pack-time helpers for the 2-bit weight
    CompInt8 GEMM (BlkBitWidth=2, BlkLen=64). Scalar-only; the vectorized
    inner loop lives in sqnbitgemm_kernel_avx512vnni_2bit_blklen64.h and is
    registered into the AVX-512 and AVX-512-VNNI dispatch tables (with VNNI
    / non-VNNI MAC variants templated from the same source).

    The scalar functions exposed here back the pack-time helpers used by
    both dispatch tables and are reachable as plain C++ symbols from unit
    tests, which use them as a correctness oracle for the vectorized paths.

    Restrictions:
      * BlkLen == 64 only. Other BlkLens are rejected by the pack helper
        and the kernel returns 0 rows handled.
      * Per-block zero-point input is supported (standard ONNX W2 layout,
        4 ZPs per packed byte along K). When no zero-point tensor is
        supplied the symmetric default of 2 is used.

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
// Workspace / pack-buffer size for the 2-bit CompInt8 path.
//
// Layout (in bytes):
//
//   [PackedQuantBData]  N * BlockCountK * kBlkBytes      (BlkLen / 4 bytes per block)
//   [BlkSum  (float)]   roundup_16(N) * BlockCountK
//   [Scales  (float)]   N * BlockCountK
//
// Alignment slack is added so that the AVX-512 dequant can use
// 64-byte aligned loads.
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
    // Only BlkLen=64 and SQNBIT_CompInt8 are supported. Anything else returns
    // 0 so MlasQNBitGemmPackQuantBDataSize reports an unsupported configuration.
    if (BlkLen != kBlkLen || ComputeType != SQNBIT_CompInt8) {
        return 0;
    }

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    // Pad BlockCountK to a multiple of 4 to match the W2 buffer ABI used by
    // PackedQuantBDataStruct (BlkBitWidth=2 always pads). The padding is a
    // few extra K-block slots per N-col (<= 48 bytes data + a few floats for
    // scales / BlkSum) -- negligible -- but lets the W2-v1 (this path) and
    // W2-v2 (super-block) pack helpers share a single buffer layout.
    const size_t BlockCountKPadded = ((BlockCountK + 3) / 4) * 4;
    size_t PackedQuantBDataSize = N * BlockCountKPadded * kBlkBytes;
    const size_t ScaleSize = N * BlockCountKPadded * sizeof(float);
    size_t BlkSumSize = MlasDivRoundup(N, 16) * BlockCountKPadded * 16 * sizeof(float);

    constexpr size_t kPackedQuantBDataAlignment = 64;  // AVX-512 friendly
    PackedQuantBDataSize += kPackedQuantBDataAlignment - 1;

    constexpr size_t kBlkSumAlignment = MlasQNBitQuantBBlkSumAlignment();
    BlkSumSize += kBlkSumAlignment - 1;

    return PackedQuantBDataSize + ScaleSize + BlkSumSize;
}

//
// Pack quantized B data + scales + per-block sums for the 2-bit kernel.
//
// Layouts produced (all column-major in N):
//   PackedQuantBData[n * BlockCountK * kBlkBytes + blk * kBlkBytes + i]
//       Block (n, blk), byte i of the kPackedBlkBytes layout (see header).
//   PackedQuantBScale[n * BlockCountK + blk]
//       Copy of the input scale; column-major.
//   QuantBBlkSum[n * BlockCountK + blk]
//       = -scale * 2  (symmetric W2 uses an implicit zero point of 2).
//
// When QuantBZPBegin is non-null, the per-block zero-point byte stream is in
// the standard ONNX MatMulNBits W2 layout: 4 zero-points per byte, packed
// along the K-block axis, row-major in N. Row stride is
// ZPCountK = ceil(BlockCountK / 4) bytes; the ZP for (n, blk) lives at byte
// index (n * ZPCountK + blk / 4), at bit offset (blk % 4) * 2.
//
// When QuantBZPBegin is null, we fall back to the symmetric default
// (kDefaultSymmetricZeroPoint2Bit = 2), preserving the prior behavior. The
// HasZeroPoint flag is informational only: if QuantBZPBegin is non-null we
// always consume it.
//
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
    assert(BlkLen == kBlkLen);
    if (BlkLen != kBlkLen) {
        return;
    }

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t NMain = (N / kNCols4) * kNCols4;
    const size_t Iterations = N * BlockCountK;

    // Pack weight bytes in the 4-col-grouped + 2-K-block-paired layout for
    // the main NMain cols; column-major for the tail N % 4 cols. See
    // PackedQuantBOffsetBytes_W2 in sqnbitgemm_kernel_avx512_2bit.h for the
    // exact mapping.
    if (QuantBDataBegin != nullptr) {
        std::byte* PackedQuantBData = PackedQuantB.PackedQuantBData;
        MlasTrySimpleParallel(
            ThreadPool, static_cast<ptrdiff_t>(Iterations),
            [&](ptrdiff_t tid) {
                const size_t n = static_cast<size_t>(tid) / BlockCountK;
                const size_t blk = static_cast<size_t>(tid) % BlockCountK;
                const size_t src_offset = (n * BlockCountK + blk) * kBlkBytes;
                const size_t dst_offset = PackedQuantBOffsetBytes_W2(n, blk, BlockCountK, NMain);
                PackBlock_BlkLen64(QuantBDataBegin + src_offset, PackedQuantBData + dst_offset);
            }
        );
    }

    // Scales follow the same 4-col-grouped layout (see PackedQuantBScaleOffset_W2).
    //
    // BlkSum uses the W4-style "width-16 row-major chunked" layout because the
    // top-level kernel performs the zero-point correction via the float SGEMM
    // micro-kernel (`GetMlasPlatform().GemmFloatKernel`), which expects this
    // pre-packed B layout:
    //
    //     BlkSum[(n / 16) * BlockCountK * 16 + blk * 16 + (n % 16)]
    //          = -scale_b * zero_point
    //
    // The allocated BlkSum buffer is sized at MlasDivRoundup(N, 16) * BlockCountK
    // * 16 floats so the layout is well-defined even when N % 16 != 0 (the tail
    // chunk's unused lanes hold whatever the buffer was initialised with, which
    // for production callers must be zero so the SGEMM correction reads zeros).
    //
    // Important: ORT's matmul_nbits.cc prepack flow invokes this function up to
    // three times per node — once each for B, scales, and zero_points — so on
    // any given invocation only one of (QuantBScaleBegin, QuantBZPBegin) may be
    // non-null. To get the correct BlkSum we therefore (a) copy scales into the
    // packed buffer when QuantBScaleBegin is provided, and (b) recompute BlkSum
    // whenever EITHER scales or zero-points are provided, reading the scales
    // from the already-packed PackedQuantBScale buffer (which is populated by a
    // previous call when only ZPs arrive in the current call). This mirrors
    // the W4 ComputePackBlkSum helper.

    if (QuantBScaleBegin != nullptr) {
        float* PackedScales = PackedQuantB.PackedQuantBScale;
        MlasTrySimpleParallel(
            ThreadPool, static_cast<ptrdiff_t>(Iterations),
            [&](ptrdiff_t tid) {
                const size_t n = static_cast<size_t>(tid) / BlockCountK;
                const size_t blk = static_cast<size_t>(tid) % BlockCountK;
                const float scale = QuantBScaleBegin[n * BlockCountK + blk];
                PackedScales[PackedQuantBScaleOffset_W2(n, blk, BlockCountK, NMain)] = scale;
            }
        );
    }

    // BlkSum needs to be (re)computed whenever scales or zero-points change.
    // Source of scales is the already-packed buffer, so this works even when
    // only zero_points are provided in the current invocation.
    if (QuantBScaleBegin != nullptr || QuantBZPBegin != nullptr) {
        const float* PackedScales = PackedQuantB.PackedQuantBScale;
        float* BlkSum = PackedQuantB.QuantBBlkSum;
        const size_t ZPCountK = MlasDivRoundup(BlockCountK, 4);
        MlasTrySimpleParallel(
            ThreadPool, static_cast<ptrdiff_t>(Iterations),
            [&](ptrdiff_t tid) {
                const size_t n = static_cast<size_t>(tid) / BlockCountK;
                const size_t blk = static_cast<size_t>(tid) % BlockCountK;
                const float scale =
                    PackedScales[PackedQuantBScaleOffset_W2(n, blk, BlockCountK, NMain)];

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
// Scalar reference kernel for SQ2BitGemmVariant_CompInt8.
//
// Inputs match the SQ4BitGemmKernel_BlkSum_CompInt8_Fn typedef so the
// vectorized implementation can drop into the same dispatch slot.
//
// Math:
//   C[m, n] = bias[n]
//           + sum_blk( scale_a[m, blk] * scale_b[n, blk]
//                      * dot(int8 a[m, blk, :], uint8 b_unpacked[n, blk, :]) )
//           + sum_blk( ABlockSum[m, blk] * QuantBBlkSum[n, blk] )
//
// The third term applies the W2 zero-point correction:
//   QuantBBlkSum[n, blk] = -scale_b * zp, and ABlockSum[m, blk] = scale_a * sum(a).
// `zp` is the per-block zero point baked in at pack time (defaults to
// kDefaultSymmetricZeroPoint2Bit = 2 when no zero-point tensor is supplied).
//
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
    if (BlkLen != kBlkLen) {
        return 0;  // Only BlkLen=64 is supported.
    }

    const size_t lda = BlockCountK * kBlkLen;       // bytes per A row (int8)
    const size_t lda_scale = BlockCountK;           // floats per A scale row
    const size_t ldb = BlockCountK * kBlkBytes;     // bytes per B column
    const size_t ldb_scale = BlockCountK;           // floats per B column

    for (size_t m = 0; m < CountM; ++m) {
        const int8_t* a_row = reinterpret_cast<const int8_t*>(QuantA + m * lda);
        const float* a_scale_row = QuantAScale + m * lda_scale;
        const float* a_blksum_row = ABlockSum + m * lda_scale;
        float* c_row = C + m * ldc;

        for (size_t n = 0; n < CountN; ++n) {
            const std::byte* b_col = QuantBData + n * ldb;
            const float* b_scale_col = QuantBScale + n * ldb_scale;
            const float* b_blksum_col = QuantBBlkSum + n * ldb_scale;

            float acc = (Bias != nullptr) ? Bias[n] : 0.0f;

            for (size_t blk = 0; blk < BlockCountK; ++blk) {
                // Unpack 64 2-bit weights into 64 uint8 values (values in [0, 3]).
                uint8_t b_unpacked[kBlkLen];
                UnpackBlock_BlkLen64_Reference(b_col + blk * kBlkBytes, b_unpacked);

                // int8 * uint8 dot product across the block.
                const int8_t* a_blk = a_row + blk * kBlkLen;
                int32_t dot = 0;
                for (size_t i = 0; i < kBlkLen; ++i) {
                    dot += static_cast<int32_t>(a_blk[i]) * static_cast<int32_t>(b_unpacked[i]);
                }

                // Integer term * scales.
                acc += a_scale_row[blk] * b_scale_col[blk] * static_cast<float>(dot);

                // W2 zero-point correction:
                // dot(a, b_signed) = dot(a, b_unsigned) - zp * sum(a)
                // so we need C += scale_a * scale_b * (-zp) * sum(a)
                //          = ABlockSum * QuantBBlkSum (QuantBBlkSum encodes -scale_b * zp,
                //            with zp = per-block ZP or 2 when no ZP tensor was supplied).
                acc += a_blksum_row[blk] * b_blksum_col[blk];
            }

            c_row[n] = acc;
        }
    }

    return CountM;
}

}  // namespace sq2bit_avx512
}  // namespace mlas
}  // namespace onnxruntime
