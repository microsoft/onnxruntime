/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx512_2bit.cpp

Abstract:

    Phase 2b reference implementation of the 2-bit weight CompInt8 GEMM
    (BlkBitWidth=2, BlkLen=64). This file contains only scalar C++ code; the
    AVX-512-VNNI vectorized inner loop lands in Phase 3 as a separate header
    (sqnbitgemm_kernel_avx512vnni_2bit_blklen64.h) that replaces the kernel
    slot in the AVX-512-VNNI dispatch table.

    The scalar functions exposed here are linked into the AVX-512-VNNI
    dispatch table only (the plain AVX-512 dispatch leaves the W2 slots
    null so non-VNNI hosts fall through to the existing LUT kernel). They
    are also reachable as plain C++ symbols from unit tests, which use them
    as a correctness oracle for the vectorized path.

    Restrictions (Phase 2):
      * BlkLen == 64 only. Other BlkLens are rejected by the pack helper
        and the kernel returns 0 rows handled.
      * Symmetric quantization only (no per-block zero-point tensor;
        an implicit zero-point of 2 is used to recentre values in [0, 3]).

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
// Alignment slack is added so that the AVX-512 dequant in Phase 3 can use
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
    // Phase 2 supports only BlkLen=64 and SQNBIT_CompInt8. Anything else returns
    // 0 so MlasQNBitGemmPackQuantBDataSize reports an unsupported configuration.
    if (BlkLen != kBlkLen || ComputeType != SQNBIT_CompInt8) {
        return 0;
    }

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    size_t PackedQuantBDataSize = N * BlockCountK * kBlkBytes;
    const size_t ScaleSize = N * BlockCountK * sizeof(float);
    size_t BlkSumSize = MlasDivRoundup(N, 16) * BlockCountK * 16 * sizeof(float);

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
// QuantBZPBegin / HasZeroPoint are accepted for ABI parity with the W4 path
// but ignored in Phase 2 because the customer model and the projection
// rely on the symmetric layout.
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
    const std::byte* /* QuantBZPBegin */,
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
    const size_t Iterations = N * BlockCountK;

    // Pack weight bytes (block-by-block, parallel over N * BlockCountK).
    if (QuantBDataBegin != nullptr) {
        std::byte* PackedQuantBData = PackedQuantB.PackedQuantBData;
        MlasTrySimpleParallel(
            ThreadPool, static_cast<ptrdiff_t>(Iterations),
            [&](ptrdiff_t tid) {
                const size_t n = static_cast<size_t>(tid) / BlockCountK;
                const size_t blk = static_cast<size_t>(tid) % BlockCountK;
                const size_t offset = (n * BlockCountK + blk) * kBlkBytes;
                PackBlock_BlkLen64(QuantBDataBegin + offset, PackedQuantBData + offset);
            }
        );
    }

    // Copy scales as-is (column-major) and compute BlkSum.
    //
    // BlkSum uses the W4-style "width-16 row-major chunked" layout because the
    // top-level kernel performs the zero-point correction via the float SGEMM
    // micro-kernel (`GetMlasPlatform().GemmFloatKernel`), which expects this
    // pre-packed B layout:
    //
    //     BlkSum[(n / 16) * BlockCountK * 16 + blk * 16 + (n % 16)]
    //          = -scale_b * 2   (symmetric W2 uses an implicit ZP of 2)
    //
    // The allocated BlkSum buffer is sized at MlasDivRoundup(N, 16) * BlockCountK
    // * 16 floats so the layout is well-defined even when N % 16 != 0 (the tail
    // chunk's unused lanes hold whatever the buffer was initialised with, which
    // for production callers must be zero so the SGEMM correction reads zeros).
    if (QuantBScaleBegin != nullptr) {
        float* PackedScales = PackedQuantB.PackedQuantBScale;
        float* BlkSum = PackedQuantB.QuantBBlkSum;
        MlasTrySimpleParallel(
            ThreadPool, static_cast<ptrdiff_t>(Iterations),
            [&](ptrdiff_t tid) {
                const size_t n = static_cast<size_t>(tid) / BlockCountK;
                const size_t blk = static_cast<size_t>(tid) % BlockCountK;
                const float scale = QuantBScaleBegin[n * BlockCountK + blk];
                PackedScales[n * BlockCountK + blk] = scale;
                const size_t blksum_offset = ((n / 16) * BlockCountK + blk) * 16 + (n % 16);
                BlkSum[blksum_offset] = -scale * static_cast<float>(kDefaultSymmetricZeroPoint2Bit);
            }
        );
    }
}

//
// Scalar reference kernel for SQ2BitGemmVariant_CompInt8.
//
// Inputs match the SQ4BitGemmKernel_BlkSum_CompInt8_Fn typedef so the
// vectorized Phase 3 implementation can drop into the same dispatch slot.
//
// Math:
//   C[m, n] = bias[n]
//           + sum_blk( scale_a[m, blk] * scale_b[n, blk]
//                      * dot(int8 a[m, blk, :], uint8 b_unpacked[n, blk, :]) )
//           + sum_blk( ABlockSum[m, blk] * QuantBBlkSum[n, blk] )
//
// The third term applies the symmetric W2 zero-point correction:
//   QuantBBlkSum[n, blk] = -scale_b * 2, and ABlockSum[m, blk] = scale_a * sum(a).
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
        return 0;  // Phase 2b only supports BlkLen=64.
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

                // Symmetric W2 zero-point correction:
                // dot(a, b_signed) = dot(a, b_unsigned) - zp * sum(a)
                // so we need C += scale_a * scale_b * (-zp) * sum(a)
                //          = ABlockSum * QuantBBlkSum (where QuantBBlkSum already encodes -scale_b * zp).
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
