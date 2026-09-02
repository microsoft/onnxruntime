/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qnbitgemm_kernel_neon.cpp

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for ARM NEON.

--*/

#include "qnbitgemm_kernel_neon.h"

#include <arm_neon.h>

#include <cassert>
#include <numeric>
#include <vector>

#include "qnbitgemm.h"
#include "sqnbitgemm_q8_block.h"

// W2 block-group pack helpers and scalar reference kernel declared in the
// Avx512-named header are pure C++ (no x86 intrinsics). They serve as the
// cross-arch layout authority for W2 and are reused on ARM64 for pack-size /
// pack / layout; compute is provided by the native NEON DotProd kernel in
// sqnbitgemm_kernel_neon_int8_2bit.cpp.
#include "sqnbitgemm_kernel_avx512_2bit.h"

namespace sqnbitgemm_neon
{

namespace
{

//
// Quantized B data packing function implementation.
//

template <int BlkBitWidth, bool QuantAUnsigned>
size_t
QNBitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    bool HasZeroPoint,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
)
{
    if constexpr (BlkBitWidth == 4) {
        MLAS_UNREFERENCED_PARAMETER(HasZeroPoint);
        MLAS_UNREFERENCED_PARAMETER(ComputeType);  // same size regardless of ComputeType
        MLAS_UNREFERENCED_PARAMETER(BackendKernelSelectorConfig);
        const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
        const size_t PackedQuantBDataSize = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
        return PackedQuantBDataSize;
    } else {
        const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
        size_t PackedQuantBDataSize = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);

        if (ComputeType == SQNBIT_CompInt8) {
            const size_t ScaleSize = N * BlockCountK * sizeof(float);
            size_t BlkSumSize = MlasDivRoundup(N, 16) * BlockCountK * 16 * sizeof(float);

            // align on a 32-byte boundary
            constexpr size_t PackedQuantBDataAlignment = 32;
            PackedQuantBDataSize += PackedQuantBDataAlignment - 1;
            constexpr size_t BlkSumAlignment = MlasQNBitQuantBBlkSumAlignment();
            BlkSumSize += BlkSumAlignment - 1;

            if constexpr (QuantAUnsigned) {
                // 2 block sum
                return PackedQuantBDataSize + ScaleSize + BlkSumSize + BlkSumSize;
            } else {
                return PackedQuantBDataSize + ScaleSize + BlkSumSize;
            }
        } else {
            return PackedQuantBDataSize;
        }
    }
}

void
SQ4BitGemmPackQuantBData(
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

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t Iterations = N * BlockCountK;  // one iteration per block

    const size_t SubBlkLen = (ComputeType == SQNBIT_CompInt8)
                                 ? ((BlkLen == 16) ? 16 : 32)
                                 : 16;

    const size_t SubBlkDataSize = SubBlkLen / 2;
    const size_t SubBlkBytePairCount = SubBlkLen / 4;

    //
    // For SubBlkLen == 16, pack 16 4-bit values (8 bytes) at a time like this:
    //
    // src: | v0 v1 | v2 v3 | v4 v5 | v6 v7 | v8 v9 | vA vB | vC vD | vE vF |
    //   =>
    // dst: | v0 v8 | v1 v9 | v2 vA | v3 vB | v4 vC | v5 vD | v6 vE | v7 vF |
    //

    //
    // For SubBlkLen == 32, pack 32 4-bit values (16 bytes) at a time like this:
    //
    // src: | v0  v1  | v2  v3  | ... | v28 v29 | v30 v31 |
    //   =>
    // dst: | v0  v16 | v1  v17 | ... | v14 v30 | v15 v31 |
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

void
SQ4BitGemmPackQuantBDataAndBlkSum(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const std::byte* QuantBDataBegin,
    const float* QuantBScaleBegin,
    bool HasZeroPoint,
    const std::byte* QuantBZPBegin,
    PackedQuantBDataStruct<float, 4>& PackedQuantB,
    MLAS_THREADPOOL* ThreadPool,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
)
{
    MLAS_UNREFERENCED_PARAMETER(QuantBScaleBegin);
    MLAS_UNREFERENCED_PARAMETER(HasZeroPoint);
    MLAS_UNREFERENCED_PARAMETER(QuantBZPBegin);
    assert(BlkLen >= 16 && BlkLen % 16 == 0);

    std::byte* PackedQuantBDataBegin = reinterpret_cast<std::byte*>(PackedQuantB.QuantBWorkspace_);
    SQ4BitGemmPackQuantBData(
        N, K, BlkLen, ComputeType, QuantBDataBegin, PackedQuantBDataBegin, ThreadPool,
        BackendKernelSelectorConfig
    );
}

void
Q8PackQuantB(
  const std::byte* QuantBDataBegin,
  std::byte* PackedQuantBDataBegin,
  float* BlkUnsignedQuantAZeroPointCorrectionBegin,
  MLAS_THREADPOOL* ThreadPool,
  const size_t N,
  const size_t K,
  const size_t BlkLen)
{
    constexpr size_t SubBlkLen = 4;
    const size_t BlkCountK = MlasDivRoundup(K, BlkLen);
    const size_t SubBlkPerBlk = BlkLen / SubBlkLen;
    const size_t StrideN = BlkCountK * BlkLen;
    const size_t Iterations = N * BlkCountK;

    // 4 rows x 8 columns pack together, then 4 rows x 4 columns, then per column.
    MlasTrySimpleParallel(
        ThreadPool, Iterations,
        [&](ptrdiff_t tid) {
            const size_t c = tid / BlkCountK;
            const size_t c8 = c & (~7), c8_res = c & 7;
            const size_t c4 = c & (~3), c4_res = c & 3;
            const size_t r_blk = tid % BlkCountK;
            size_t r_subblk = r_blk * SubBlkPerBlk;

            const std::byte* src = QuantBDataBegin + c * StrideN + r_blk * BlkLen;
            const uint8_t* src8 = reinterpret_cast<const uint8_t*>(src);

            for (size_t i = 0; i < SubBlkPerBlk; ++i, src += SubBlkLen, ++r_subblk) {
                if (c8 + 8 <= N) { // full 8 cols
                    std::byte* dest =
                        PackedQuantBDataBegin + c8 * StrideN + r_subblk * SubBlkLen * 8 + c8_res * SubBlkLen;
                    std::copy(src, src + SubBlkLen, dest);
                } else if (c4 + 4 <= N) { // full 4 cols
                    std::byte* dest =
                        PackedQuantBDataBegin + c4 * StrideN + r_subblk * SubBlkLen * 4 + c4_res * SubBlkLen;
                    std::copy(src, src + SubBlkLen, dest);
                } else { // remainder cols
                    std::byte* dest =
                        PackedQuantBDataBegin + c * StrideN + r_subblk * SubBlkLen;
                    std::copy(src, src + SubBlkLen, dest);
                }
            }

            if (BlkUnsignedQuantAZeroPointCorrectionBegin) {
                const int accu = std::accumulate(src8, src8 + std::min(BlkLen, K - r_blk * BlkLen), 0);

                // for sgemmc
                const size_t dst_offset = ((c / 16) * BlkCountK + r_blk) * 16 + c % 16;
                BlkUnsignedQuantAZeroPointCorrectionBegin[dst_offset] = static_cast<float>(accu);
            }
        }
    );
}

void
Q8ComputePackBlkSum(
  const size_t BlkLen,
  const size_t N,
  const size_t K,
  float* QuantBScaleBegin,
  const std::byte* QuantBZPBegin,
  float* BlockSumBegin,
  float* BlockSum2Begin,
  MLAS_THREADPOOL* ThreadPool)
{
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    std::vector<float> QuantBScaleBeginCopy(N * BlockCountK);
    std::copy(QuantBScaleBegin, QuantBScaleBegin + N * BlockCountK, QuantBScaleBeginCopy.begin());

    MlasTrySimpleParallel(ThreadPool, N * BlockCountK, [&](ptrdiff_t tid) {
        const size_t n = tid / BlockCountK;
        const size_t n8 = n & (~7), n8_res = n & 7;
        const size_t n4 = n & (~3), n4_res = n & 3;
        const size_t k_blk = tid % BlockCountK;

        const size_t src_blk_offset = n * BlockCountK + k_blk;
        const float QuantBScale = QuantBScaleBeginCopy[src_blk_offset];
        uint8_t zp = 128;
        if (QuantBZPBegin) {
            const std::byte* QuantBZP = QuantBZPBegin + src_blk_offset;
            zp = (uint8_t)(*QuantBZP);
        }

        // BlockSum is a width 16 row major matrix
        const size_t dst_offset = ((n / 16) * BlockCountK + k_blk) * 16 + n % 16;
        *(BlockSumBegin + dst_offset) = -QuantBScale * zp;
        if (BlockSum2Begin) {
            BlockSum2Begin[dst_offset] = QuantBScale * (static_cast<float>(zp) * std::min(BlkLen, K - k_blk * BlkLen) - BlockSum2Begin[dst_offset]);
        }

        // re-arrange scale to the same order as packed data
        if (n4 + 4 > N) { // remainder cols
            *(QuantBScaleBegin + n * BlockCountK + k_blk) = QuantBScale;
        } else if (n8 + 8 > N) { // full 4 cols
            *(QuantBScaleBegin + n4 * BlockCountK + k_blk * 4 + n4_res) = QuantBScale;
        } else { // full 8 cols
            *(QuantBScaleBegin + n8 * BlockCountK + k_blk * 8 + n8_res) = QuantBScale;
        }
    });
}

/**
 * 4 rows x 8 cols pack together, along all K. Then 4 rows x 4 cols, along all K.
 * When rows < 4, keep original layout.
 *
 * dotprod: vdotq_laneq_u32.
 * convert quant a from int8 to uint8. zp is 128.
 *
 * i8mm: vusdotq_laneq_s32.
 */
void
SQ8BitGemmPackQuantBDataAndBlkSum(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /* ComputeType */,
    const std::byte* QuantBDataBegin,
    const float* QuantBScaleBegin,
    bool HasZeroPoint,
    const std::byte* QuantBZPBegin,
    PackedQuantBDataStruct<float, 8>& PackedQuantB,
    MLAS_THREADPOOL* ThreadPool,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /* BackendKernelSelectorConfig */
)
{
    assert(BlkLen >= 16 && BlkLen % 16 == 0);

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);

    // Pack the quantized weights
    if (QuantBDataBegin) {
        Q8PackQuantB(QuantBDataBegin, PackedQuantB.PackedQuantBData, PackedQuantB.BlkUnsignedQuantAZeroPointCorrection, ThreadPool, N, K, BlkLen);
    } else {
        // We ignore the scales and zero points if they are provided when pre-packing the weights as there is
        // some "state" associated with 'BlkUnsignedQuantAZeroPointCorrection'.

        // We accumulate the block sum into 'BlkUnsignedQuantAZeroPointCorrection' while packing the weights
        // in the previous step. If we were to use 'scales' while pre-packing the weights and if there were no
        // zero points, then we would enter 'Q8ComputePackBlkSum' twice - once while pre-packing the weights
        // and once while pre-packing the scales which would lead to erroneous 'BlkUnsignedQuantAZeroPointCorrection'
        // computation as the buffer is "used" in-place for the "block sum" temporary values (obtained while pre-packing
        // the weights) and the actual 'BlkUnsignedQuantAZeroPointCorrection' which will use the scales.
        // Hence, to ensure that the piece of logic to calculate 'BlkUnsignedQuantAZeroPointCorrection' is only invoked
        // once, we do it while we are pre-packing the scales and ignore any provided 'scales' and 'zero points' while
        // pre-packing the weights.
        // The flip side is that the user has to ensure that this function is called once each for 'weights',
        // 'scales', and 'zero points'. This is a reasonable expectation and hence we go with that design.

        // Pack the block scales
        if (QuantBScaleBegin) {
            std::copy(QuantBScaleBegin, QuantBScaleBegin + N * BlockCountK, PackedQuantB.PackedQuantBScale);
        }

        // Pack the blksum (and BlkUnsignedQuantAZeroPointCorrection if applicable)
        if ((QuantBScaleBegin && !HasZeroPoint) || QuantBZPBegin) {
            Q8ComputePackBlkSum(BlkLen, N, K, PackedQuantB.PackedQuantBScale, QuantBZPBegin, PackedQuantB.QuantBBlkSum, PackedQuantB.BlkUnsignedQuantAZeroPointCorrection, ThreadPool);
        }
    }
}

//
// Workspace size calculation function implementation.
//

size_t
QNBitGemmPerGemmWorkspaceSize(
    size_t M,
    size_t N,
    size_t K,
    size_t BlkLen,
    bool HasZeroPoint,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    size_t BlkBitWidth,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
)
{
    MLAS_UNREFERENCED_PARAMETER(N);
    MLAS_UNREFERENCED_PARAMETER(HasZeroPoint);
    MLAS_UNREFERENCED_PARAMETER(BlkBitWidth);
    MLAS_UNREFERENCED_PARAMETER(BackendKernelSelectorConfig);

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
QNBitGemmPerGemmWorkspaceAlignment(
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(BlkLen);

    switch (ComputeType) {
        case SQNBIT_CompInt8: {
            return Q8BlkAlignment();
        }
        default: {
            return 1;
        }
    }
}

}  // namespace

template<bool QuantAUnsigned>
size_t
SQ8BitGemmKernel_BlkSum_CompInt8(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* /*QuantBZeroPoint*/,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum,
    const float* BlkUnsignedQuantAZeroPointCorrection
)
{
    MlasQ8Int8GemmKernelNeon<QuantAUnsigned>(
        BlkLen,
        reinterpret_cast<const QuantAType<QuantAUnsigned>*>(QuantA),
        QuantAScale,
        reinterpret_cast<const uint8_t*>(QuantBData),
        QuantBScale,
        C,
        CountM,
        CountN,
        CountK,
        Bias,
        ldc
    );

    {
        float* c_blk = C;
        const float* b_blk_sum = QuantBBlkSum;

        size_t RowsRemaining = CountM;
        const float* a_blksum_row = ABlockSum;
        while (RowsRemaining > 0) {
            auto RowsHandled = MlasSgemmKernelAdd(a_blksum_row, b_blk_sum, c_blk, BlockCountK, RowsRemaining, CountN, BlockCountK, ldc, 1.f);

            c_blk += ldc * RowsHandled;
            a_blksum_row += BlockCountK * RowsHandled;
            RowsRemaining -= RowsHandled;
        }
    }

    if constexpr (QuantAUnsigned) {
        {
            assert(BlkUnsignedQuantAZeroPointCorrection != nullptr);
            float* c_blk = C;
            const float* b_blk_sum2 = BlkUnsignedQuantAZeroPointCorrection;

            size_t RowsRemaining = CountM;
            const float* a_scale_row = QuantAScale;
            while (RowsRemaining > 0) {
                auto RowsHandled = MlasSgemmKernelAdd(a_scale_row, b_blk_sum2, c_blk, BlockCountK, RowsRemaining, CountN, BlockCountK, ldc, 128.f);

                c_blk += ldc * RowsHandled;
                a_scale_row += BlockCountK * RowsHandled;
                RowsRemaining -= RowsHandled;
            }
        }
    }

    return CountM;
}

}  // namespace sqnbitgemm_neon

//
// Kernel dispatch structure accessor.
//

const MLAS_QNBIT_GEMM_DISPATCH&
GetMlasQNBitGemmDispatchNeon(
    bool InitializeWithDotSupport,
    bool InitializeWithI8MMSupport
)
{
    // Note: The InitializeWithX parameters are only used in the invocation of this method that initializes the static
    // MLAS_QNBIT_GEMM_DISPATCH instance.

    static const MLAS_QNBIT_GEMM_DISPATCH MlasQNBitGemmDispatchNeon = [&]() {
        MLAS_QNBIT_GEMM_DISPATCH d;

        d.Q4BitGemmPackQuantBDataSize = sqnbitgemm_neon::QNBitGemmPackQuantBDataSize<4, false>;
        d.Q8BitGemmPackQuantBDataSize = sqnbitgemm_neon::QNBitGemmPackQuantBDataSize<8, true>;
        d.SQ4BitGemmPackQuantBData = sqnbitgemm_neon::SQ4BitGemmPackQuantBData;
        d.SQ4BitGemmPackQuantBDataAndBlkSum = sqnbitgemm_neon::SQ4BitGemmPackQuantBDataAndBlkSum;
        d.SQ8BitGemmPackQuantBDataAndBlkSum = sqnbitgemm_neon::SQ8BitGemmPackQuantBDataAndBlkSum;

        d.QNBitGemmPerGemmWorkspaceSize = sqnbitgemm_neon::QNBitGemmPerGemmWorkspaceSize;
        d.QNBitGemmPerGemmWorkspaceAlignment = sqnbitgemm_neon::QNBitGemmPerGemmWorkspaceAlignment;

        d.SQ4BitGemmM1Kernel_CompFp32 = sqnbitgemm_neon::SQ4BitGemmM1Kernel_CompFp32;
        d.SQ4BitBlkDequantBForSgemm_CompFp32 = sqnbitgemm_neon::SQ4BitBlkDequantBForSgemm_CompFp32;

        if (InitializeWithDotSupport) {
            d.SQ4BitGemmKernel_CompInt8 = sqnbitgemm_neon::SQ4BitGemmKernel_CompInt8;
            d.QuantizeARow_CompInt8 = sqnbitgemm_neon::QuantizeARow_CompInt8;

            d.QuantizeARowComputeBlkSum_CompInt8 = sqnbitgemm_neon::QuantizeARowComputeBlkSum_CompInt8<true>;
            d.SQ8BitGemmKernel_BlkSum_CompInt8 = sqnbitgemm_neon::SQ8BitGemmKernel_BlkSum_CompInt8<true>;
        }

        if (InitializeWithI8MMSupport) {
            d.Q8BitGemmPackQuantBDataSize = sqnbitgemm_neon::QNBitGemmPackQuantBDataSize<8, false>;
            d.QuantizeARowComputeBlkSum_CompInt8 = sqnbitgemm_neon::QuantizeARowComputeBlkSum_CompInt8<false>;
            d.SQ8BitGemmKernel_BlkSum_CompInt8 = sqnbitgemm_neon::SQ8BitGemmKernel_BlkSum_CompInt8<false>;
        }

        // W2 native CompInt8 path.
        //
        // Pack-size, pack-and-blksum, and EffectiveBlockCountK use the
        // portable AVX-512-namespaced helpers (the file is misleadingly
        // named -- the TU contains no x86 intrinsics) which serve as the
        // cross-arch layout authority for W2.
        //
        // SQ2BitGemmKernel_BlkSum_CompInt8 is wired to the native NEON
        // DotProd kernel when FEAT_DotProd is available; the kernel
        // handles BlkLen ∈ {32, 64, 128} natively via SDOT inner loops.
        // The 2-bit weights are unpacked to unsigned values in [0, 3] and
        // fed directly to SDOT; the B zero-point correction is fused into
        // the accumulator via the ABlockSum × QuantBBlkSum term (where
        // QuantBBlkSum bakes in -scale × zp per block), so no post-kernel
        // zero-point correction SGEMM is needed.
        //
        // FEAT_I8MM hosts also take this path: FEAT_I8MM always implies
        // FEAT_DotProd per the ARM spec, and USDOT and SDOT have identical
        // throughput on every core that implements both -- a separate I8MM
        // TU using USDOT would be pure duplication. The real I8MM-only
        // throughput win lives in SMMLA (R2-tile 2×2 matrix-multiply,
        // 2× the dots/cycle of SDOT), which is future work.
        //
        // The Scalar fall-back is defensive: an I8MM-without-DotProd host
        // is unreachable on conformant ARMv8.2+ hardware.
        if (InitializeWithDotSupport || InitializeWithI8MMSupport) {
            d.Q2BitGemmPackQuantBDataSize       = onnxruntime::mlas::sq2bit_avx512::Q2BitGemmPackQuantBDataSize_Avx512;
            d.SQ2BitGemmPackQuantBDataAndBlkSum = onnxruntime::mlas::sq2bit_avx512::SQ2BitGemmPackQuantBDataAndBlkSum_Scalar;
            d.SQ2BitGemmKernel_BlkSum_CompInt8  = InitializeWithDotSupport
                ? sqnbitgemm_neon::SQ2BitGemmKernel_BlkSum_CompInt8_NeonDotProd
                : onnxruntime::mlas::sq2bit_avx512::SQ2BitGemmKernel_BlkSum_CompInt8_Scalar;
            d.Q2BitGemmEffectiveBlockCountK     = [](size_t BlockCountK) {
                return MlasDivRoundup(BlockCountK, kSq2BitAvx512WeightKBlockGroup) * kSq2BitAvx512WeightKBlockGroup;
            };
            // W2 NEON DotProd kernel uses vdotq_s32 over A and so requires
            // SIGNED int8 A. The shared QuantizeARowComputeBlkSum is wired
            // to the UNSIGNED W8 variant on DotProd-only hosts; route W2
            // through the signed variant explicitly.
            d.QuantizeARowComputeBlkSum_CompInt8_W2 = sqnbitgemm_neon::QuantizeARowComputeBlkSum_CompInt8<false>;
        }

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
        d.HQ4BitGemmPackQuantBData = sqnbitgemm_neon::HQ4BitGemmPackQuantBData_CompFp16;
        d.HQ4BitBlkDequantBForHgemm_CompFp16 = sqnbitgemm_neon::HQ4BitBlkDequantBForHgemm_CompFp16;
        d.HQ4BitGemmKernel_CompFp16 = sqnbitgemm_neon::HQ4BitGemmKernel_CompFp16;
        d.HQ8BitGemmPackQuantBData = sqnbitgemm_neon::HQ8BitGemmPackQuantBData_CompFp16;
        d.HQ8BitBlkDequantBForHgemm_CompFp16 = sqnbitgemm_neon::HQ8BitBlkDequantBForHgemm_CompFp16;
#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED && MLAS_TARGET_ARM64

        return d;
    }();

    return MlasQNBitGemmDispatchNeon;
}
