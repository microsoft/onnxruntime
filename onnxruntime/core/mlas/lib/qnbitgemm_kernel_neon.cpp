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
#include <vector>

#include "qnbitgemm.h"
#include "sqnbitgemm_q8_block.h"

#ifdef USE_KLEIDIAI
#include "kai/kai_common.h"
#include "kai_ukernel_interface.h"
#endif

namespace sqnbitgemm_neon
{

namespace
{

//
// Quantized B data packing function implementation.
//

size_t
Q4BitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    bool HasZeroPoint,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
#ifndef USE_KLEIDIAI
    MLAS_UNREFERENCED_PARAMETER(HasZeroPoint);
    MLAS_UNREFERENCED_PARAMETER(ComputeType);  // same size regardless of ComputeType
#endif

#ifdef USE_KLEIDIAI
    if (ComputeType == SQNBIT_CompInt8 && UseKleidiAI(K, BlkLen, HasZeroPoint)) {
        return GetKleidiAIGemmStrategy().GetRHSPackedSize(N, K, BlkLen);
    } else
#endif
    {
        constexpr size_t BlkBitWidth = 4;

        const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
        const size_t PackedQuantBDataSize = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
        return PackedQuantBDataSize;
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
    MLAS_THREADPOOL* ThreadPool
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
    const std::byte*,
    PackedQuantBDataStruct<float>& PackedQuantB,
    MLAS_THREADPOOL* ThreadPool
)
{
#ifndef USE_KLEIDIAI
    MLAS_UNREFERENCED_PARAMETER(QuantBScaleBegin);
    MLAS_UNREFERENCED_PARAMETER(HasZeroPoint);
#endif
    assert(BlkLen >= 16 && BlkLen % 16 == 0);

#ifdef USE_KLEIDIAI
    if (UseKleidiAI(K, BlkLen, HasZeroPoint)) {
        GetKleidiAIGemmStrategy().RunRHSPack(N, K, BlkLen, QuantBDataBegin,
            QuantBScaleBegin, PackedQuantB.PackedQuantBData);
    } else
#endif
    {
        std::byte* PackedQuantBDataBegin = reinterpret_cast<std::byte*>(PackedQuantB.QuantBWorkspace_);
        SQ4BitGemmPackQuantBData(N, K, BlkLen, ComputeType, QuantBDataBegin, PackedQuantBDataBegin, ThreadPool);
    }
}

//
// Workspace size calculation function implementation.
//

size_t
Q4BitGemmPerGemmWorkspaceSize(
    size_t M,
    size_t N,
    size_t K,
    size_t BlkLen,
    bool HasZeroPoint,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(N);
#ifndef USE_KLEIDIAI
    MLAS_UNREFERENCED_PARAMETER(HasZeroPoint);
#endif

    switch (ComputeType) {
        case SQNBIT_CompInt8: {
            // workspace buffer is used for block quantization of A to int8
#ifdef USE_KLEIDIAI
            if (UseKleidiAI(K, BlkLen, HasZeroPoint)) {
                const KleidiAIQ4BitGemmStrategy& strategy =
                    M == 1? GetKleidiAIGemvStrategy() : GetKleidiAIGemmStrategy();

                return strategy.GetLHSPackedSize(M, K);
            } else
#endif
            {
                const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
                const size_t PerGemmWorkspaceSize = M * BlockCountK * Q8BlkSize(BlkLen);
                return PerGemmWorkspaceSize;
            }
        }
        default: {
            return 0;
        }
    }
}

size_t
Q4BitGemmPerGemmWorkspaceAlignment(
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

bool
UseKleidiAI(size_t K, size_t BlkLen, bool HasZp)
{
#ifdef USE_KLEIDIAI
    bool has_dotprod = MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeonDot();
    return (BlkLen % 32) == 0 && (K % BlkLen) == 0 && !HasZp && has_dotprod;
#else
    MLAS_UNREFERENCED_PARAMETER(K);
    MLAS_UNREFERENCED_PARAMETER(BlkLen);
    MLAS_UNREFERENCED_PARAMETER(HasZp);
    return false;
#endif
}

}  // namespace sqnbitgemm_neon

//
// Kernel dispatch structure accessor.
//

const MLAS_QNBIT_GEMM_DISPATCH&
GetMlasQNBitGemmDispatchNeon(
    bool InitializeWithDotSupport
)
{
    // Note: The InitializeWithX parameters are only used in the invocation of this method that initializes the static
    // MLAS_QNBIT_GEMM_DISPATCH instance.

    static const MLAS_QNBIT_GEMM_DISPATCH MlasQNBitGemmDispatchNeon = [&]() {
        MLAS_QNBIT_GEMM_DISPATCH d;

        d.Q4BitGemmPackQuantBDataSize = sqnbitgemm_neon::Q4BitGemmPackQuantBDataSize;
        d.SQ4BitGemmPackQuantBData = sqnbitgemm_neon::SQ4BitGemmPackQuantBData;
        d.SQ4BitGemmPackQuantBDataAndBlkSum = sqnbitgemm_neon::SQ4BitGemmPackQuantBDataAndBlkSum;

        d.Q4BitGemmPerGemmWorkspaceSize = sqnbitgemm_neon::Q4BitGemmPerGemmWorkspaceSize;
        d.Q4BitGemmPerGemmWorkspaceAlignment = sqnbitgemm_neon::Q4BitGemmPerGemmWorkspaceAlignment;

        d.SQ4BitGemmM1Kernel_CompFp32 = sqnbitgemm_neon::SQ4BitGemmM1Kernel_CompFp32;
        d.SQ4BitBlkDequantBForSgemm_CompFp32 = sqnbitgemm_neon::SQ4BitBlkDequantBForSgemm_CompFp32;

        if (InitializeWithDotSupport) {
            d.SQ4BitGemmKernel_CompInt8 = sqnbitgemm_neon::SQ4BitGemmKernel_CompInt8;
            d.QuantizeARow_CompInt8 = sqnbitgemm_neon::QuantizeARow_CompInt8;
            d.UsePacked_CompInt8 = sqnbitgemm_neon::UsePacked_CompInt8;

#ifdef USE_KLEIDIAI
            d.SQ4BitGemmKernel_Packed_CompInt8 = sqnbitgemm_neon::SQ4BitGemmKernel_Packed_CompInt8;
            d.QuantizeA_Packed_CompInt8 = sqnbitgemm_neon::QuantizeA_Packed_CompInt8;
#endif
        }

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
        d.HQ4BitGemmPackQuantBData = sqnbitgemm_neon::HQ4BitGemmPackQuantBData_CompFp16;
        d.HQ4BitBlkDequantBForHgemm_CompFp16 = sqnbitgemm_neon::HQ4BitBlkDequantBForHgemm_CompFp16;
        d.HQ4BitGemmKernel_CompFp16 = sqnbitgemm_neon::HQ4BitGemmKernel_CompFp16;
#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED && MLAS_TARGET_ARM64

        return d;
    }();

    return MlasQNBitGemmDispatchNeon;
}
