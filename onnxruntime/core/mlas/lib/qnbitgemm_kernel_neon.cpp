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
#include <numeric>

#include "qnbitgemm.h"
#include "sqnbitgemm_q8_block.h"

#ifdef USE_KLEIDIAI
#include "kai/kai_common.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai_ukernel_interface.h"
#endif

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
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    if constexpr (BlkBitWidth == 4) {
#ifndef USE_KLEIDIAI
        MLAS_UNREFERENCED_PARAMETER(HasZeroPoint);
        MLAS_UNREFERENCED_PARAMETER(ComputeType);  // same size regardless of ComputeType
#endif

#ifdef USE_KLEIDIAI
        if (ComputeType == SQNBIT_CompInt8 && UseKleidiAI(K, BlkLen, HasZeroPoint)) {
            const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel& ukernel = GetKleidiAIGemmUKernel();
            const size_t nr = ukernel.get_nr();
            const size_t kr = ukernel.get_kr();
            const size_t sr = ukernel.get_sr();
            return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(N, K, nr, kr, sr, BlkLen, kai_dt_bf16);
        } else
#endif
        {
            const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
            const size_t PackedQuantBDataSize = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
            return PackedQuantBDataSize;
        }
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
    PackedQuantBDataStruct<float, 4>& PackedQuantB,
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
        const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel& ukernel = GetKleidiAIGemmUKernel();
        std::byte* PackedQuantBDataBegin = PackedQuantB.PackedQuantBData;

        const size_t nr = ukernel.get_nr();
        const size_t kr = ukernel.get_kr();
        const size_t sr = ukernel.get_sr();

        kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params;
        params.lhs_zero_point = 1;
        params.rhs_zero_point = 8;
        params.scale_dt = kai_dt_bf16;

        const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
        const size_t scales_len = N * BlockCountK;
        std::vector<uint16_t> scales(scales_len);
        for (size_t i = 0; i < scales_len; i++) {
            const uint32_t* i32 = reinterpret_cast<const uint32_t*>(&QuantBScaleBegin[i]);
            scales[i] = *i32 >> 16;
        }

        kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(1, N, K, nr, kr, sr, BlkLen,
                reinterpret_cast<const uint8_t*>(QuantBDataBegin), BlockCountK * BlkLen / 2,
                nullptr, scales.data(), BlockCountK * sizeof(uint16_t),
                PackedQuantBDataBegin, 0, &params);
    } else
#endif
    {
        std::byte* PackedQuantBDataBegin = reinterpret_cast<std::byte*>(PackedQuantB.QuantBWorkspace_);
        SQ4BitGemmPackQuantBData(N, K, BlkLen, ComputeType, QuantBDataBegin, PackedQuantBDataBegin, ThreadPool);
    }
}

void
Q8PackQuantB(
  const std::byte* QuantBDataBegin,
  std::byte* PackedQuantBDataBegin,
  float* BlockSum2Begin,
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

            if (BlockSum2Begin) {
                const int accu = std::accumulate(src8, src8 + std::min(BlkLen, K - r_blk * BlkLen), 0);

                // for sgemmc
                const size_t blksum2_dst_offset = ((c / 16) * BlkCountK + r_blk) * 16 + c % 16;
                BlockSum2Begin[blksum2_dst_offset] = static_cast<float>(accu);
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
 * When rol < 4, keep original layout.
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
    MLAS_THREADPOOL* ThreadPool
)
{
    assert(BlkLen >= 16 && BlkLen % 16 == 0);

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);

    // Pack the quantized weights
    if (QuantBDataBegin) {
        Q8PackQuantB(QuantBDataBegin, PackedQuantB.PackedQuantBData, PackedQuantB.QuantBBlkSum2, ThreadPool, N, K, BlkLen);
    }

    // Pack the block scales
    if (QuantBScaleBegin) {
        std::copy(QuantBScaleBegin, QuantBScaleBegin + N * BlockCountK, PackedQuantB.PackedQuantBScale);
    }

    // Pack the blksum (and blksum2 if applicable)
    if ((QuantBScaleBegin && !HasZeroPoint) || QuantBZPBegin) {
        Q8ComputePackBlkSum(BlkLen, N, K, PackedQuantB.PackedQuantBScale, QuantBZPBegin, PackedQuantB.QuantBBlkSum, PackedQuantB.QuantBBlkSum2, ThreadPool);
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
    size_t BlkBitWidth
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
            if (BlkBitWidth == 4 && UseKleidiAI(K, BlkLen, HasZeroPoint)) {
                const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel& ukernel =
                    M == 1? GetKleidiAIGemvUKernel() : GetKleidiAIGemmUKernel();

                const size_t mr = ukernel.get_mr();
                const size_t kr = ukernel.get_kr();
                const size_t sr = ukernel.get_sr();
                return kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
            } else
#endif
            {
                // workspace buffer is used for block quantization of A to int8
                const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
                // QuantData + Scale + BlkSum
                const size_t PerGemmWorkspaceSize = M * BlockCountK * (Q8BlkSize(BlkLen) + sizeof(float));
                return PerGemmWorkspaceSize;
            }
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
    const float* QuantBBlkSum2
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
            assert(QuantBBlkSum2 != nullptr);
            float* c_blk = C;
            const float* b_blk_sum2 = QuantBBlkSum2;

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
            d.UsePacked_CompInt8 = sqnbitgemm_neon::UsePacked_CompInt8;

            d.QuantizeARowComputeBlkSum_CompInt8 = sqnbitgemm_neon::QuantizeARowComputeBlkSum_CompInt8<true>;
            d.SQ8BitGemmKernel_BlkSum_CompInt8 = sqnbitgemm_neon::SQ8BitGemmKernel_BlkSum_CompInt8<true>;

#ifdef USE_KLEIDIAI
            d.SQ4BitGemmKernel_Packed_CompInt8 = sqnbitgemm_neon::SQ4BitGemmKernel_Packed_CompInt8;
            d.QuantizeA_Packed_CompInt8 = sqnbitgemm_neon::QuantizeA_Packed_CompInt8;
#endif
        }

        if (InitializeWithI8MMSupport) {
            d.Q8BitGemmPackQuantBDataSize = sqnbitgemm_neon::QNBitGemmPackQuantBDataSize<8, false>;
            d.QuantizeARowComputeBlkSum_CompInt8 = sqnbitgemm_neon::QuantizeARowComputeBlkSum_CompInt8<false>;
            d.SQ8BitGemmKernel_BlkSum_CompInt8 = sqnbitgemm_neon::SQ8BitGemmKernel_BlkSum_CompInt8<false>;
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
