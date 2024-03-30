/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_neon.h

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for ARM NEON.

--*/

#include <algorithm>
#include <cassert>
#include <utility>

#include "sqnbitgemm.h"

//
// Quantized B data packing function implementation.
//

namespace
{

size_t
SQ4BitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(ComputeType);  // same size regardless of ComputeType

    constexpr size_t BlkBitWidth = 4;

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t PackedQuantBDataSize = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    return PackedQuantBDataSize;
}

void
SQ4BitGemmPackQuantBData(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
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

    const size_t SubBlkLen = (ComputeType == CompInt8)
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

}  // namespace

//
// General helpers.
//

namespace
{

template <typename IterationFn, size_t... Indices>
MLAS_FORCEINLINE void
UnrolledLoopIterations(IterationFn&& f, std::index_sequence<Indices...> /* indices */)
{
    (f(Indices), ...);
}

template <size_t N, typename IterationFn>
MLAS_FORCEINLINE void
UnrolledLoop(IterationFn&& f)
{
    UnrolledLoopIterations(std::forward<IterationFn>(f), std::make_index_sequence<N>());
}
}  // namespace

//
// CompFp32 kernel implementation.
//

namespace
{

template <size_t NCols, bool HasZeroPoint>
MLAS_FORCEINLINE void
ComputeDotProducts_BlkBitWidth4_CompFp32(
    size_t BlkLen,
    const float* ARowPtr,
    const std::byte* QuantBDataColPtr,
    const float* QuantBScaleColPtr,
    const std::byte* QuantBZeroPointColPtr,
    float* SumPtr,
    size_t CountK,
    size_t StrideQuantBData,
    size_t StrideQuantBScale,
    size_t StrideQuantBZeroPoint,
    const float* BiasPtr
)
{
    BlkLen, ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK;
    StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint, BiasPtr;
}

template <bool HasZeroPoint>
void
SQ4BitGemmM1Kernel_CompFp32_Impl(
    size_t BlkLen,
    const float* A,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t BlockStrideQuantB,
    const float* Bias
)
{
    constexpr size_t BlkBitWidth = 4;
    constexpr size_t NCols = 4;

    const float* ARowPtr = A;
    float* CRowPtr = C;

    const size_t BlockCountK = BlockStrideQuantB;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    const float* BiasPtr = Bias;

    const std::byte* QuantBDataColPtr = QuantBData;
    const float* QuantBScaleColPtr = QuantBScale;
    const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

    float* SumPtr = CRowPtr;

    int64_t nblk = static_cast<int64_t>(CountN) - NCols;

    while (nblk >= 0) {
        ComputeDotProducts_BlkBitWidth4_CompFp32<NCols, HasZeroPoint>(
            BlkLen,
            ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
            StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
            BiasPtr
        );

        // move to next `NCols` columns

        QuantBDataColPtr += NCols * StrideQuantBData;
        QuantBScaleColPtr += NCols * StrideQuantBScale;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointColPtr += NCols * StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? NCols : 0;
        SumPtr += NCols;

        nblk -= NCols;
    }

    // left over columns less than `NCols`?
    nblk += NCols;
    for (int64_t n = 0; n < nblk; ++n) {
        ComputeDotProducts_BlkBitWidth4_CompFp32<1, HasZeroPoint>(
            BlkLen,
            ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
            StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
            BiasPtr
        );

        // move to next column

        QuantBDataColPtr += StrideQuantBData;
        QuantBScaleColPtr += StrideQuantBScale;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointColPtr += StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? 1 : 0;
        SumPtr += 1;
    }
}

MLAS_FORCEINLINE void
SQ4BitGemmM1Kernel_CompFp32(
    size_t BlkLen,
    const float* A,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t BlockStrideQuantB,
    const float* Bias
)
{
    if (QuantBZeroPoint != nullptr) {
        SQ4BitGemmM1Kernel_CompFp32_Impl<true>(
            BlkLen,
            A,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountN,
            CountK,
            BlockStrideQuantB,
            Bias
        );
    } else {
        SQ4BitGemmM1Kernel_CompFp32_Impl<false>(
            BlkLen,
            A,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountN,
            CountK,
            BlockStrideQuantB,
            Bias
        );
    }
}

MLAS_FORCEINLINE void
Q4BitBlkDequantBForSgemm_CompFp32(
    size_t BlkLen,
    float* FpData,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockStrideQuantB
)
{
    auto impl0_reference = [&]() {
        constexpr size_t BlkBitWidth = 4;
        constexpr size_t SubBlkLen = 16;

        float* Dst = FpData;

        const std::byte* QuantBDataCol = QuantBData;
        const float* QuantBScaleCol = QuantBScale;
        const std::byte* QuantBZeroPointCol = QuantBZeroPoint;

        for (size_t n = 0; n < CountN; n += 16) {
            const size_t nnlen = std::min(CountN - n, size_t{16});

            for (size_t nn = 0; nn < nnlen; ++nn) {
                for (size_t k = 0, k_blk_idx = 0; k < CountK; k += BlkLen, k_blk_idx += 1) {
                    const size_t kklen = std::min(CountK - k, BlkLen);

                    const std::byte* b_data =
                        QuantBDataCol + k_blk_idx * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
                    const float b_s = QuantBScaleCol[k_blk_idx];
                    const uint8_t b_z =
                        (QuantBZeroPointCol != nullptr)
                            ? ((k_blk_idx & 1) == 1)
                                  ? std::to_integer<uint8_t>(QuantBZeroPointCol[k_blk_idx / 2] >> 4)
                                  : std::to_integer<uint8_t>(QuantBZeroPointCol[k_blk_idx / 2] & std::byte{0x0F})
                            : 8;

                    for (size_t kk = 0; kk < kklen; ++kk) {
                        const size_t packed_idx = kk % SubBlkLen;

                        const bool is_low_half = packed_idx < (SubBlkLen / 2);
                        const size_t packed_byte_idx = packed_idx % (SubBlkLen / 2);
                        const size_t packed_range_offset = (kk / SubBlkLen) * (SubBlkLen / 2);

                        const std::byte b_packed = b_data[packed_range_offset + packed_byte_idx];
                        const std::byte b_byte = is_low_half ? (b_packed & std::byte{0x0F}) : (b_packed >> 4);
                        const float b_value = (std::to_integer<int8_t>(b_byte) - b_z) * b_s;

                        Dst[(k + kk) * 16 + nn] = b_value;
                    }
                }

                QuantBDataCol += BlockStrideQuantB * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
                QuantBScaleCol += BlockStrideQuantB;
                if (QuantBZeroPointCol != nullptr) {
                    QuantBZeroPointCol += MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockStrideQuantB);
                }
            }

            // zero out any remaining columns

            if (nnlen < 16) {
                for (size_t k = 0; k < CountK; ++k) {
                    std::fill_n(Dst + (k * 16) + nnlen, 16 - nnlen, 0.0f);
                }
            }

            Dst += CountK * 16;
        }
    };

    impl0_reference();
}

//
// CompInt8 kernel implementation.
//

template <size_t SubBlkLen>
MLAS_FORCEINLINE void
QuantizeBlock(
    size_t BlkLen,
    const float* A,
    size_t ElementCount,
    std::byte* QuantA
)
{
    BlkLen;
    A;
    ElementCount;
    QuantA;
}

void MLASCALL
QuantizeARow_CompInt8(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA
)
{
    const float* ADataBlkPtr = A;
    std::byte* QuantABlkPtr = QuantA;

    for (size_t k = 0; k < CountK; k += BlkLen) {
        const size_t k_blk_len = std::min(CountK - k, BlkLen);

        QuantizeBlock<16>(BlkLen, ADataBlkPtr, k_blk_len, QuantABlkPtr);

        ADataBlkPtr += BlkLen;
        QuantABlkPtr += Q8BlkSize(BlkLen);
    }
}

template <bool HasZeroPoint>
void
SQ4BitGemmM1Kernel_CompInt8_Impl_BlkLen16(
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias
)
{
    QuantA, QuantBData, QuantBScale, QuantBZeroPoint, C, CountN, BlockCountK, Bias;
}

template <bool HasZeroPoint>
void
SQ4BitGemmM1Kernel_CompInt8_Impl_BlkLen32(
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias
)
{
    QuantA, QuantBData, QuantBScale, QuantBZeroPoint, C, CountN, BlockCountK, Bias;
}

template <bool HasZeroPoint>
void
SQ4BitGemmM1Kernel_CompInt8_Impl_BlkLenGreaterThan32(
    size_t BlkLen,
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias
)
{
    BlkLen, QuantA, QuantBData, QuantBScale, QuantBZeroPoint, C, CountN, BlockCountK, Bias;
}

template <bool HasZeroPoint>
MLAS_FORCEINLINE void
SQ4BitGemmM1Kernel_CompInt8_DispatchOnBlkLen(
    size_t BlkLen,
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t BlockStrideQuantB,
    const float* Bias
)
{
    if (BlkLen == 16) {
        SQ4BitGemmM1Kernel_CompInt8_Impl_BlkLen16<HasZeroPoint>(
            QuantA,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountN,
            BlockStrideQuantB,
            Bias
        );
    } else if (BlkLen == 32) {
        SQ4BitGemmM1Kernel_CompInt8_Impl_BlkLen32<HasZeroPoint>(
            QuantA,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountN,
            BlockStrideQuantB,
            Bias
        );
    } else {
        SQ4BitGemmM1Kernel_CompInt8_Impl_BlkLenGreaterThan32<HasZeroPoint>(
            BlkLen,
            QuantA,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountN,
            BlockStrideQuantB,
            Bias
        );
    }
}

MLAS_FORCEINLINE
void
SQ4BitGemmM1Kernel_CompInt8(
    size_t BlkLen,
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t /*CountK*/,
    size_t BlockStrideQuantB,
    const float* Bias
)
{
    if (QuantBZeroPoint != nullptr) {
        SQ4BitGemmM1Kernel_CompInt8_DispatchOnBlkLen<true>(
            BlkLen,
            QuantA,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountN,
            BlockStrideQuantB,
            Bias
        );
    } else {
        SQ4BitGemmM1Kernel_CompInt8_DispatchOnBlkLen<false>(
            BlkLen,
            QuantA,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountN,
            BlockStrideQuantB,
            Bias
        );
    }
}

}  // namespace

//
// Kernel dispatch structure definition.
//

const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchAvx2 = []() {
    MLAS_SQNBIT_GEMM_DISPATCH d;

    d.SQ4BitGemmPackQuantBDataSize = SQ4BitGemmPackQuantBDataSize;
    d.SQ4BitGemmPackQuantBData = SQ4BitGemmPackQuantBData;

    d.SQ4BitGemmM1Kernel_CompFp32 = SQ4BitGemmM1Kernel_CompFp32;
    d.Q4BitBlkDequantBForSgemm_CompFp32 = Q4BitBlkDequantBForSgemm_CompFp32; // MlasQ4GemmKernel

    d.SQ4BitGemmM1Kernel_CompInt8 = SQ4BitGemmM1Kernel_CompInt8;
    d.QuantizeARow_CompInt8 = QuantizeARow_CompInt8;

    return d;
}();
