/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm.h

Abstract:

    This module includes:

    - Declaration of the set of template functions used to implement a kernel
    for a matrix/matrix multiplication, A*B, where A is a float matrix and B is
    a n-bit quantized integer matrix (QNBitGemm).

    - A shared kernel driver function template, MlasSQNBitGemmOperation.

    - Kernel dispatch structure.

    The B matrix is block quantized, which means that its values are grouped
    into blocks which each have one scale and optional zero point. Each
    quantized value in B is n-bits wide.

--*/

#pragma once

#include "mlas_qnbit.h"
#include "mlasi.h"

//
// Kernel implementation template declarations
//

/// <summary>
/// Multiply float matrix A with quantized n-bit integer matrix B.
/// </summary>
/// <typeparam name="KernelType">Hardware-specific kernel type.</typeparam>
/// <typeparam name="BlkLen">Number of values in a block.</typeparam>
/// <typeparam name="BlkBitWidth">Bit width of each value in a block.</typeparam>
/// <param name="A">Supplies the A matrix.</param>
/// <param name="PackedBData">Supplies the packed B matrix block data.</param>
/// <param name="PackedBScale">Supplies the packed B matrix block scale values.</param>
/// <param name="PackedBZeroPoint">Supplies the packed B matrix block zero point values. Optional.</param>
/// <param name="C">Supplies the output C matrix.</param>
/// <param name="CountM">Number of rows of A and C.</param>
/// <param name="CountN">Number of columns of B and C.</param>
/// <param name="CountK">Number of columns of A and rows of B.</param>
/// <param name="lda">Leading dimension of A.</param>
/// <param name="BlockStridePackedB">
/// Number of blocks between adjacent columns of B (packed B values are transposed).
/// </param>
/// <param name="ldc">Leading dimension of C.</param>
/// <param name="Bias">Bias vector of length N. Optional.</param>
/// <returns>Number of rows of A handled.</returns>
template <size_t BlkBitWidth, size_t BlkLen, typename KernelType>
MLAS_FORCEINLINE size_t
MlasSQNBitGemmKernel(
    const float* A,
    const uint8_t* PackedBData,
    const float* PackedBScale,
    const uint8_t* PackedBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t BlockStridePackedB,
    size_t ldc,
    const float* Bias
);

// dequantize B into the format expected by MlasSgemmKernelZero
template <size_t BlkBitWidth, size_t BlkLen, typename KernelType>
MLAS_FORCEINLINE void
MlasQNBitBlkDequantBForSgemm(
    float* FpData,
    const uint8_t* PackedBData,
    const float* PackedBScale,
    const uint8_t* PackedBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockStridePackedB
);

//
// MlasQNBitGemmOperation and helpers
//

constexpr MLAS_FORCEINLINE size_t
MlasQNBitBlkDataSizeInBytes(size_t BlkBitWidth, size_t BlkLen)
{
    return BlkLen * BlkBitWidth / 8;
}

template<size_t BlkBitWidth>
constexpr MLAS_FORCEINLINE size_t
MlasQNBitZeroPointsForBlksSizeInBytes(size_t BlkCount)
{
    if constexpr (BlkBitWidth <= 4) {
        return MlasDivRoundup(BlkCount, 2);  // 2 blocks per byte
    } else {
        return BlkCount;
    }
}

MLAS_FORCEINLINE void
MlasAddBiasForGemm(const float* Bias, float* C, size_t CountM, size_t CountN, size_t ldc)
{
    for (size_t m = 0; m < CountM; m++) {
        const float* bias = Bias;
        float* sum = C;
        for (size_t n = 0; n < CountN; n += 4) {
            if (CountN - n < 4) {
                for (size_t nn = n; nn < CountN; nn++) {
                    *sum += *bias;
                    sum++;
                    bias++;
                }
                break;
            }

            MLAS_FLOAT32X4 acc_x = MlasLoadFloat32x4(sum);
            acc_x = MlasAddFloat32x4(acc_x, MlasLoadFloat32x4(bias));
            MlasStoreFloat32x4(sum, acc_x);
            bias += 4;
            sum += 4;
        }
        C += ldc;
    }
}

template <size_t BlkBitWidth, size_t BlkLen, typename KernelType>
MLAS_FORCEINLINE void MLASCALL
MlasSQNBitGemmOperation(
    const size_t K,
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* const DataParams,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
)
{
    const size_t lda = DataParams->lda;
    const size_t ldc = DataParams->ldc;

    const size_t k_blks = MlasDivRoundup(K, BlkLen);
    const size_t ldb = k_blks * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const float* A = DataParams->A + RangeStartM * lda;
    const uint8_t* PackedBData = static_cast<const uint8_t*>(DataParams->PackedBData);
    const float* PackedBScale = DataParams->PackedBScale;
    const uint8_t* PackedBZeroPoint = static_cast<const uint8_t*>(DataParams->PackedBZeroPoint);
    float* C = DataParams->C + RangeStartM * ldc + RangeStartN;
    const float* Bias = DataParams->Bias;

    if (RangeCountM == 1) {
        size_t CountN;
        for (size_t n = 0; n < RangeCountN; n += CountN) {
            CountN = std::min(RangeCountN - n, (size_t)128);

            //
            // Step through each slice of matrix A along the M dimension.
            //
            const float* bias = (Bias == nullptr) ? nullptr : Bias + RangeStartN + n;
            const uint8_t* b_col = PackedBData + (RangeStartN + n) * ldb;
            const float* b_col_scale = PackedBScale + (RangeStartN + n) * k_blks;
            const uint8_t* b_col_zp =
                (PackedBZeroPoint == nullptr)
                    ? nullptr
                    : PackedBZeroPoint + (RangeStartN + n) * MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(k_blks);
            float* c_blk = C + n;
            const float* a_row = A;

            size_t RowsRemaining = RangeCountM;
            while (RowsRemaining > 0) {
                auto RowsHandled = MlasSQNBitGemmKernel<BlkBitWidth, BlkLen, KernelType>(
                    a_row, b_col, b_col_scale, b_col_zp, c_blk, RowsRemaining, CountN, K, lda, k_blks, ldc, bias
                );

                if (DataParams->PostProcessor != nullptr) {
                    DataParams->PostProcessor->Process(
                        DataParams->C, RangeStartM + RangeCountM - RowsRemaining, RangeStartN,
                        RowsHandled, CountN, ldc
                    );
                }

                c_blk += ldc * RowsHandled;
                a_row += lda * RowsHandled;
                RowsRemaining -= RowsHandled;
            }
        }
        return;
    }

    constexpr size_t StrideN = 32;
    size_t bufsize = k_blks * BlkLen * StrideN * sizeof(float);
    MlasThreadedBufAlloc(bufsize);
    auto* dequant_b = reinterpret_cast<float*>(ThreadedBufHolder.get());
    //
    // Step through each slice of matrix B along the N dimension.
    //

    size_t CountN;
    for (size_t n = 0; n < RangeCountN; n += CountN) {
        CountN = std::min(RangeCountN - n, (size_t)StrideN);

        //
        // Step through each slice of matrix A along the M dimension.
        //
        const float* bias = (Bias == nullptr) ? nullptr : Bias + RangeStartN + n;
        const uint8_t* b_col = PackedBData + (RangeStartN + n) * ldb;
        const float* b_col_scale = PackedBScale + (RangeStartN + n) * k_blks;
        const uint8_t* b_col_zp =
            (PackedBZeroPoint == nullptr)
                ? nullptr
                : PackedBZeroPoint + (RangeStartN + n) * MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(k_blks);
        float* c_blk = C + n;
        const float* a_row = A;

        MlasQNBitBlkDequantBForSgemm<BlkBitWidth, BlkLen, KernelType>(
            dequant_b, b_col, b_col_scale, b_col_zp, CountN, K, k_blks
        );

        size_t RowsRemaining = RangeCountM;
        while (RowsRemaining > 0) {
#if defined(MLAS_TARGET_AMD64_IX86) || defined(MLAS_TARGET_POWER)
            auto RowsHandled = GetMlasPlatform().GemmFloatKernel(
                a_row, dequant_b, c_blk, K, RowsRemaining, CountN, lda, ldc, 1.f, true
            );
#else
            auto RowsHandled = MlasSgemmKernelZero(a_row, dequant_b, c_blk, K, RowsRemaining, CountN, lda, ldc, 1.f);
#endif

            if (bias) {
                MlasAddBiasForGemm(bias, c_blk, RowsHandled, CountN, ldc);
            }
            if (DataParams->PostProcessor != nullptr) {
                DataParams->PostProcessor->Process(
                    DataParams->C, RangeStartM + RangeCountM - RowsRemaining, RangeStartN,
                    RowsHandled, CountN, ldc
                );
            }

            c_blk += ldc * RowsHandled;
            a_row += lda * RowsHandled;
            RowsRemaining -= RowsHandled;
        }
    }
}

//
// Kernel dispatch structure.
//

typedef void(MLASCALL MLAS_SQNBIT_GEMM_OPERATION)(
    size_t K,
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* DataParams,
    size_t RangeStartM,
    size_t RangeCountM,
    size_t RangeStartN,
    size_t RangeCountN
);

enum QuantVariant {
    QuantVariant_BitWidth4_BlockSize16,
    QuantVariant_BitWidth4_BlockSize32,
    QuantVariantCount,  // keep this element last
};

struct MLAS_SQNBIT_GEMM_DISPATCH {
    MLAS_SQNBIT_GEMM_OPERATION* Operations[QuantVariantCount] = {
        // Initialized to nullptrs. Overwrite in hardware-specific kernel implementation.
    };
};
