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

/**
 * @brief Multiply float matrix A with quantized n-bit integer matrix B.
 *        B is block quantized and column major.
 *
 * @tparam BlkBitWidth  Bit width of each value in a block.
 * @tparam BlkLen       Number of values in a block.
 * @tparam KernelType   Hardware-specific kernel type.
 *
 * @param       A                   Supplies the A matrix.
 * @param       QuantBData          Supplies the quantized B matrix block data.
 * @param       QuantBScale         Supplies the quantized B matrix block scale values.
 * @param       QuantBZeroPoint     Supplies the quantized B matrix block zero point values. Optional.
 * @param[out]  C                   Supplies the output C matrix.
 * @param       CountM              Number of rows of A and C.
 * @param       CountN              Number of columns of B and C.
 * @param       CountK              Number of columns of A and rows of B.
 * @param       lda                 Leading dimension of A.
 * @param       BlockStrideQuantB   Number of blocks between adjacent columns of the quantized B matrix.
 * @param       ldc                 Leading dimension of C.
 * @param       Bias                Bias vector of length N.
 *
 * @return  Number of rows of A handled.
 */
template <size_t BlkBitWidth, size_t BlkLen, typename KernelType>
MLAS_FORCEINLINE size_t
MlasSQNBitGemmKernel(
    const float* A,
    const uint8_t* QuantBData,
    const float* QuantBScale,
    const uint8_t* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t BlockStrideQuantB,
    size_t ldc,
    const float* Bias
);

/**
 * @brief Dequantize B into the format expected by the Sgemm kernel.
 *        B is block quantized and column major.
 *        This is equivalent to dequantizing B and then running
 *        MlasSgemmCopyPackB.
 *
 * @tparam BlkBitWidth  Bit width of each value in a block.
 * @tparam BlkLen       Number of values in a block.
 * @tparam KernelType   Hardware-specific kernel type.
 *
 * @param[out]  FpData              Supplies the output buffer for the dequantized B float data.
 * @param       QuantBData          Supplies the quantized B matrix block data.
 * @param       QuantBScale         Supplies the quantized B matrix block scale values.
 * @param       QuantBZeroPoint     Supplies the quantized B matrix block zero point values. Optional.
 * @param       CountN              Number of columns of B.
 * @param       CountK              Number of rows of B.
 * @param       BlockStrideQuantB   Number of blocks between adjacent columns of the quantized B matrix.
 */
template <size_t BlkBitWidth, size_t BlkLen, typename KernelType>
MLAS_FORCEINLINE void
MlasQNBitBlkDequantBForSgemm(
    float* FpData,
    const uint8_t* QuantBData,
    const float* QuantBScale,
    const uint8_t* QuantBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockStrideQuantB
);

//
// MlasQNBitGemmOperation and helpers
//

constexpr MLAS_FORCEINLINE size_t
MlasQNBitBlkDataSizeInBytes(size_t BlkBitWidth, size_t BlkLen)
{
    return BlkLen * BlkBitWidth / 8;
}

template <size_t BlkBitWidth>
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
    const uint8_t* QuantBData = static_cast<const uint8_t*>(DataParams->QuantBData);
    const float* QuantBScale = DataParams->QuantBScale;
    const uint8_t* QuantBZeroPoint = static_cast<const uint8_t*>(DataParams->QuantBZeroPoint);
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
            const uint8_t* b_col = QuantBData + (RangeStartN + n) * ldb;
            const float* b_col_scale = QuantBScale + (RangeStartN + n) * k_blks;
            const uint8_t* b_col_zp =
                (QuantBZeroPoint == nullptr)
                    ? nullptr
                    : QuantBZeroPoint + (RangeStartN + n) * MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(k_blks);
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
        const uint8_t* b_col = QuantBData + (RangeStartN + n) * ldb;
        const float* b_col_scale = QuantBScale + (RangeStartN + n) * k_blks;
        const uint8_t* b_col_zp =
            (QuantBZeroPoint == nullptr)
                ? nullptr
                : QuantBZeroPoint + (RangeStartN + n) * MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(k_blks);
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
    QuantVariant_BitWidth4_BlockSize64,
    QuantVariant_BitWidth4_BlockSize128,
    QuantVariantCount,  // Keep this element last and ensure that its value is the number of other QuantVariant values.
                        // Its value is used as an array size.
};

struct MLAS_SQNBIT_GEMM_DISPATCH {
    MLAS_SQNBIT_GEMM_OPERATION* Operations[QuantVariantCount] = {
        // Initialized to nullptrs. Overwrite in hardware-specific kernel implementation.
    };
};
