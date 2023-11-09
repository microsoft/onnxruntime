/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    q4gemm.h

Abstract:

    int4 block quantization gemm kernel template declarations.

    Int4 block quantization is used to compress weight tensors of large
    language models. It takes a number (must be multiple of 32) of floating
    point values, calculates their quantization parameters, and saves
    the parameters and the quantized data in a blob.
--*/

#include "q4common.h"


template<typename Q4TYPE, typename KERNEL>
MLAS_FORCEINLINE
size_t
MlasQ4GemmKernel(
    const float* A,
    const uint8_t* PackedB,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const float* Bias
);

template <typename Q4Type, typename KERNEL>
MLAS_FORCEINLINE
void
MlasBlkQ4DequantB(float* FpData, const uint8_t* PackedB, size_t CountN, size_t CountK, size_t ldb);


template <typename KERNEL>
MLAS_FORCEINLINE void
AddBiasAvx(const float* Bias, float* C, size_t CountM, size_t CountN, size_t ldc);



template <typename Q4TYPE, typename KERNEL>
void MLASCALL
MlasQ4GemmOperation(
    const size_t K,
    const MLAS_Q4_GEMM_DATA_PARAMS* DataParams,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
)
{
    const size_t lda = DataParams->lda;
    const size_t ldc = DataParams->ldc;

    const size_t k_blks = MlasDivRoundup(K, Q4TYPE::BlkLen);
    const size_t ldb = k_blks * Q4TYPE::BlobSize;
    const float* A = DataParams->A + RangeStartM * lda;
    const uint8_t* PackedB = (const uint8_t*)DataParams->B;
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
            const uint8_t* b_col = PackedB + (RangeStartN + n) * ldb;
            float* c_blk = C + n;
            const float* a_row = A;

            size_t RowsRemaining = RangeCountM;
            while (RowsRemaining > 0) {
                auto RowsHandled = MlasQ4GemmKernel<Q4TYPE, KERNEL>(
                    a_row, b_col, c_blk, RowsRemaining, CountN, K, lda, ldb, ldc, bias);

                if (DataParams->OutputProcessor != nullptr) {
                    DataParams->OutputProcessor->Process(
                        DataParams->C, RangeStartM + RangeCountM - RowsRemaining, RangeStartN,
                        RowsHandled, CountN, ldc);
                }

                c_blk += ldc * RowsHandled;
                a_row += lda * RowsHandled;
                RowsRemaining -= RowsHandled;
            }
        }
        return;
    }

    constexpr size_t StrideN = 32;
    size_t bufsize = k_blks * Q4TYPE::BlkLen * StrideN * sizeof(float);
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
        const uint8_t* b_col = PackedB + (RangeStartN + n) * ldb;
        float* c_blk = C + n;
        const float* a_row = A;

        MlasBlkQ4DequantB<Q4TYPE, KERNEL>(dequant_b, b_col,  CountN, K, ldb);

        size_t RowsRemaining = RangeCountM;
        while (RowsRemaining > 0) {
#if defined(MLAS_TARGET_AMD64_IX86) || defined(MLAS_TARGET_POWER)
            auto RowsHandled = GetMlasPlatform().GemmFloatKernel(
                a_row, dequant_b, c_blk, K, RowsRemaining, CountN, lda, ldc, 1.f, true);
#else
            auto RowsHandled = MlasSgemmKernelZero(a_row, dequant_b, c_blk, K, RowsRemaining,
                                                   CountN, lda, ldc, 1.f);
#endif

            if (bias) {
                AddBiasAvx<KERNEL>(bias, c_blk, RowsHandled, CountN, ldc);
            }
            if (DataParams->OutputProcessor != nullptr) {
                DataParams->OutputProcessor->Process(
                    DataParams->C, RangeStartM + RangeCountM - RowsRemaining, RangeStartN,
                    RowsHandled, CountN, ldc);
            }

            c_blk += ldc * RowsHandled;
            a_row += lda * RowsHandled;
            RowsRemaining -= RowsHandled;
        }
    }
}

typedef
void
(MLAS_Q4GEMM_OPERATION)(
    const size_t K,
    const MLAS_Q4_GEMM_DATA_PARAMS* DataParams,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
    );

struct MLAS_FPQ4GEMM_DISPATCH {
    MLAS_Q4GEMM_OPERATION** Operations;
};

/**
 * @brief Compute the size of a quantized block, one byte per value + fp32 scale
 * @tparam QType
 * @return
 */
template <typename QType>
constexpr size_t
Q8BlobUnitSize()
{
    return (QType::BlkLen + sizeof(float));
}

template <typename QType>
constexpr size_t
MlasQ80BlkQuantSizeImpl(size_t M, size_t K)
{
    const size_t KBlocks = MlasDivRoundup(K, QType::BlkLen);

    const size_t NumBlocks = M * KBlocks;

    return NumBlocks * Q8BlobUnitSize<QType>();
}

typedef
void
(MLAS_Q80_BLKQUANT)(
    void* Qblob,
    const float* A,
    size_t M,
    size_t K,
    size_t lda,
    MLAS_THREADPOOL* ThreadPool
    );

template<typename Q4TYPE, typename KERNEL>
MLAS_FORCEINLINE
size_t
MlasQ8Q4GemmKernel(
    const int8_t* QuantA,
    const uint8_t* PackedB,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const float* Bias
    );


template <typename Q4TYPE, typename KERNEL>
void MLASCALL
MlasQ8Q4GemmOperation(
    const size_t K,
    const MLAS_Q8Q4_GEMM_DATA_PARAMS* DataParams,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
)
{
    const size_t k_blks = MlasDivRoundup(K, Q4TYPE::BlkLen);
    const size_t ldb = k_blks * Q4TYPE::BlobSize;
    const size_t lda = k_blks * Q8BlobUnitSize<Q4TYPE>();
    const size_t ldc = DataParams->ldc;

    const int8_t* A = reinterpret_cast<const int8_t*>(DataParams->A) + RangeStartM * lda;
    const uint8_t* PackedB = (const uint8_t*)DataParams->B;
    float* C = DataParams->C + RangeStartM * ldc + RangeStartN;
    const float* Bias = DataParams->Bias;

    //
    // Step through each slice of matrix B along the N dimension.
    //

    size_t CountN;
    for (size_t n = 0; n < RangeCountN; n += CountN) {
        CountN = std::min(RangeCountN - n, (size_t)128);

        //
        // Step through each slice of matrix A along the M dimension.
        //
        const float* bias = (Bias == nullptr) ? nullptr : Bias + RangeStartN + n;
        const uint8_t* b_col = PackedB + (RangeStartN + n) * ldb;
        float* c_blk = C + n;
        const int8_t* a_row = A;

        size_t RowsRemaining = RangeCountM;
        while (RowsRemaining > 0) {
            auto RowsHandled = MlasQ8Q4GemmKernel<Q4TYPE, KERNEL>(
                a_row, b_col, c_blk, RowsRemaining, CountN, K, lda, ldb, ldc, bias);

            if (DataParams->OutputProcessor != nullptr) {
                DataParams->OutputProcessor->Process(
                    DataParams->C, RangeStartM + RangeCountM - RowsRemaining, RangeStartN,
                    RowsHandled, CountN, DataParams->ldc);
            }

            c_blk += ldc * RowsHandled;
            a_row += lda * RowsHandled;
            RowsRemaining -= RowsHandled;
        }
    }
}

typedef
void
(MLAS_Q8Q4GEMM_OPERATION)(
    const size_t K,
    const MLAS_Q8Q4_GEMM_DATA_PARAMS* DataParams,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
    );

struct MLAS_Q8Q4GEMM_DISPATCH {
    MLAS_Q80_BLKQUANT** Quants;
    MLAS_Q8Q4GEMM_OPERATION** Operations;
};
