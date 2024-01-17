/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the MIT License.

Module Name:

    sbgemm.h

Abstract:

    This module defines the set of template functions to implement bfloat16
    precision matrix/matrix multiply operation (SBGEMM).

    To implement a new kernel, template functions below need to be specialized:
       MlasSBGemmConvertPackB
       MlasSBGemmPackedBOffset
       MlasSBGemmPackedBLeadingDim
       MlasSBGemmKernel

    MlasSBGemmOperation is the shared kernel driver.

    A kernel type should define the following constants:
        bool PackNeeded;         Whether B needs to be packed
        size_t KernelMaxM;       Max # rows the vectorized kernel can process
        size_t PackedK;          Packed alignment on the K dim (power of 2)
        size_t PackedN;          Packed alignment on the n dim (power of 2)
        MLAS_SBGEMM_STRIDES Strides{128, 128, 256};
--*/

#if defined(__aarch64__) && defined(__linux__)

#pragma once

#include <cassert>
#include <cstdlib>
#include <string>

#include "mlasi.h"

/**
 * @brief Define the default striding parameters for
 *        the bfloat16 precision gemm operation
 */
struct MLAS_SBGEMM_STRIDES {
    size_t M;
    size_t N;
    size_t K;
};

/**
 * @brief Convert fp32 matrix B to bf16 and pack the data
 *
 * @tparam KernelType
 * @param[out] D         Address of packing buffer
 * @param[in]  B         Address of source matrix B in fp32
 * @param[in]  ldb       Leading dimension of B
 * @param[in]  CountN    # of column to pack
 * @param[in]  CountK    # of rows to pack
 */
template <typename KernelType>
void
MlasSBGemmConvertPackB(
    bfloat16_t* PackedB, const float* B, size_t ldb, size_t CountN, size_t CountK);

/**
 * @brief Find the location of PackedB[StartK, StartN]
 *
 * @tparam KernelType
 * @param PackedB
 * @param DimN       Total columns of the packing buffer
 * @param DimK       Total rows of the packing buffer
 * @param StartN
 * @param StartK
 * @return  Address of PackedB[StartK, StartN]
 */
template <typename KernelType>
MLAS_FORCEINLINE const bfloat16_t*
MlasSBGemmPackedBOffset(
    const bfloat16_t* PackedB, size_t DimN, size_t DimK, size_t StartN, size_t StartK)
{
    // By default the packed buffer is just a row major
    // K row by N column buffer
    MLAS_UNREFERENCED_PARAMETER(DimK);
    return PackedB + StartK * DimN + StartN;
}

/**
 * @brief leading dimension of the packed B buffer
 *        Related to how B is packed
 * @tparam KernelType
 * @param DimN
 * @param DimK
 * @return leading dimension of the packed B buffer
 */
template <typename KernelType>
MLAS_FORCEINLINE size_t
MlasSBGemmPackedBLeadingDim(size_t DimN, size_t DimK)
{
    // By default the packed buffer is just a row major
    // K row by N column buffer
    MLAS_UNREFERENCED_PARAMETER(DimK);
    return DimN;
}

template <typename KernelType>
void
MlasSBGemmKernel(const size_t CountM,
                 const size_t CountN,
                 const size_t CountK,
                 const float* A,
                 const size_t lda,
                 const bfloat16_t* B,
                 float* C,
                 size_t ldc,
                 const float* Bias,
                 const bool ZeroMode);

template <typename KernelType>
MLAS_FORCEINLINE void
MlasSBGemmPackedOperation(size_t M,
                          size_t RangeStartN,
                          size_t RangeCountN,
                          size_t AlignedN,
                          size_t K,
                          const float* A,
                          size_t lda,
                          const void* PackedB,
                          float* C,
                          size_t ldc,
                          const float* Bias,
                          void* PostProcessor)
{
    constexpr MLAS_SBGEMM_STRIDES Strides = KernelType::Strides;
    size_t PackedStrideN = Strides.N;
    size_t PackedStrideK = Strides.K;

    //
    // Step through each slice of matrix B along the N dimension.
    //
    size_t CountN;
    for (size_t n = 0; n < RangeCountN; n += CountN) {
        const size_t SliceStartN = RangeStartN + n;
        CountN = std::min(RangeCountN - n, PackedStrideN);

        //
        // Step through each slice of matrix B along the K dimension.
        //
        size_t CountK;
        for (size_t k = 0; k < K; k += CountK) {
            bool ZeroMode = (k == 0);
            CountK = std::min(K - k, PackedStrideK);

            const bfloat16_t* pb = (const bfloat16_t*)PackedB + AlignedN * k + CountK * SliceStartN;
            float* c = C + n;
            const float* pbias = ((nullptr == Bias) ? nullptr : Bias + RangeStartN + n);
            MlasSBGemmKernel<KernelType>(M, CountN, CountK, A + k, lda, pb, c, ldc,
                                         ZeroMode ? pbias : nullptr, ZeroMode);
        }
        if (PostProcessor != nullptr) {
            ((MLAS_SBGEMM_POSTPROCESSOR*)PostProcessor)
                ->Process(C + n, M, SliceStartN, M, CountN, ldc);
        }
    }
}

template <typename KernelType>
void
MlasSBGemmNonPackedOperation(size_t M,
                             size_t N,
                             size_t K,
                             const float* A,
                             size_t lda,
                             const float* B,
                             size_t ldb,
                             float* C,
                             size_t ldc,
                             const float* Bias,
                             void* PostProcessor)
{
    //
    // Compute the strides to step through slices of the input matrices.
    //
    // Expand the N stride if K is small or expand the K stride if N is small
    // for better utilization of the B panel. Avoid changing the K stride if
    // the A panel needs to be used for transposing.
    //
    constexpr MLAS_SBGEMM_STRIDES Strides = KernelType::Strides;
    size_t StrideN = Strides.N;
    size_t StrideK = Strides.K;

    if (N >= K) {
        while (StrideK / 2 >= K) {
            StrideN *= 2;
            StrideK /= 2;
        }
    } else {
        while (StrideN > 16 && StrideN / 2 >= N) {
            StrideK *= 2;
            StrideN /= 2;
        }
    }

    constexpr size_t packBSize = UpAlignSize(Strides.N * Strides.K * sizeof(bfloat16_t));
    MlasThreadedBufAlloc(packBSize);
    uint8_t* p = ThreadedBufHolder.get();
    auto* PanelB = reinterpret_cast<bfloat16_t*>(p);

    //
    // Step through each slice of matrix B along the N dimension.
    //
    size_t CountN;
    for (size_t n = 0; n < N; n += CountN) {
        CountN = std::min(N - n, StrideN);

        //
        // Step through each slice of matrix B along the N dimension.
        //
        size_t CountK;
        for (size_t k = 0; k < K; k += CountK) {
            CountK = std::min(K - k, StrideK);

            //
            // Copy a panel of matrix B to a local packed buffer.
            //
            MlasSBGemmConvertPackB<KernelType>(PanelB, B + n + k * ldb, ldb, CountN, CountK);

            auto* c = C + n;
            const float* pbias =
                ((nullptr == Bias) ? nullptr : Bias + n);  // TODO: check the SliceNStart

            bool ZeroMode = (k == 0);
            MlasSBGemmKernel<KernelType>(M, CountN, CountK, A + k, lda, PanelB, c, ldc,
                                         ZeroMode ? pbias : nullptr, ZeroMode);
        }
        if (PostProcessor != nullptr) {
            ((MLAS_SBGEMM_POSTPROCESSOR*)PostProcessor)->Process(C + n, M, N, M, CountN, ldc);
        }
    }
}

template <typename KernelType>
void
MlasSBGemmOperation(const ptrdiff_t ThreadCountM,
                    const ptrdiff_t ThreadCountN,
                    const size_t M,
                    const size_t N,
                    const size_t K,
                    const MLAS_SBGEMM_DATA_PARAMS* DataParams,
                    ptrdiff_t ThreadId)
{
    const ptrdiff_t ThreadIdM = ThreadId / ThreadCountN;
    const ptrdiff_t ThreadIdN = ThreadId % ThreadCountN;

    //
    // Partition the operation along the M dimension.
    //
    size_t RangeStartM;
    size_t RangeCountM;

    MlasPartitionWork(ThreadIdM, ThreadCountM, M, &RangeStartM, &RangeCountM);

    //
    // Partition the operation along the N dimension.
    //
    size_t RangeStartN;
    size_t RangeCountN;

    const size_t BlockedN =
        (N + MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1) / MLAS_SGEMM_STRIDEN_THREAD_ALIGN;

    MlasPartitionWork(ThreadIdN, ThreadCountN, BlockedN, &RangeStartN, &RangeCountN);

    RangeStartN *= MLAS_SGEMM_STRIDEN_THREAD_ALIGN;
    RangeCountN *= MLAS_SGEMM_STRIDEN_THREAD_ALIGN;

    RangeCountN = std::min(N - RangeStartN, RangeCountN);

    //
    // Dispatch the partitioned operation.
    //
    const size_t lda = DataParams->lda;
    const size_t ldc = DataParams->ldc;
    const float* A = (const float*)DataParams->A + RangeStartM * lda;
    float* C = DataParams->C + RangeStartM * ldc + RangeStartN;
    const float* bias = DataParams->Bias;

    if (!DataParams->BIsfp32) {
        MlasSBGemmPackedOperation<KernelType>(
            RangeCountM, RangeStartN, RangeCountN, BlockedN * MLAS_SGEMM_STRIDEN_THREAD_ALIGN, K, A,
            lda, DataParams->B, C, ldc, bias, (void*)DataParams->OutputProcessor);
    } else {
        const size_t ldb = DataParams->ldb;
        const float* B = (const float*)DataParams->B + RangeStartN;
        MlasSBGemmNonPackedOperation<KernelType>(RangeCountM, RangeCountN, K, A, lda, B, ldb, C,
                                                 ldc, bias, (void*)DataParams->OutputProcessor);
    }
}

//
// dispatch structure.
//
typedef void(MLAS_SBGEMM_OPERATION)(const ptrdiff_t ThreadCountM,
                                    const ptrdiff_t ThreadCountN,
                                    const size_t M,
                                    const size_t N,
                                    const size_t K,
                                    const MLAS_SBGEMM_DATA_PARAMS* DataParams,
                                    ptrdiff_t ThreadId);

typedef void(MLAS_SBGEMM_CONVERTPACKB_ROUTINE)(
    bfloat16_t* D, const float* B, size_t ldb, size_t CountN, size_t CountK);

/**
 * @brief Hardware dependent dispatch for half precision GEMM
 */
struct MLAS_SBGEMM_DISPATCH {
    MLAS_SBGEMM_OPERATION* Operation;                      /**< HalfGemm driver */
    MLAS_SBGEMM_CONVERTPACKB_ROUTINE* ConvertPackBRoutine; /**< Convert and pack function for B */
    size_t PackedK;
    size_t PackedN;
    size_t StrideM;
    size_t BufOverRead;
};

extern const MLAS_SBGEMM_DISPATCH MlasSBGemmDispatchNeon;

MLAS_FORCEINLINE
const MLAS_SBGEMM_DISPATCH*
MlasSBGemmGetDispatch()
{
#if defined(MLAS_TARGET_ARM64)
    return &MlasSBGemmDispatchNeon;
#else
    std::cerr << "SBGemm Kernel is supported only on ARM64 platform.";
    exit(1);
#endif
}

size_t MLASCALL
MlasSBGemmPackBSize(size_t N, size_t K)
{
    //
    // Compute the number of bytes required to hold the packed buffer.
    //
    const auto* dispatch = MlasSBGemmGetDispatch();
    if (dispatch == nullptr) return 0;

    const auto padding = dispatch->BufOverRead;
    const auto PackedK = dispatch->PackedK;
    const auto PackedN = dispatch->PackedN;

    const size_t AlignedK = (K + PackedK - 1) & ~(PackedK - 1);
    const size_t AlignedN = (N + PackedN - 1) & ~(PackedN - 1);
    const size_t BytesRequired = AlignedN * AlignedK * sizeof(bfloat16_t) + padding;
    const size_t BufferAlignment = MlasGetPreferredBufferAlignment();
    const size_t AlignedBytesRequired =
        (BytesRequired + BufferAlignment - 1) & ~(BufferAlignment - 1);

    return AlignedBytesRequired;
}

void MLASCALL
MlasSBGemmConvertPackB(size_t N, size_t K, const float* B, size_t ldb, void* PackedB)
{
    const auto* dispatch = MlasSBGemmGetDispatch();
    if (dispatch == nullptr) return;

    dispatch->ConvertPackBRoutine((bfloat16_t*)PackedB, B, ldb, N, K);
}

void MLASCALL
MlasSBGemmBatch(const size_t M,
                const size_t N,
                const size_t K,
                const size_t BatchN,
                const MLAS_SBGEMM_DATA_PARAMS* Data,
                MLAS_THREADPOOL* ThreadPool)
{
    const MLAS_SBGEMM_DISPATCH* dispatch = MlasSBGemmGetDispatch();
    if (dispatch == nullptr) return;

    MLAS_SBGEMM_OPERATION* operation = dispatch->Operation;

    //
    // Compute the number of target threads given the complexity of the SGEMM
    // operation. Small requests should run using the single threaded path.
    //

    const double Complexity = double(M) * double(N) * double(K);

    ptrdiff_t TargetThreadCount;

    if (Complexity < double(MLAS_SBGEMM_THREAD_COMPLEXITY * GetMlasPlatform().MaximumThreadCount)) {
        TargetThreadCount = ptrdiff_t(Complexity / double(MLAS_SGEMM_THREAD_COMPLEXITY)) + 1;
    } else {
        TargetThreadCount = GetMlasPlatform().MaximumThreadCount;
    }

    ptrdiff_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool);

    if (TargetThreadCount >= MaximumThreadCount) {
        TargetThreadCount = MaximumThreadCount;
    }

    //
    // Segment the operation across multiple threads.
    //
    // N.B. Currently, the operation is segmented as a 1D partition, which
    // works okay for operations involving skinny matrices.
    //
    ptrdiff_t ThreadsPerGemm = (TargetThreadCount + BatchN - 1) / BatchN;
    ptrdiff_t ThreadCountM;
    ptrdiff_t ThreadCountN;

    if (N > M) {
        const size_t BlockedN =
            (N + MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1) / MLAS_SGEMM_STRIDEN_THREAD_ALIGN;

        if (size_t(ThreadsPerGemm) > BlockedN) {
            ThreadsPerGemm = ptrdiff_t(BlockedN);
        }

        ThreadCountM = 1;
        ThreadCountN = ThreadsPerGemm;

    } else {
        if (size_t(ThreadsPerGemm) > M) {
            ThreadsPerGemm = ptrdiff_t(M);
        }

        ThreadCountM = ThreadsPerGemm;
        ThreadCountN = 1;
    }

    MlasTrySimpleParallel(
        ThreadPool, ThreadsPerGemm * static_cast<ptrdiff_t>(BatchN), [=](ptrdiff_t tid) {
            ptrdiff_t GemmIdx = tid / ThreadsPerGemm;
            ptrdiff_t ThreadIdx = tid % ThreadsPerGemm;
            operation(ThreadCountM, ThreadCountN, M, N, K, &(Data[GemmIdx]), ThreadIdx);
        });
}
#endif //defined(__aarch64__) && defined(__linux__)
