/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm.cpp

Abstract:

    This module implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

--*/

#include "mlasi.h"
#include "qgemm.h"

//
// Define the parameters to execute segments of a QGEMM operation on worker
// threads.
//

struct MLAS_GEMM_QUANT_WORK_BLOCK {
    ptrdiff_t ThreadCountM;
    ptrdiff_t ThreadCountN;
};

void
MlasGemmQuantThreaded(
    const MLAS_GEMM_QUANT_WORK_BLOCK* WorkBlock,
    const MLAS_GEMM_QUANT_SHAPE_PARAMS* Shape,
    const MLAS_GEMM_QUANT_DATA_PARAMS* Data,
    ptrdiff_t ThreadId
    )
/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    QGEMM operation.

Arguments:

    ThreadInfo - Supplies the structure containing the thread task partition info.

    Shape - Supplies the structure containing the GEMM input and output shapes.

    Data  - Supplies the structure containing the GEMM input and output data layout

    ThreadId - Supplies the current index of the threaded operation.

Return Value:

    None.

--*/
{
    const ptrdiff_t ThreadIdM = ThreadId / WorkBlock->ThreadCountN;
    const ptrdiff_t ThreadIdN = ThreadId % WorkBlock->ThreadCountN;

    //
    // Partition the operation along the M dimension.
    //

    size_t RangeStartM;
    size_t RangeCountM;

    const size_t M = Shape->M;

    MlasPartitionWork(ThreadIdM, WorkBlock->ThreadCountM, M, &RangeStartM, &RangeCountM);

    //
    // Partition the operation along the N dimension.
    //

    size_t RangeStartN;
    size_t RangeCountN;

    const size_t N = Shape->N;

    const size_t BlockedN = (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) /
        MLAS_QGEMM_STRIDEN_THREAD_ALIGN;

    MlasPartitionWork(ThreadIdN, WorkBlock->ThreadCountN, BlockedN,
        &RangeStartN, &RangeCountN);

    RangeStartN *= MLAS_QGEMM_STRIDEN_THREAD_ALIGN;
    RangeCountN *= MLAS_QGEMM_STRIDEN_THREAD_ALIGN;

    RangeCountN = std::min(N - RangeStartN, RangeCountN);

    //
    // Dispatch the partitioned operation.
    //

    const auto* GemmQuantDispatch = MlasGemmQuantGetDispatch(Shape->AIsSigned, Shape->BIsSigned);
    MLAS_GEMM_QUANT_OPERATION* GemmQuantOperation;

    if (Data->BIsPacked) {
        GemmQuantOperation = GemmQuantDispatch->PackedOperation;
    } else {
        GemmQuantOperation = GemmQuantDispatch->Operation;
    }

    GemmQuantOperation(Shape, Data, RangeStartM, RangeCountM, RangeStartN, RangeCountN);
}


#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
// VC++ suggests we can attempt to make 'MlasBitsOfFp32' constexpr, but it is not valid.
#pragma warning(disable : 26451)
#endif

void
MLASCALL
MlasGemmBatch(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_QUANT_DATA_PARAMS* DataParams,
    const size_t BatchN,
    MLAS_THREADPOOL* ThreadPool)
{
    const size_t M = Shape.M;
    const size_t N = Shape.N;
    const size_t K = Shape.K;

    //
    // Compute the number of target threads given the complexity of the SGEMM
    // operation. Small requests should run using the single threaded path.
    //

    const double Complexity = double(M) * double(N) * double(K) * double(BatchN);

    ptrdiff_t TargetThreadCount;

    if (Complexity < double(MLAS_QGEMM_THREAD_COMPLEXITY * GetMlasPlatform().MaximumThreadCount)) {
        TargetThreadCount = ptrdiff_t(Complexity / double(MLAS_QGEMM_THREAD_COMPLEXITY)) + 1;
    } else {
        TargetThreadCount = GetMlasPlatform().MaximumThreadCount;
    }

    ptrdiff_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool);

    if (TargetThreadCount >= MaximumThreadCount) {
        TargetThreadCount = MaximumThreadCount;
    }

    ptrdiff_t ThreadsPerGemm = TargetThreadCount / BatchN;
    if (ThreadsPerGemm < 1) {
        ThreadsPerGemm = 1;
    }

    //
    // Segment the operation across multiple threads.
    //
    // N.B. Currently, the operation is segmented as a 1D partition, which
    // works okay for operations involving skinny matrices.
    //

    MLAS_GEMM_QUANT_WORK_BLOCK WorkBlock;

    if (N > M) {

        const size_t BlockedN = (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) /
            MLAS_QGEMM_STRIDEN_THREAD_ALIGN;

        if (size_t(ThreadsPerGemm) > BlockedN) {
            ThreadsPerGemm = ptrdiff_t(BlockedN);
        }

        WorkBlock.ThreadCountM = 1;
        WorkBlock.ThreadCountN = ThreadsPerGemm;

    } else {

        if (size_t(ThreadsPerGemm) > M) {
            ThreadsPerGemm = ptrdiff_t(M);
        }

        WorkBlock.ThreadCountM = ThreadsPerGemm;
        WorkBlock.ThreadCountN = 1;
    }
    TargetThreadCount = ThreadsPerGemm * BatchN;

    MlasTrySimpleParallel(ThreadPool, TargetThreadCount, [&](ptrdiff_t tid) {
        const auto gemm_i = tid / ThreadsPerGemm;
        const auto blk_i = tid % ThreadsPerGemm;
        MlasGemmQuantThreaded(&WorkBlock, &Shape, &DataParams[gemm_i], blk_i);
    });
}

void
MLASCALL
MlasSymmQgemmBatch(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_SYMM_QGEMM_DATA_PARAMS* DataParams,
    const size_t BatchN,
    MLAS_THREADPOOL* ThreadPool
    )
{
    const size_t M = Shape.M;
    const size_t N = Shape.N;
    const size_t K = Shape.K;
    const MLAS_SYMM_QGEMM_DISPATCH* dispatch = GetMlasPlatform().SymmQgemmDispatch;

    if (ThreadPool == nullptr) {
        // So our caller handles threaded job partition.
        // Call single threaded operation directly
        auto uarch = GetMlasPlatform().GetCoreType();
        MLAS_SYMM_QGEMM_OPERATION* operation =
            uarch == mlas_core_little ? dispatch->LitOperation : dispatch->BigOperation;

        for (size_t gemm_i = 0; gemm_i < BatchN; gemm_i++) {
            auto Data = &DataParams[gemm_i];
            operation(&Shape, Data, 0, M, 0, N);
        }
        return;
    }

    //
    // Compute the number of target threads given the complexity of the SGEMM
    // operation. Small requests should run using the single threaded path.
    //

    const double Complexity = double(M) * double(N) * double(K) * double(BatchN);

    ptrdiff_t TargetThreadCount = ptrdiff_t(Complexity / double(MLAS_QGEMM_THREAD_COMPLEXITY)) + 1;

    ptrdiff_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool);

    if (TargetThreadCount >= MaximumThreadCount) {
        TargetThreadCount = MaximumThreadCount;
    }

    ptrdiff_t ThreadsPerGemm = TargetThreadCount / BatchN;
    if (ThreadsPerGemm < 1) {
        ThreadsPerGemm = 1;
    }

    const size_t StrideM = dispatch->StrideM;

    size_t nc = N;
    if ((size_t)MlasGetMaximumThreadCount(ThreadPool) > BatchN) {
        // more than one thread per GEMM

        const size_t BlockedM = MlasDivRoundup(M, StrideM);
        const size_t max_nc = MlasDivRoundup(N * BlockedM, ThreadsPerGemm);
        if (max_nc < nc) {
            nc = std::min(nc, MlasDivRoundup(nc, max_nc * MLAS_QGEMM_STRIDEN_THREAD_ALIGN) *
                                  MLAS_QGEMM_STRIDEN_THREAD_ALIGN);
        }
    }
    const size_t StrideN = nc;

    const size_t ThreadCountM = MlasDivRoundup(M, StrideM);
    const size_t ThreadCountN = MlasDivRoundup(N, StrideN);
    ThreadsPerGemm = ThreadCountM * ThreadCountN;

    MlasTrySimpleParallel(ThreadPool, ThreadsPerGemm * BatchN, [&](ptrdiff_t tid) {
        auto uarch = GetMlasPlatform().GetCoreType();
        MLAS_SYMM_QGEMM_OPERATION* operation =
            uarch == mlas_core_little ? dispatch->LitOperation : dispatch->BigOperation;

        const auto gemm_i = tid / ThreadsPerGemm;
        const auto blk_i = tid % ThreadsPerGemm;
        auto Data = &DataParams[gemm_i];

        const ptrdiff_t ThreadIdN = blk_i / ThreadCountM;
        const ptrdiff_t ThreadIdM = blk_i % ThreadCountM;

        const size_t RangeStartM = ThreadIdM * StrideM;
        const size_t RangeCountM = std::min(Shape.M - RangeStartM, (size_t)StrideM);

        const size_t RangeStartN = ThreadIdN * StrideN;
        const size_t RangeCountN = std::min(Shape.N - RangeStartN, (size_t)StrideN);

        operation(&Shape, Data, RangeStartM, RangeCountM, RangeStartN, RangeCountN);
    });
}

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif

size_t
MLASCALL
MlasGemmPackBSize(
    size_t N,
    size_t K, 
    bool AIsSigned,
    bool BIsSigned
    )
/*++

Routine Description:

    This routine computes the number of bytes required to pack a matrix with
    the supplied shape and type.

Arguments:

    N - Supplies the number of columns of matrix B.

    K - Supplies the the number of rows of matrix B.

    BIsSigned - Supplies true if matrix B is signed data, else false if matrix
        B is unsigned data.

Return Value:

    Returns the number of bytes required to pack the matrix, else zero if the
        current implementation does not support packing.

--*/
{
    //
    // Retrieve the packing parameters.
    //

    const auto* GemmQuantDispatch = MlasGemmQuantGetDispatch(AIsSigned, BIsSigned);

    size_t PackedK = GemmQuantDispatch->PackedK;
    size_t PackedStrideK = GemmQuantDispatch->PackedStrideK;

    if (PackedStrideK == 0) {
        return 0;
    }

    //
    // Compute the number of bytes required to hold the packed buffer.
    //

    const size_t AlignedN =
        (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);
    const size_t AlignedK = (K + PackedK - 1) & ~(PackedK - 1);

    const size_t BytesRequired =
        (AlignedN * sizeof(int32_t)) + (AlignedN * AlignedK * sizeof(uint8_t));
    const size_t BufferAlignment = MlasGetPreferredBufferAlignment();
    const size_t AlignedBytesRequired = (BytesRequired + BufferAlignment - 1) &
        ~(BufferAlignment - 1);

    return AlignedBytesRequired;
}

void
MLASCALL
MlasGemmPackB(
    size_t N,
    size_t K,
    const uint8_t* B,
    size_t ldb,
    bool AIsSigned,
    bool BIsSigned,
    void* PackedB
    )
/*++

Routine Description:

    This routine packs the supplied matrix B to the supplied packed matrix B
    buffer. The size of the packed buffer was obtained from MlasGemmPackBSize.

Arguments:

    N - Supplies the number of columns of matrix B.

    K - Supplies the the number of rows of matrix B.

    B - Supplies the address of matrix B.

    ldb - Supplies the first dimension of matrix B.

    BIsSigned - Supplies true if matrix B is signed data, else false if matrix
        B is unsigned data.

    PackedB - Supplies the address of packed matrix B.

Return Value:

    None.

--*/
{
    //
    // Retrieve the packing parameters.
    //

    const auto* GemmQuantDispatch = MlasGemmQuantGetDispatch(AIsSigned, BIsSigned);

    size_t PackedK = GemmQuantDispatch->PackedK;
    size_t PackedStrideK = GemmQuantDispatch->PackedStrideK;

    //
    // Reserve and initialize storage for the column sum buffer to hold the sums
    // of the elements along each of the columns.
    //

    const size_t AlignedN =
        (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);

    int32_t* PackedColumnSumBuffer = (int32_t*)PackedB;
    std::fill_n(PackedColumnSumBuffer, AlignedN, 0);
    PackedB = PackedColumnSumBuffer + AlignedN;

    //
    // Step through each slice of matrix B along the K dimension.
    //

    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {

        CountK = std::min(K - k, PackedStrideK);

        //
        // Step through each slice of matrix B along the N dimension.
        //

        const size_t AlignedK = (CountK + PackedK - 1) & ~(PackedK - 1);
        uint8_t* pb = (uint8_t*)PackedB;
        size_t CountN;

        for (size_t n = 0; n < N; n += CountN) {

            constexpr size_t BatchedN = 128;
            MLAS_DECLSPEC_ALIGN(int32_t ColumnSumBuffer[BatchedN], 64);

            CountN = std::min(N - n, BatchedN);

            GemmQuantDispatch->CopyPackBRoutine(pb, B + n, ldb, CountN, CountK, ColumnSumBuffer, BIsSigned);

            //
            // Accumulate this batch of the column sum buffer into the packed
            // buffer accumulators.
            //

            for (size_t nn = 0; nn < CountN; nn++) {
                PackedColumnSumBuffer[n + nn] += ColumnSumBuffer[nn];
            }

            pb += CountN * AlignedK;
        }

        PackedB = (uint8_t*)PackedB + AlignedN * AlignedK;
        B += ldb * CountK;
    }
}

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
// We can not make this function constexpr across different platforms
#pragma warning(disable : 26497)
#endif

size_t
MLASCALL
MlasSymmQgemmPackBSize(
    size_t N,
    size_t K, 
    bool AIsSigned
    )
{
#ifndef MLAS_TARGET_ARM64

    // Only have arm64 impl for now
    MLAS_UNREFERENCED_PARAMETER(N);
    MLAS_UNREFERENCED_PARAMETER(K);
    MLAS_UNREFERENCED_PARAMETER(AIsSigned);
    return 0;
#else

    // Only support s8s8 for now
    if (!AIsSigned) {
        return 0;
    }

    const auto* Dispatch = GetMlasPlatform().SymmQgemmDispatch;

    size_t PackedK = Dispatch->PackedK;

    //
    // Compute the number of bytes required to hold the packed buffer.
    //

    const size_t AlignedN =
        (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);
    const size_t AlignedK = (K + PackedK - 1) & ~(PackedK - 1);

    const size_t BytesRequired =
        (AlignedN * sizeof(int32_t)) + (AlignedN * AlignedK * sizeof(uint8_t));
    const size_t BufferAlignment = MlasGetPreferredBufferAlignment();
    const size_t AlignedBytesRequired = (BytesRequired + BufferAlignment - 1) &
        ~(BufferAlignment - 1);

    return AlignedBytesRequired;
#endif  // !MLAS_TARGET_ARM64
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif


void
MLASCALL
MlasSymmQgemmPackB(
    size_t N,
    size_t K,
    const int8_t* B,
    size_t ldb,
    bool AIsSigned,
    int32_t ZeroPointA,
    void* PackedB
    )
{
    MLAS_UNREFERENCED_PARAMETER(AIsSigned);

    const MLAS_SYMM_QGEMM_DISPATCH* SymmQgemmDispatch = GetMlasPlatform().SymmQgemmDispatch;

    const size_t AlignedN =
        (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);
    int32_t* PackedColumnSumBuffer = (int32_t*)PackedB;
    PackedB = PackedColumnSumBuffer + AlignedN;

    SymmQgemmDispatch->CopyPackBRoutine((uint8_t*)PackedB, (const uint8_t*)B, ldb, N, K,
                                        PackedColumnSumBuffer, true);
    for (size_t n = 0; n < AlignedN; n++) {
        PackedColumnSumBuffer[n] *= -ZeroPointA;
    }
}
