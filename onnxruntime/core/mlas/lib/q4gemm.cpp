/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    q4gemm.cpp

Abstract:

    This module implements the fp32 matrix multiplication with compressed
    weight tensor (right hand side). The assumption is the right hand side
    tensor can be pre-packed and compressed using int-4 quantization to save
    memory.
--*/

#include "q4gemm.h"

#ifdef MLAS_JBLAS
#include "mlas_jblas_defs.h"
using namespace jblas;
#endif

size_t MLASCALL
MlasQ80BlkQuantSize(MLAS_BLK_QUANT_TYPE QType, size_t M, size_t K)
{
    if (GetMlasPlatform().Q8Q4GemmDispatch == nullptr) {
        return 0;
    }
    switch (QType) {
        case BlkQ4Zp8:
            return MlasQ80BlkQuantSizeImpl<MLAS_Q4TYPE_BLK1>(M, K);
        case BlkQ4Sym64:
            return MlasQ80BlkQuantSizeImpl<MLAS_Q4TYPE_BLK2>(M, K);
        case BlkQ4Sym128:
            return MlasQ80BlkQuantSizeImpl<MLAS_Q4TYPE_BLK4>(M, K);
        case BlkQ4SymPerN:
        default:
            return MlasQ80BlkQuantSizeImpl<MLAS_Q4TYPE_BLK0>(M, K);
    }
}

void MLASCALL
MlasQ80BlkQuant(MLAS_BLK_QUANT_TYPE QType,
                void* Qblob,
                const float* A,
                size_t M,
                size_t K,
                size_t lda,
                MLAS_THREADPOOL* ThreadPool)
{
    auto* dispatch = GetMlasPlatform().Q8Q4GemmDispatch;
    dispatch->Quants[QType](Qblob, A, M, K, lda, ThreadPool);
}

template <typename ParamBlockType>
MLAS_FORCEINLINE void
MlasQ4GemmBatchDriver(MLAS_BLK_QUANT_TYPE QType,
                      const size_t M,
                      const size_t N,
                      const size_t K,
                      const size_t BatchN,
                      const ParamBlockType* DataParams,
                      MLAS_THREADPOOL* ThreadPool)
{
    // const MLAS_Q4GEMM_DISPATCH* dispatch = MlasQ4GemmGetDispatch();
    // MLAS_Q4GEMM_OPERATION* operation = dispatch->Operation;
    void (*operation)(const size_t, const ParamBlockType*, const size_t, const size_t, const size_t,
                      const size_t) = nullptr;

    if constexpr (std::is_same_v<ParamBlockType, MLAS_Q4_GEMM_DATA_PARAMS>) {
        operation = GetMlasPlatform().FpQ4GemmDispatch->Operations[QType];
    } else {
        operation = GetMlasPlatform().Q8Q4GemmDispatch->Operations[QType];
    }

    if (ThreadPool == nullptr) {
        for (size_t gemm_i = 0; gemm_i < BatchN; gemm_i++) {
            auto Data = &DataParams[gemm_i];
            operation(K, Data, 0, M, 0, N);
        }
        return;
    }

    //
    // Compute the number of target threads given the complexity of the SGEMM
    // operation. Small requests should run using the single threaded path.
    //

    const double Complexity = double(M) * double(N) * double(K) * double(BatchN);

    ptrdiff_t TargetThreadCount = ptrdiff_t(Complexity / double(MLAS_QGEMM_THREAD_COMPLEXITY)) + 1;

    ptrdiff_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool) * 8;

    if (TargetThreadCount >= MaximumThreadCount) {
        TargetThreadCount = MaximumThreadCount;
    }

    ptrdiff_t ThreadsPerGemm = TargetThreadCount / BatchN;
    if (ThreadsPerGemm < 1) {
        ThreadsPerGemm = 1;
    }

    constexpr size_t StrideM = 128;

    size_t nc = N;
    if (ThreadsPerGemm > 1) {
        // more than one thread per GEMM

        const size_t BlockedM = MlasDivRoundup(M, StrideM);
        const size_t max_nc = MlasDivRoundup(N * BlockedM, ThreadsPerGemm);
        if (max_nc < nc) {
            nc = std::min(nc, MlasDivRoundup(max_nc, MLAS_QGEMM_STRIDEN_THREAD_ALIGN) *
                                  MLAS_QGEMM_STRIDEN_THREAD_ALIGN);
        }
    }
    const size_t StrideN = nc;

    const size_t ThreadCountM = MlasDivRoundup(M, StrideM);
    const size_t ThreadCountN = MlasDivRoundup(N, StrideN);
    ThreadsPerGemm = ThreadCountM * ThreadCountN;

    MlasTrySimpleParallel(ThreadPool, ThreadsPerGemm * BatchN, [&](ptrdiff_t tid) {
        const auto gemm_i = tid / ThreadsPerGemm;
        const auto blk_i = tid % ThreadsPerGemm;
        auto Data = &DataParams[gemm_i];

        const ptrdiff_t ThreadIdN = blk_i / ThreadCountM;
        const ptrdiff_t ThreadIdM = blk_i % ThreadCountM;

        const size_t RangeStartM = ThreadIdM * StrideM;
        const size_t RangeCountM = std::min(M - RangeStartM, (size_t)StrideM);

        const size_t RangeStartN = ThreadIdN * StrideN;
        const size_t RangeCountN = std::min(N - RangeStartN, (size_t)StrideN);

        operation(K, Data, RangeStartM, RangeCountM, RangeStartN, RangeCountN);
    });
}

#ifdef MLAS_JBLAS

jblas::ORTThreading::ORTThreading(void* tp)
    : IThreading(MLAS_THREADPOOL::DegreeOfParallelism((MLAS_THREADPOOL*)tp)), mTp(tp)
{
}

void
jblas::ORTThreading::parallel_for(const jblas::parallel::thread_func& func)
{
    MlasTrySimpleParallel((MLAS_THREADPOOL*)mTp, mThreadNum,
                          [&](ptrdiff_t tid) { func(int(tid)); });
}

template <class Parallel_T, class Launch_T>
void
GemmKBlockRun(Launch_T& launcher,
              const typename Launch_T::Param& args,
              parallel::IThreading* threading)
{
    device::CpuBase cb;
    Parallel_T para({
        threading->num_threads(),
        cb.mL2Cache,
        args.M,
        args.N,
        args.K,
        args.KBlock,
    });
    threading->parallel_for([&](int tidx) {
        typename Parallel_T::ThreadProblem thdp{tidx};
        para.getIndex(thdp);
        if (thdp.valid) {
            launcher.run(args, thdp);
        }
    });
}

template <class GemmCore_T>
void
JblasQ4GemmCompF32(const int M,
                   const int N,
                   const int K,
                   const float* A,
                   const int lda,
                   jblas::storage::gemm::StorageWeightKBlockS4* B,
                   float* C,
                   const int ldc,
                   jblas::parallel::IThreading* th)
{
    using Parallel = jblas::parallel::gemm::SchedulerKBlock<GemmCore_T>;
    using Launcher = JBLAS_FP32_S4_F32F32<GemmCore_T>;
    typename Launcher::Param args{
        M,     N, K, B->mBlockSize, {A, K}, {B}, {B->template SPtr<int8_t>(), B->mScaT, B->mCStep},
        {C, N}};
    static Launcher kernel;
    GemmKBlockRun<Parallel>(kernel, args, th);
}

template <class GemmCore_T>
void
JblasQ4GemmCompInt8(const int M,
                    const int N,
                    const int K,
                    const float* A,
                    const int lda,
                    jblas::storage::gemm::StorageWeightKBlockS4* B,
                    float* C,
                    const int ldc,
                    jblas::parallel::IThreading* th)
{
    using Parallel = jblas::parallel::gemm::SchedulerKBlock<GemmCore_T>;
    using Launcher = JBLAS_INT8_S4_F32F32<GemmCore_T>;

    static Launcher kernel;
    auto quanA = kernel.mProA.createStorage(M, K, B->mBlockSize);
    auto buf = jblas::utils::amalloc<int8_t>(quanA.mSize);
    quanA.assign(buf);
    kernel.mProA.quantize({A, K, &quanA}, M, K, th);
    typename Launcher::Param args{
        M,
        N,
        K,
        B->mBlockSize,
        {A, K, &quanA},
        {B},
        {B->template SPtr<int8_t>(), B->mScaT, B->mCStep, quanA.template SPtr<float>(),
         quanA.mCStep, quanA.template ZPtr<uint8_t>(), B->template RPtr<float>(),
         B->mIsAsym ? B->template ZPtr<int8_t>() : nullptr, B->mIsAsym ? nullptr : nullptr,
         B->mBlockSize},
        {C, N}};
    GemmKBlockRun<Parallel>(kernel, args, th);
    jblas::utils::afree(buf);
}

void
JblasQ4GemmBatchDriver(const size_t M,
                       const size_t N,
                       const size_t K,
                       const size_t BatchN,
                       const MLAS_Q4_GEMM_DATA_PARAMS* DataParams,
                       MLAS_THREADPOOL* ThreadPool)
{
    GetCPUDevice();
    ORTThreading orth(ThreadPool);
    for (size_t i = 0; i < BatchN; i++) {
        auto ptr = jblas::storage::gemm::PackedWeightParser::deserialBuffer(
            const_cast<void*>(DataParams[i].B));
        if (ptr) {
            if (ptr->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockS4) {
                auto kptr = (jblas::storage::gemm::StorageWeightKBlockS4*)ptr;
                auto coretype = ptr->mCoreType;
                auto NTile = uint32_t(coretype) & uint32_t(JBLAS_GEMM_CORE::NTILE_MASK);
                auto CType = uint32_t(coretype) & uint32_t(JBLAS_GEMM_CORE::COMP_MASK);
                if (NTile == 48 && CType == uint32_t(JBLAS_GEMM_CORE::COMP_FP32)) {
                    if (_cd->AVX512F()) {
                        JblasQ4GemmCompF32<gemm::GemmCore_Row_NN_8x48_AVX512F>(
                            M, N, K, DataParams[i].A, DataParams[i].lda,
                            (jblas::storage::gemm::StorageWeightKBlockS4*)ptr, DataParams[i].C,
                            DataParams[i].ldc, &orth);
                        return;
                    }
                    if (_cd->AVX2()) {
                        JblasQ4GemmCompF32<gemm::GemmCore_Row_NN_2x48_AVX2>(
                            M, N, K, DataParams[i].A, DataParams[i].lda,
                            (jblas::storage::gemm::StorageWeightKBlockS4*)ptr, DataParams[i].C,
                            DataParams[i].ldc, &orth);
                        return;
                    }
                }
                if (NTile == 48 && CType == uint32_t(JBLAS_GEMM_CORE::COMP_INT8_US)) {
                    if (_cd->AVX512_VNNI()) {
                        JblasQ4GemmCompInt8<gemm::GemmCore_Row_NN_8x48_AVX512_VNNI>(
                            M, N, K, DataParams[i].A, DataParams[i].lda,
                            (jblas::storage::gemm::StorageWeightKBlockS4*)ptr, DataParams[i].C,
                            DataParams[i].ldc, &orth);
                        return;
                    }
                    if (_cd->AVX_VNNI()) {
                        JblasQ4GemmCompInt8<gemm::GemmCore_Row_NN_2x48_AVX_VNNI>(
                            M, N, K, DataParams[i].A, DataParams[i].lda,
                            (jblas::storage::gemm::StorageWeightKBlockS4*)ptr, DataParams[i].C,
                            DataParams[i].ldc, &orth);
                        return;
                    }
                }
            }
            delete ptr;
        }
    }
}

void MLASCALL
MlasJblasQ4GemmBatch(const size_t M,
                     const size_t N,
                     const size_t K,
                     const size_t BatchN,
                     const MLAS_Q4_GEMM_DATA_PARAMS* DataParams,
                     MLAS_THREADPOOL* ThreadPool)
{
    JblasQ4GemmBatchDriver(M, N, K, BatchN, DataParams, ThreadPool);
}
#endif

void MLASCALL
MlasQ4GemmBatch(MLAS_BLK_QUANT_TYPE QType,
                const size_t M,
                const size_t N,
                const size_t K,
                const size_t BatchN,
                const MLAS_Q4_GEMM_DATA_PARAMS* DataParams,
                MLAS_THREADPOOL* ThreadPool)
{
    MlasQ4GemmBatchDriver(QType, M, N, K, BatchN, DataParams, ThreadPool);
}

void MLASCALL
MlasQ8Q4GemmBatch(MLAS_BLK_QUANT_TYPE QType,
                  const size_t M,
                  const size_t N,
                  const size_t K,
                  const size_t BatchN,
                  const MLAS_Q8Q4_GEMM_DATA_PARAMS* DataParams,
                  MLAS_THREADPOOL* ThreadPool)
{
    MlasQ4GemmBatchDriver(QType, M, N, K, BatchN, DataParams, ThreadPool);
}
