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
#if !defined(__APPLE__)
#include "jblas/jit_blas_weight_compression.h"
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

#if !defined(__APPLE__)
template <class T, JBLAS_ISA ISA>
using WeiS4ClipFp32PerN =
    jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32PerN<T, ISA>;

template <template <class GC, JBLAS_ISA ISA> class ProB>
using AVX512VNNIFp32Fp32 = jblas::wrapper::gemm_pack_weight::GemmInterfaceParallelAB<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        JblasAVX512_VNNI,
        jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI,
        jblas::prologue::gemm::ActivationFp32AsymU8Quantize,
        ProB,
        jblas::epilogue::gemm::ZpDequantInt32ToFp32>,
    jblas::utils::parallel::Parallel2DGemm>;

template <template <class GC, JBLAS_ISA ISA> class ProB>
using AMXINT8Fp32Fp32 = jblas::wrapper::gemm_pack_weight::GemmInterfaceParallelAB<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        JblasAMX_INT8,
        jblas::gemm::GemmCore_Row_NN_16x48_AMX_S8S8,
        jblas::prologue::gemm::ActivationFp32SymS8Quantize,
        ProB,
        jblas::epilogue::gemm::DequantInt32ToFp32>,
    jblas::utils::parallel::Parallel2DGemm>;

static AVX512VNNIFp32Fp32<WeiS4ClipFp32PerN> avx512vnni_s4pernkernl;
static AMXINT8Fp32Fp32<WeiS4ClipFp32PerN> amxint8_s4pernkernl;

void
JblasQ4GemmBatchDriver(const size_t M,
                       const size_t N,
                       const size_t K,
                       const size_t BatchN,
                       const MLAS_Q4_GEMM_DATA_PARAMS* DataParams,
                       MLAS_THREADPOOL* ThreadPool)
{
    GetCPUDevice();
    for (size_t i = 0; i < BatchN; i++) {
        auto ptr = jblas::prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(
            const_cast<void*>(DataParams[i].B));
        if (ptr) {
            if (ptr->mPrologueID == int(jblas::prologue::weight_comp::gemm_kblcok::PrologueBIDs::
                                            WeightS4ClipScaleFp32PerChannelN)) {
                auto weiptr = (jblas::prologue::weight_comp::gemm_kblcok::
                                   StorageWeightS4ScaleFp32PerChannelN*)(ptr);
                if (ptr->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_8x48) {
                    if (_cd->AMX_INT8()) {
                        jblas::utils::request_perm_xtile_data();//TODO(yu) move to framework level, call once.
                        auto quanA = amxint8_s4pernkernl.getActivationPtr()->createStorage(M, K);
                        std::vector<int8_t> quanBuf(quanA.mSize);
                        quanA.assign(quanBuf.data());
                        amxint8_s4pernkernl.template compute<true, false>(
                            {int(M * BatchN), int(N), int(K), DataParams[i].A,
                             int(DataParams[i].lda), &quanA, weiptr, DataParams[i].C,
                             int(DataParams[i].ldc), quanA.mSPtr, quanA.mCStep, weiptr->mRPtr,
                             weiptr->mSPtr});
                    } else if (_cd->AVX512_VNNI()) {
                        auto quanA = avx512vnni_s4pernkernl.getActivationPtr()->createStorage(M, K);
                        std::vector<int8_t> quanBuf(quanA.mSize);
                        quanA.assign(quanBuf.data());
                        avx512vnni_s4pernkernl.template compute<true, false>(
                            {int(M * BatchN), int(N), int(K), DataParams[i].A,
                             int(DataParams[i].lda), &quanA, weiptr, DataParams[i].C,
                             int(DataParams[i].ldc), quanA.mZPtr, quanA.mSPtr, quanA.mCStep,
                             weiptr->mRPtr, weiptr->mSPtr});
                    }
                }
            }
            delete ptr;
        }
    }
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
#if !defined(__APPLE__)
    if (QType == MLAS_BLK_QUANT_TYPE::BlkQ4SymPerN) {
        return JblasQ4GemmBatchDriver(M, N, K, BatchN, DataParams, ThreadPool);
    }
#endif
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
