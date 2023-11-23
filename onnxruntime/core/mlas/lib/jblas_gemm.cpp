/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    jblas_gemm.cpp

Abstract:

    Currently only support Q4 gemm.
--*/

#include "jblas_gemm.h"

#include "jblas_defs.h"
#include "mlasi.h"

using namespace jblas;

jblas::ORTThreading::ORTThreading(void* tp)
    : IThreading(MLAS_THREADPOOL::DegreeOfParallelism((MLAS_THREADPOOL*)tp)), mTp(tp)
{
}

void
jblas::ORTThreading::parallel_for(const jblas::parallel::thread_func& func)
{
    MlasTrySimpleParallel((MLAS_THREADPOOL*)mTp, mThreadNum, [&](ptrdiff_t tid) { func(int(tid)); });
}

template <class GemmCore_T>
void
JblasQ4GemmCompF32(
    const int M,
    const int N,
    const int K,
    const float* A,
    const int lda,
    jblas::storage::gemm::StorageWeightKBlockS4* B,
    float* C,
    const int ldc,
    int8_t* WorkSpace,
    jblas::parallel::IThreading* th
)
{
    if (M <= 32) {
        using Parallel = jblas::parallel::gemm::SchedulerKBlock<GemmCore_T>;
        using Launcher = tLauncher_Fp32_S4_F32F32<GemmCore_T>;
        static Launcher kernel;
        auto reduceA = kernel.mProA.createStorage(M, K, B->mBlockSize);
        if (B->mIsAsym) {
            reduceA.assign(WorkSpace);
            ORTThreading single(nullptr);
            kernel.mProA.reduce({A, K}, &reduceA, M, K, &single);
        }
        typename Launcher::BEpiParam blkargs{
            B->template SPtr<int8_t>(), B->mScaT, B->mCStep, B->template ZPtr<int8_t>(),
            reduceA.template get<float>(), reduceA.lda};

        typename Launcher::Param args{M, N, K, B->mBlockSize, {A, K}, {B}, blkargs, {C, N}};
        jblas::parallel::GemmKBlockRun<Parallel>(kernel, args, th);
    } else {
        using Parallel = jblas::parallel::gemm::SchedulerBase<GemmCore_T>;
        using Launcher = jblas::wrapper::gemm::LauncherBase<
            GemmCore_T::ISA, GemmCore_T, jblas::prologue_a::gemm::ActivationBase,
            jblas::prologue_b::gemm::WeightKBlockS4, jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
        static Launcher kernel;

        typename Launcher::Param args{M, N, K, {A, K}, {B}, {C, N}};
        jblas::parallel::GemmBaseRun<Parallel>(kernel, args, th);
    }
}

template <class GemmCore_T>
void
JblasQ4GemmCompInt8(
    const int M,
    const int N,
    const int K,
    const float* A,
    const int lda,
    jblas::storage::gemm::StorageWeightKBlockS4* B,
    float* C,
    const int ldc,
    int8_t* WorkSpace,
    jblas::parallel::IThreading* th
)
{
    using Parallel = jblas::parallel::gemm::SchedulerKBlock<GemmCore_T>;
    using Launcher = tLauncher_Int8_S4_F32F32<GemmCore_T>;

    static Launcher kernel;
    auto quanA = kernel.mProA.createStorage(M, K, B->mBlockSize, B->mIsAsym);
    quanA.assign(WorkSpace);
    if (M <= 32) {
        ORTThreading single(nullptr);
        kernel.mProA.quantize({A, K, &quanA}, M, K, &single);
    } else {
        kernel.mProA.quantize({A, K, &quanA}, M, K, th);
    }
    typename Launcher::Param args{
        M,
        N,
        K,
        B->mBlockSize,
        {A, K, &quanA},
        {B},
        {B->template SPtr<int8_t>(), B->mScaT, B->mCStep, quanA.template SPtr<float>(), quanA.mCStep,
         quanA.template ZPtr<uint8_t>(), B->template RPtr<float>(), B->mRedT, B->template ZPtr<int8_t>(),
         quanA.template RPtr<float>(), B->mBlockSize},
        {C, N}};
    jblas::parallel::GemmKBlockRun<Parallel>(kernel, args, th);
}

bool
JblasQ4GemmBatchDriver(
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const MLAS_NBITS_GEMM_DATA_PACKED_PARAMS* DataParams,
    int8_t* WorkSpace,
    MLAS_THREADPOOL* ThreadPool
)
{
    GetCPUDevice();
    ORTThreading orth(ThreadPool);
    bool processed = true;
    for (size_t i = 0; i < BatchN; i++) {
        auto ptr = jblas::storage::gemm::PackedWeightParser::deserialBuffer(const_cast<void*>(DataParams[i].B));
        if (ptr) {
            if (ptr->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockS4) {
                auto kptr = (jblas::storage::gemm::StorageWeightKBlockS4*)ptr;
                auto coretype = ptr->mCoreId;
                auto NTile = jblas::gemm::CoreAttr::get_mask_val(
                    ptr->mCoreId, jblas::gemm::CoreAttr::NTILE_MASK, jblas::gemm::CoreAttr::NTILE_SHIFT
                );
                auto CType = jblas::gemm::CoreAttr::get_mask_val(
                    ptr->mCoreId, jblas::gemm::CoreAttr::COMP_MASK, jblas::gemm::CoreAttr::COMP_SHIFT
                );
                if (CType == uint32_t(gemm::CompType::COMP_FP32)) {
                    if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
                        JblasQ4GemmCompF32<tAVX512F>(
                            M, N, K, DataParams[i].A, DataParams[i].lda,
                            (jblas::storage::gemm::StorageWeightKBlockS4*)ptr, DataParams[i].C, DataParams[i].ldc,
                            WorkSpace, &orth
                        );
                        goto __END;
                    }
                    if (NTile == tAVX2::NTILE && _cd->AVX2()) {
                        JblasQ4GemmCompF32<tAVX2>(
                            M, N, K, DataParams[i].A, DataParams[i].lda,
                            (jblas::storage::gemm::StorageWeightKBlockS4*)ptr, DataParams[i].C, DataParams[i].ldc,
                            WorkSpace, &orth
                        );
                        goto __END;
                    }
                }
                if (CType == uint32_t(gemm::CompType::COMP_INT8_US_INT32)) {
                    if (NTile == tAMX_INT8_US::NTILE && _cd->AMX_INT8()) {
                        JblasQ4GemmCompInt8<tAMX_INT8_US>(
                            M, N, K, DataParams[i].A, DataParams[i].lda,
                            (jblas::storage::gemm::StorageWeightKBlockS4*)ptr, DataParams[i].C, DataParams[i].ldc,
                            WorkSpace, &orth
                        );
                        goto __END;
                    }
                    if (NTile == tAVX512_VNNI::NTILE && _cd->AVX512_VNNI()) {
                        JblasQ4GemmCompInt8<tAVX512_VNNI>(
                            M, N, K, DataParams[i].A, DataParams[i].lda,
                            (jblas::storage::gemm::StorageWeightKBlockS4*)ptr, DataParams[i].C, DataParams[i].ldc,
                            WorkSpace, &orth
                        );
                        goto __END;
                    }
                    if (NTile == tAVX_VNNI::NTILE && _cd->AVX_VNNI()) {
                        JblasQ4GemmCompInt8<tAVX_VNNI>(
                            M, N, K, DataParams[i].A, DataParams[i].lda,
                            (jblas::storage::gemm::StorageWeightKBlockS4*)ptr, DataParams[i].C, DataParams[i].ldc,
                            WorkSpace, &orth
                        );
                        goto __END;
                    }
                }
                if (CType == uint32_t(gemm::CompType::COMP_INT8_SS_INT32)) {
                    if (NTile == tAMX_INT8_SS::NTILE && _cd->AMX_INT8()) {
                        JblasQ4GemmCompInt8<tAMX_INT8_SS>(
                            M, N, K, DataParams[i].A, DataParams[i].lda,
                            (jblas::storage::gemm::StorageWeightKBlockS4*)ptr, DataParams[i].C, DataParams[i].ldc,
                            WorkSpace, &orth
                        );
                        goto __END;
                    }
                }
            }
        __END:
            delete ptr;
        } else {
            processed = false;
            break;
        }
    }
    return processed;
}

template <typename T>
static size_t
JblasQ4BuSize(int block_size, size_t N, size_t K, bool isAsym)
{
    static T launcher;
    auto stor = launcher.mProB.createStorage(
        N, K, block_size, JBLAS_DTYPE::S4_CLIP, JBLAS_DTYPE::F32, JBLAS_DTYPE::BF16, isAsym
    );
    // TODO(Yu) support more S4 quant type, scale dtype
    return stor.mSize;
}

size_t
JblasQ4GemmPackBSize(size_t N, size_t K, size_t BlkSize, bool isAsym, MLAS_COMPUTE_TYPE CompType)
{
    GetCPUDevice();
    // from low precision to high precision
    switch (CompType) {
        case CompInt8:
            if (_cd->AMX_INT8() && BlkSize % tAMX_INT8_SS::KTILE == 0) {
                return JblasQ4BuSize<tLauncher_Int8_S4_F32F32<tAMX_INT8_SS>>(int(BlkSize), N, K, isAsym);
            }
            if (_cd->AVX512_VNNI() && BlkSize % tAVX512_VNNI::KTILE == 0) {
                return JblasQ4BuSize<tLauncher_Int8_S4_F32F32<tAVX512_VNNI>>(int(BlkSize), N, K, isAsym);
            }
            if (_cd->AVX_VNNI() && BlkSize % tAVX_VNNI::KTILE == 0) {
                return JblasQ4BuSize<tLauncher_Int8_S4_F32F32<tAVX_VNNI>>(int(BlkSize), N, K, isAsym);
            }
        case CompBf16:
        case CompFp16:
        case CompFp32:
        case CompUndef:
            if (_cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
                return JblasQ4BuSize<tLauncher_Int8_S4_F32F32<tAVX512F>>(int(BlkSize), N, K, isAsym);
            }
            if (_cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
                return JblasQ4BuSize<tLauncher_Int8_S4_F32F32<tAVX2>>(int(BlkSize), N, K, isAsym);
            }
            break;
        default:
            return 0;
    }
    return 0;
}

template <typename T>
void
JblaNBitsGemmPackB(
    void* PackedBuf,
    int BlkSize,
    const uint8_t* QData,
    const float* Scale,
    const uint8_t* Zp,
    int N,
    int K,
    bool IsAsym,
    bool lastCall,
    int ldb,
    MLAS_THREADPOOL* ThreadPool
)
{
    static T JblasKernel;
    auto stor = JblasKernel.mProB.createStorage(
        N, K, BlkSize, JBLAS_DTYPE::S4_CLIP, JBLAS_DTYPE::F32, JBLAS_DTYPE::BF16, IsAsym
    );
    stor.assign((int8_t*)PackedBuf);
    ORTThreading orth(ThreadPool);
    JblasKernel.mProB.packNbitsWeight(N, K, IsAsym, QData, ldb, Scale, Zp, &stor, &orth);
    if (lastCall) {
        JblasKernel.mProB.reduceWeight(&stor, &orth);
    }
}

bool
JblasQ4GemmPackB(
    void* PackedBuf,
    const uint8_t* QData,
    const float* Scale,
    const uint8_t* Zp,
    size_t N,
    size_t K,
    size_t ldb,
    size_t BlkSize,
    bool isAsym,
    bool lastCall,
    MLAS_COMPUTE_TYPE CompType,
    MLAS_THREADPOOL* ThreadPool
)
{
    GetCPUDevice();
    switch (CompType) {
        case CompInt8:
            if (_cd->AMX_INT8() && BlkSize % tAMX_INT8_SS::KTILE == 0) {
                JblaNBitsGemmPackB<tLauncher_Int8_S4_F32F32<tAMX_INT8_SS>>(
                    PackedBuf, int(BlkSize), QData, Scale, Zp, int(N), int(K), isAsym, lastCall, int(ldb), ThreadPool
                );
                return true;
            }
            if (_cd->AVX512_VNNI() && BlkSize % tAVX512_VNNI::KTILE == 0) {
                JblaNBitsGemmPackB<tLauncher_Int8_S4_F32F32<tAVX512_VNNI>>(
                    PackedBuf, int(BlkSize), QData, Scale, Zp, int(N), int(K), isAsym, lastCall, int(ldb), ThreadPool
                );
                return true;
            }
            if (_cd->AVX_VNNI() && BlkSize % tAVX_VNNI::KTILE == 0) {
                JblaNBitsGemmPackB<tLauncher_Int8_S4_F32F32<tAVX_VNNI>>(
                    PackedBuf, int(BlkSize), QData, Scale, Zp, int(N), int(K), isAsym, lastCall, int(ldb), ThreadPool
                );
                return true;
            }
        case CompBf16:
        case CompFp16:
        case CompFp32:
        case CompUndef:
            if (_cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
                JblaNBitsGemmPackB<tLauncher_Fp32_S4_F32F32<tAVX512F>>(
                    PackedBuf, int(BlkSize), QData, Scale, Zp, int(N), int(K), isAsym, lastCall, int(ldb), ThreadPool
                );
                return true;
            }
            if (_cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
                JblaNBitsGemmPackB<tLauncher_Fp32_S4_F32F32<tAVX2>>(
                    PackedBuf, int(BlkSize), QData, Scale, Zp, int(N), int(K), isAsym, lastCall, int(ldb), ThreadPool
                );
                return true;
            }
        default:
            return false;
    }
    return false;
}

bool
JblasQ4GemmUnPackB(float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb, MLAS_THREADPOOL* ThreadPool)
{
    auto ptr = jblas::storage::gemm::PackedWeightParser::deserialBuffer(const_cast<void*>(PackedBuf));
    ORTThreading orth(ThreadPool);
    GetCPUDevice();
    if (ptr) {
        if (ptr->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockS4) {
            auto NTile = jblas::gemm::CoreAttr::get_mask_val(
                ptr->mCoreId, jblas::gemm::CoreAttr::NTILE_MASK, jblas::gemm::CoreAttr::NTILE_SHIFT
            );
            auto CType = jblas::gemm::CoreAttr::get_mask_val(
                ptr->mCoreId, jblas::gemm::CoreAttr::COMP_MASK, jblas::gemm::CoreAttr::COMP_SHIFT
            );
            if (CType == uint32_t(jblas::gemm::CompType::COMP_FP32)) {
                if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
                    static jblas::prologue_b::gemm::WeightKBlockS4<tAVX512F, tAVX512F::ISA> proB;
                    proB.unpackWeight(int(N), int(K), ptr, FpData, int(ldb), &orth);
                    goto __END;
                }
                if (NTile == tAVX2::NTILE && _cd->AVX2()) {
                    static jblas::prologue_b::gemm::WeightKBlockS4<tAVX2, tAVX2::ISA> proB;
                    proB.unpackWeight(int(N), int(K), ptr, FpData, int(ldb), &orth);
                    goto __END;
                }
            }
            if (CType == uint32_t(jblas::gemm::CompType::COMP_INT8_US_INT32)) {
                if (NTile == tAMX_INT8_US::NTILE && _cd->AMX_INT8()) {
                    static jblas::prologue_b::gemm::WeightKBlockS4<tAMX_INT8_US, tAMX_INT8_US::ISA> proB;
                    proB.unpackWeight(int(N), int(K), ptr, FpData, int(ldb), &orth);
                    goto __END;
                }
                if (NTile == tAVX512_VNNI::NTILE && _cd->AVX512_VNNI()) {
                    static jblas::prologue_b::gemm::WeightKBlockS4<tAVX512_VNNI, tAVX512_VNNI::ISA> proB;
                    proB.unpackWeight(int(N), int(K), ptr, FpData, int(ldb), &orth);
                    goto __END;
                }
                if (NTile == tAVX_VNNI::NTILE && _cd->AVX_VNNI()) {
                    static jblas::prologue_b::gemm::WeightKBlockS4<tAVX_VNNI, tAVX_VNNI::ISA> proB;
                    proB.unpackWeight(int(N), int(K), ptr, FpData, int(ldb), &orth);
                    goto __END;
                }
            }
            if (CType == uint32_t(jblas::gemm::CompType::COMP_INT8_SS_INT32)) {
                if (NTile == tAMX_INT8_SS::NTILE && _cd->AMX_INT8()) {
                    static jblas::prologue_b::gemm::WeightKBlockS4<tAMX_INT8_SS, tAMX_INT8_SS::ISA> proB;
                    proB.unpackWeight(int(N), int(K), ptr, FpData, int(ldb), &orth);
                    goto __END;
                }
            }
        }
    __END:
        delete ptr;
        return true;
    }
    return false;
}
