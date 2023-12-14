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
    : IThreading(MLAS_THREADPOOL::DegreeOfParallelism(reinterpret_cast<MLAS_THREADPOOL*>(tp))), mTp(tp)
{
}

void
jblas::ORTThreading::parallel_for(const jblas::parallel::thread_func& func)
{
    MlasTrySimpleParallel(reinterpret_cast<MLAS_THREADPOOL*>(mTp), mThreadNum, [&](ptrdiff_t tid) {
        func(static_cast<int>(tid));
    });
}

template <class GemmCore_T>
static void
JblasSQ4GemmCompF32(
    const size_t M,
    const size_t N,
    const size_t K,
    const float* A,
    const size_t lda,
    jblas::storage::gemm::StorageWeightKBlockS4* B,
    float* C,
    const size_t ldc,
    int8_t* WorkSpace,
    jblas::parallel::IThreading* th
)
{
    auto M_ = static_cast<int>(M);
    auto N_ = static_cast<int>(N);
    auto K_ = static_cast<int>(K);
    auto lda_ = static_cast<int>(lda);
    auto ldc_ = static_cast<int>(ldc);
    if (M <= 32) {
        using Parallel = jblas::parallel::gemm::SchedulerKBlock<GemmCore_T>;
        using Launcher = tLauncher_Fp32_S4_F32F32<GemmCore_T>;
        static Launcher kernel;
        auto reduceA = kernel.mProA.createStorage(M_, K_, B->mBlockSize);
        if (B->mIsAsym) {
            reduceA.assign(WorkSpace);
            ORTThreading single(nullptr);
            kernel.mProA.reduce({A, lda_}, &reduceA, M_, K_, &single);
        }
        typename Launcher::BEpiParam blkargs{
            B->template SPtr<int8_t>(),    B->mScaT,   B->mCStep, B->template ZPtr<int8_t>(),
            reduceA.template get<float>(), reduceA.lda};

        typename Launcher::Param args{M_, N_, K_, B->mBlockSize, {A, lda_}, {B}, blkargs, {C, ldc_}};
        jblas::parallel::GemmKBlockRun<Parallel>(kernel, args, th);
    } else {
        using Parallel = jblas::parallel::gemm::SchedulerBase<GemmCore_T>;
        using Launcher = jblas::wrapper::gemm::LauncherBase<
            GemmCore_T::ISA, GemmCore_T, jblas::prologue_a::gemm::ActivationBase,
            jblas::prologue_b::gemm::WeightKBlockS4, jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
        static Launcher kernel;

        typename Launcher::Param args{M_, N_, K_, {A, lda_}, {B}, {C, ldc_}};
        jblas::parallel::GemmBaseRun<Parallel>(kernel, args, th);
    }
}

template <class GemmCore_T>
static void
JblasSQ4GemmCompInt8(
    const size_t M,
    const size_t N,
    const size_t K,
    const float* A,
    const size_t lda,
    jblas::storage::gemm::StorageWeightKBlockS4* B,
    float* C,
    const size_t ldc,
    int8_t* WorkSpace,
    jblas::parallel::IThreading* th
)
{
    using Parallel = jblas::parallel::gemm::SchedulerKBlock<GemmCore_T>;
    using Launcher = tLauncher_Int8_S4_F32F32<GemmCore_T>;
    auto M_ = static_cast<int>(M);
    auto N_ = static_cast<int>(N);
    auto K_ = static_cast<int>(K);
    auto lda_ = static_cast<int>(lda);
    auto ldc_ = static_cast<int>(ldc);
    static Launcher kernel;
    auto quanA = kernel.mProA.createStorage(M_, K_, B->mBlockSize, B->mIsAsym);
    quanA.assign(WorkSpace);
    if (M <= 32) {
        ORTThreading single(nullptr);
        kernel.mProA.quantize({A, lda_, &quanA}, M_, K_, &single);
    } else {
        kernel.mProA.quantize({A, lda_, &quanA}, M_, K_, th);
    }
    typename Launcher::Param args{
        M_,
        N_,
        K_,
        B->mBlockSize,
        {A, lda_, &quanA},
        {B},
        {B->template SPtr<int8_t>(), B->mScaT, B->mCStep, quanA.template SPtr<float>(), quanA.mCStep,
         quanA.template ZPtr<uint8_t>(), B->template RPtr<float>(), B->mRedT, B->template ZPtr<int8_t>(),
         quanA.template RPtr<float>(), B->mBlockSize},
        {C, ldc_}};
    jblas::parallel::GemmKBlockRun<Parallel>(kernel, args, th);
}

bool
JblasSQ4GemmBatchDriver(
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const MLAS_SQNBITS_GEMM_DATA_PACKED_PARAMS* DataParams,
    int8_t* WorkSpace,
    MLAS_THREADPOOL* ThreadPool
)
{
    GetCPUDevice();
    ORTThreading orth(ThreadPool);
    bool processed = true;
    for (size_t i = 0; i < BatchN; i++) {
        auto ptr = jblas::storage::gemm::PackedWeightParser::deserialBuffer(const_cast<void*>(DataParams[i].B));
        auto uptr = std::unique_ptr<jblas::storage::gemm::WeightBase>(ptr);
        if (ptr) {
            if (ptr->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockS4) {
                auto kptr = reinterpret_cast<jblas::storage::gemm::StorageWeightKBlockS4*>(ptr);
                auto coretype = ptr->mCoreId;
                auto NTile = jblas::gemm::CoreAttr::get_mask_val(
                    ptr->mCoreId, jblas::gemm::CoreAttr::NTILE_MASK, jblas::gemm::CoreAttr::NTILE_SHIFT
                );
                auto CType = jblas::gemm::CoreAttr::get_mask_val(
                    ptr->mCoreId, jblas::gemm::CoreAttr::COMP_MASK, jblas::gemm::CoreAttr::COMP_SHIFT
                );
                if (CType == uint32_t(gemm::CompType::COMP_FP32)) {
                    if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
                        JblasSQ4GemmCompF32<tAVX512F>(
                            M, N, K, DataParams[i].A, DataParams[i].lda, kptr, DataParams[i].C, DataParams[i].ldc,
                            WorkSpace, &orth
                        );
                    } else if (NTile == tAVX2::NTILE && _cd->AVX2()) {
                        JblasSQ4GemmCompF32<tAVX2>(
                            M, N, K, DataParams[i].A, DataParams[i].lda, kptr, DataParams[i].C, DataParams[i].ldc,
                            WorkSpace, &orth
                        );
                    }
                }
                if (CType == uint32_t(gemm::CompType::COMP_INT8_US_INT32)) {
                    if (NTile == tAMX_INT8_US::NTILE && _cd->AMX_INT8()) {
                        JblasSQ4GemmCompInt8<tAMX_INT8_US>(
                            M, N, K, DataParams[i].A, DataParams[i].lda, kptr, DataParams[i].C, DataParams[i].ldc,
                            WorkSpace, &orth
                        );
                    } else if (NTile == tAVX512_VNNI::NTILE && _cd->AVX512_VNNI()) {
                        JblasSQ4GemmCompInt8<tAVX512_VNNI>(
                            M, N, K, DataParams[i].A, DataParams[i].lda, kptr, DataParams[i].C, DataParams[i].ldc,
                            WorkSpace, &orth
                        );
                    } else if (NTile == tAVX_VNNI::NTILE && _cd->AVX_VNNI()) {
                        JblasSQ4GemmCompInt8<tAVX_VNNI>(
                            M, N, K, DataParams[i].A, DataParams[i].lda, kptr, DataParams[i].C, DataParams[i].ldc,
                            WorkSpace, &orth
                        );
                    }
                }
                if (CType == uint32_t(gemm::CompType::COMP_INT8_SS_INT32)) {
                    if (NTile == tAMX_INT8_SS::NTILE && _cd->AMX_INT8()) {
                        JblasSQ4GemmCompInt8<tAMX_INT8_SS>(
                            M, N, K, DataParams[i].A, DataParams[i].lda, kptr, DataParams[i].C, DataParams[i].ldc,
                            WorkSpace, &orth
                        );
                    }
                }
            }
        } else {
            processed = false;
            break;
        }
    }
    return processed;
}

template <class GemmCore_T>
static size_t
JblasSQ4GemmCompF32WorkspaceSize(
    const size_t M,
    const size_t N,
    const size_t K,
    const float* A,
    const size_t lda,
    jblas::storage::gemm::StorageWeightKBlockS4* B,
    float* C,
    const size_t ldc
)
{
    auto M_ = static_cast<int>(M);
    auto K_ = static_cast<int>(K);
    (void)(N);
    (void)(lda);
    (void)(ldc);
    if (M <= 32) {
        using Launcher = tLauncher_Fp32_S4_F32F32<GemmCore_T>;
        static Launcher kernel;
        if (B->mIsAsym) {
            auto reduceA = kernel.mProA.createStorage(M_, K_, B->mBlockSize);
            return reduceA.mSize;
        }
        return 0;
    } else {
        using Launcher = jblas::wrapper::gemm::LauncherBase<
            GemmCore_T::ISA, GemmCore_T, jblas::prologue_a::gemm::ActivationBase,
            jblas::prologue_b::gemm::WeightKBlockS4, jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
        static Launcher kernel;
        return 0;
    }
    return 0;
}

template <class GemmCore_T>
static size_t
JblasSQ4GemmCompInt8WorkspaceSize(
    const size_t M,
    const size_t N,
    const size_t K,
    const float* A,
    const size_t lda,
    jblas::storage::gemm::StorageWeightKBlockS4* B,
    float* C,
    const size_t ldc
)
{
    using Parallel = jblas::parallel::gemm::SchedulerKBlock<GemmCore_T>;
    using Launcher = tLauncher_Int8_S4_F32F32<GemmCore_T>;
    static Launcher kernel;
    (void)(N);
    (void)(lda);
    (void)(ldc);
    auto quanA = kernel.mProA.createStorage(
        static_cast<int>(M), static_cast<int>(K), static_cast<int>(B->mBlockSize), B->mIsAsym
    );
    return quanA.mSize;
}

size_t
JblasSQ4GemmBatchWorkspaceSize(
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const MLAS_SQNBITS_GEMM_DATA_PACKED_PARAMS* DataParams
)
{
    GetCPUDevice();
    size_t size = 0;
    for (size_t i = 0; i < BatchN; i++) {
        auto ptr = jblas::storage::gemm::PackedWeightParser::deserialBuffer(const_cast<void*>(DataParams[i].B));
        auto uptr = std::unique_ptr<jblas::storage::gemm::WeightBase>(ptr);
        if (ptr) {
            if (ptr->mPrologueID == JBLAS_PROLOGUEB_IDS::WeightKBlockS4) {
                auto kptr = reinterpret_cast<jblas::storage::gemm::StorageWeightKBlockS4*>(ptr);
                auto coretype = ptr->mCoreId;
                auto NTile = jblas::gemm::CoreAttr::get_mask_val(
                    ptr->mCoreId, jblas::gemm::CoreAttr::NTILE_MASK, jblas::gemm::CoreAttr::NTILE_SHIFT
                );
                auto CType = jblas::gemm::CoreAttr::get_mask_val(
                    ptr->mCoreId, jblas::gemm::CoreAttr::COMP_MASK, jblas::gemm::CoreAttr::COMP_SHIFT
                );
                if (CType == uint32_t(gemm::CompType::COMP_FP32)) {
                    if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
                        size = std::max(
                            JblasSQ4GemmCompF32WorkspaceSize<tAVX512F>(
                                M, N, K, DataParams[i].A, DataParams[i].lda, kptr, DataParams[i].C, DataParams[i].ldc
                            ),
                            size
                        );
                    } else if (NTile == tAVX2::NTILE && _cd->AVX2()) {
                        size = std::max(
                            JblasSQ4GemmCompF32WorkspaceSize<tAVX2>(
                                M, N, K, DataParams[i].A, DataParams[i].lda, kptr, DataParams[i].C, DataParams[i].ldc
                            ),
                            size
                        );
                    }
                }
                if (CType == uint32_t(gemm::CompType::COMP_INT8_US_INT32)) {
                    if (NTile == tAMX_INT8_US::NTILE && _cd->AMX_INT8()) {
                        size = std::max(
                            JblasSQ4GemmCompInt8WorkspaceSize<tAMX_INT8_US>(
                                M, N, K, DataParams[i].A, DataParams[i].lda, kptr, DataParams[i].C, DataParams[i].ldc
                            ),
                            size
                        );
                    } else if (NTile == tAVX512_VNNI::NTILE && _cd->AVX512_VNNI()) {
                        size = std::max(
                            JblasSQ4GemmCompInt8WorkspaceSize<tAVX512_VNNI>(
                                M, N, K, DataParams[i].A, DataParams[i].lda, kptr, DataParams[i].C, DataParams[i].ldc
                            ),
                            size
                        );
                    } else if (NTile == tAVX_VNNI::NTILE && _cd->AVX_VNNI()) {
                        size = std::max(
                            JblasSQ4GemmCompInt8WorkspaceSize<tAVX_VNNI>(
                                M, N, K, DataParams[i].A, DataParams[i].lda, kptr, DataParams[i].C, DataParams[i].ldc
                            ),
                            size
                        );
                    }
                }
                if (CType == uint32_t(gemm::CompType::COMP_INT8_SS_INT32)) {
                    if (NTile == tAMX_INT8_SS::NTILE && _cd->AMX_INT8()) {
                        size = std::max(
                            JblasSQ4GemmCompInt8WorkspaceSize<tAMX_INT8_SS>(
                                M, N, K, DataParams[i].A, DataParams[i].lda, kptr, DataParams[i].C, DataParams[i].ldc
                            ),
                            size
                        );
                    }
                }
            }
        }
    }
    return size;
}

template <typename T>
static size_t
JblasQ4BuSize(size_t block_size, size_t N, size_t K, bool isAsym)
{
    static T launcher;
    auto stor = launcher.mProB.createStorage(
        static_cast<int>(N), static_cast<int>(K), static_cast<int>(block_size), JBLAS_DTYPE::S4_CLIP, JBLAS_DTYPE::F32,
        JBLAS_DTYPE::BF16, isAsym
    );
    // TODO(Yu) support more scale dtype
    return stor.mSize;
}

size_t
JblasQ4GemmPackBSize(size_t N, size_t K, size_t BlkSize, bool isAsym, MLAS_SQNBIT_COMPUTE_TYPE CompType)
{
    GetCPUDevice();
    if (K % BlkSize != 0) {
        return 0;
    }
    // from low precision to high precision
    switch (CompType) {
        case CompInt8:
            if (_cd->AMX_INT8() && BlkSize % tAMX_INT8_SS::KTILE == 0) {
                return JblasQ4BuSize<tLauncher_Int8_S4_F32F32<tAMX_INT8_SS>>(BlkSize, N, K, isAsym);
            }
            if (_cd->AVX512_VNNI() && BlkSize % tAVX512_VNNI::KTILE == 0) {
                return JblasQ4BuSize<tLauncher_Int8_S4_F32F32<tAVX512_VNNI>>(BlkSize, N, K, isAsym);
            }
            if (_cd->AVX_VNNI() && BlkSize % tAVX_VNNI::KTILE == 0) {
                return JblasQ4BuSize<tLauncher_Int8_S4_F32F32<tAVX_VNNI>>(BlkSize, N, K, isAsym);
            }
        case CompBf16:
        case CompFp16:
        case CompFp32:
        case CompUndef:
            if (_cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
                return JblasQ4BuSize<tLauncher_Int8_S4_F32F32<tAVX512F>>(BlkSize, N, K, isAsym);
            }
            if (_cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
                return JblasQ4BuSize<tLauncher_Int8_S4_F32F32<tAVX2>>(BlkSize, N, K, isAsym);
            }
            break;
        default:
            return 0;
    }
    return 0;
}

template <typename T>
static void
JblasQ4GemmPackBImpl(
    void* PackedBuf,
    size_t BlkSize,
    const uint8_t* QData,
    const float* Scale,
    const uint8_t* Zp,
    size_t N,
    size_t K,
    bool IsAsym,
    bool lastCall,
    size_t ldb,
    MLAS_THREADPOOL* ThreadPool
)
{
    static T JblasKernel;
    auto N_ = static_cast<int>(N);
    auto K_ = static_cast<int>(K);
    auto stor = JblasKernel.mProB.createStorage(
        N_, K_, static_cast<int>(BlkSize), JBLAS_DTYPE::S4_CLIP, JBLAS_DTYPE::F32, JBLAS_DTYPE::BF16, IsAsym
    );
    stor.assign(reinterpret_cast<int8_t*>(PackedBuf));
    ORTThreading orth(ThreadPool);
    JblasKernel.mProB.packNbitsWeight(N_, K_, IsAsym, QData, static_cast<int>(ldb), Scale, Zp, &stor, &orth);
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
    MLAS_SQNBIT_COMPUTE_TYPE CompType,
    MLAS_THREADPOOL* ThreadPool
)
{
    GetCPUDevice();
    // explicit statement fall through.
    switch (CompType) {
        case CompInt8:
            if (_cd->AMX_INT8() && BlkSize % tAMX_INT8_SS::KTILE == 0) {
                JblasQ4GemmPackBImpl<tLauncher_Int8_S4_F32F32<tAMX_INT8_SS>>(
                    PackedBuf, BlkSize, QData, Scale, Zp, N, K, isAsym, lastCall, ldb, ThreadPool
                );
                return true;
            }
            if (_cd->AVX512_VNNI() && BlkSize % tAVX512_VNNI::KTILE == 0) {
                JblasQ4GemmPackBImpl<tLauncher_Int8_S4_F32F32<tAVX512_VNNI>>(
                    PackedBuf, BlkSize, QData, Scale, Zp, N, K, isAsym, lastCall, ldb, ThreadPool
                );
                return true;
            }
            if (_cd->AVX_VNNI() && BlkSize % tAVX_VNNI::KTILE == 0) {
                JblasQ4GemmPackBImpl<tLauncher_Int8_S4_F32F32<tAVX_VNNI>>(
                    PackedBuf, BlkSize, QData, Scale, Zp, N, K, isAsym, lastCall, ldb, ThreadPool
                );
                return true;
            }
        case CompBf16:
        case CompFp16:
        case CompFp32:
        case CompUndef:
            if (_cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
                JblasQ4GemmPackBImpl<tLauncher_Fp32_S4_F32F32<tAVX512F>>(
                    PackedBuf, BlkSize, QData, Scale, Zp, N, K, isAsym, lastCall, ldb, ThreadPool
                );
                return true;
            }
            if (_cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
                JblasQ4GemmPackBImpl<tLauncher_Fp32_S4_F32F32<tAVX2>>(
                    PackedBuf, BlkSize, QData, Scale, Zp, N, K, isAsym, lastCall, ldb, ThreadPool
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
    auto uptr = std::unique_ptr<jblas::storage::gemm::WeightBase>(ptr);
    ORTThreading orth(ThreadPool);
    auto N_ = static_cast<int>(N);
    auto K_ = static_cast<int>(K);
    auto ldb_ = static_cast<int>(ldb);
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
                    proB.unpackWeight(N_, K_, ptr, FpData, ldb_, &orth);
                } else if (NTile == tAVX2::NTILE && _cd->AVX2()) {
                    static jblas::prologue_b::gemm::WeightKBlockS4<tAVX2, tAVX2::ISA> proB;
                    proB.unpackWeight(N_, K_, ptr, FpData, ldb_, &orth);
                }
            }
            if (CType == uint32_t(jblas::gemm::CompType::COMP_INT8_US_INT32)) {
                if (NTile == tAMX_INT8_US::NTILE && _cd->AMX_INT8()) {
                    static jblas::prologue_b::gemm::WeightKBlockS4<tAMX_INT8_US, tAMX_INT8_US::ISA> proB;
                    proB.unpackWeight(N_, K_, ptr, FpData, ldb_, &orth);
                } else if (NTile == tAVX512_VNNI::NTILE && _cd->AVX512_VNNI()) {
                    static jblas::prologue_b::gemm::WeightKBlockS4<tAVX512_VNNI, tAVX512_VNNI::ISA> proB;
                    proB.unpackWeight(N_, K_, ptr, FpData, ldb_, &orth);
                } else if (NTile == tAVX_VNNI::NTILE && _cd->AVX_VNNI()) {
                    static jblas::prologue_b::gemm::WeightKBlockS4<tAVX_VNNI, tAVX_VNNI::ISA> proB;
                    proB.unpackWeight(N_, K_, ptr, FpData, ldb_, &orth);
                }
            }
            if (CType == uint32_t(jblas::gemm::CompType::COMP_INT8_SS_INT32)) {
                if (NTile == tAMX_INT8_SS::NTILE && _cd->AMX_INT8()) {
                    static jblas::prologue_b::gemm::WeightKBlockS4<tAMX_INT8_SS, tAMX_INT8_SS::ISA> proB;
                    proB.unpackWeight(N_, K_, ptr, FpData, ldb_, &orth);
                }
            }
        }
        return true;
    }
    return false;
}
