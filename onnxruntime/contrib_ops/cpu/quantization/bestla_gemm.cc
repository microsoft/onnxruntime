/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    bestla_gemm.cpp

Abstract:

    Currently only support Q4 gemm.
--*/

#include "contrib_ops/cpu/quantization/bestla_defs.h"
#include "contrib_ops/cpu/quantization/bestla_gemm.h"
#include "core/platform/threadpool.h"

using ThreadPool = onnxruntime::concurrency::ThreadPool;

namespace bestla {

ORTThreading::ORTThreading(void* tp)
    : IThreading(ThreadPool::DegreeOfParallelism(reinterpret_cast<ThreadPool*>(tp))), mTp(tp) {}

void ORTThreading::parallel_for(const parallel::thread_func& func) const {
  ThreadPool::TrySimpleParallelFor(reinterpret_cast<ThreadPool*>(mTp), mThreadNum,
                                   [&](ptrdiff_t tid) { func(static_cast<int>(tid)); });
}

Platform::Platform() {
  GetCPUDevice();
  if (_cd->AMX_INT8() || _cd->AMX_BF16()) {
    utils::request_perm_xtile_data();
  }
}

Platform* Platform::get() {
  static Platform instance;
  return &instance;
}

template <class GemmCore_T>
static void NSSQ4GemmCompF32(size_t M, size_t N, size_t K, const float* A, size_t lda,
                             storage::gemm::StorageWeightKBlockNInteger* B, float* C, size_t ldc, int8_t* WorkSpace,
                             parallel::IThreading* th) {
  auto M_ = static_cast<int>(M);
  auto N_ = static_cast<int>(N);
  auto K_ = static_cast<int>(K);
  auto lda_ = static_cast<int>(lda);
  auto ldc_ = static_cast<int>(ldc);
  utils::GemmProblem gp(1, M_, N_, K_, B->mBlockSize);
  if (M <= 16) {
    using Parallel = parallel::gemm::SchedulerKBlock<GemmCore_T>;
    using Launcher =
        wrapper::gemm::LauncherKBlock<GemmCore_T::ISA, GemmCore_T, prologue_a::gemm::ActivationKBlockBaseF32,
                                      prologue_b::gemm::WeightKBlockNInteger, epilogue::gemm::CompFp32BlockEpilogue,
                                      epilogue::gemm::AccumulatorWriteBackFp32>;
    static Launcher kernel;
    auto reduceA = kernel.mProA.createStorage(M_, K_, B->mBlockSize);
    if (B->IsAsym()) {
      reduceA.assign(WorkSpace);
      ORTThreading single(nullptr);
      kernel.mProA.reduce({A, lda_, &reduceA}, M_, K_, B->mBlockSize, &single);
    }
    typename Launcher::Param args{gp,
                                  {A, lda_, &reduceA},
                                  {B},
                                  {B->template SPtr<int8_t>(), B->SDtype(), B->CStep(), B->template ZPtr<int8_t>(),
                                   reduceA.template RPtr<float>(), reduceA.lda},
                                  {C, ldc_, nullptr}};
    parallel::GemmRun<Parallel>(kernel, args, th);
  } else {
    using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;
    using Launcher =
        wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T, prologue_a::gemm::ActivationBase,
                                    prologue_b::gemm::WeightKBlockNInteger, epilogue::gemm::AccumulatorWriteBackFp32>;
    static Launcher kernel;
    typename Launcher::Param args{gp, {A, lda_}, {B}, {C, ldc_, nullptr}};
    parallel::GemmRun<Parallel>(kernel, args, th);
  }
}

template <class GemmCore_T>
static void NSSQ4GemmCompInt8(size_t M, size_t N, size_t K, const float* A, size_t lda,
                              storage::gemm::StorageWeightKBlockNInteger* B, float* C, size_t ldc, int8_t* WorkSpace,
                              parallel::IThreading* th) {
  using Parallel = parallel::gemm::SchedulerKBlockS<GemmCore_T>;
  using Launcher =
      wrapper::gemm::LauncherIntKBlock<GemmCore_T::ISA, GemmCore_T, prologue_a::gemm::ActivationF32KBlockQuantize,
                                       prologue_b::gemm::WeightKBlockNInteger,
                                       epilogue::gemm::AccumulatorWriteBackFp32>;
  auto M_ = static_cast<int>(M);
  auto N_ = static_cast<int>(N);
  auto K_ = static_cast<int>(K);
  auto lda_ = static_cast<int>(lda);
  auto ldc_ = static_cast<int>(ldc);
  static Launcher kernel;
  auto quanA = kernel.mProA.createStorage(M_, K_, B->mBlockSize, B->IsAsym());
  quanA.assign(WorkSpace);
  if (M <= 16) {
    ORTThreading single(nullptr);
    kernel.mProA.quantize({A, lda_, &quanA}, M_, K_, &single);
  } else {
    kernel.mProA.quantize({A, lda_, &quanA}, M_, K_, th);
  }
  utils::GemmProblem gp(1, M_, N_, K_, B->mBlockSize);
  typename Launcher::Param args{gp, {A, lda_, &quanA}, {B}, {C, ldc_, nullptr}};
  parallel::GemmRun<Parallel>(kernel, args, th);
}

template <class GemmCore_T>
static size_t NSSQ4GemmCompF32WorkspaceSize(size_t M, size_t N, size_t K, const float* A, size_t lda,
                                            storage::gemm::StorageWeightKBlockNInteger* B, float* C, size_t ldc) {
  auto M_ = static_cast<int>(M);
  auto K_ = static_cast<int>(K);
  (void)(A);
  (void)(N);
  (void)(C);
  (void)(lda);
  (void)(ldc);
  if (M <= 16) {
    using ProA = prologue_a::gemm::ActivationKBlockBaseF32<GemmCore_T, GemmCore_T::ISA>;
    static ProA proA;
    if (B->IsAsym()) {
      auto reduceA = proA.createStorage(M_, K_, B->mBlockSize);
      return reduceA.mSize;
    }
    return 0;
  } else {
    // using ProA = prologue_a::gemm::ActivationBase<GemmCore_T, GemmCore_T::ISA>;
    return 0;
  }
  return 0;
}

template <class GemmCore_T>
static size_t NSSQ4GemmCompInt8WorkspaceSize(size_t M, size_t N, size_t K, const float* A, size_t lda,
                                             storage::gemm::StorageWeightKBlockNInteger* B, float* C, size_t ldc) {
  (void)(N);
  (void)(lda);
  (void)(ldc);
  (void)(A);
  (void)(C);
  using ProA = prologue_a::gemm::ActivationF32KBlockQuantize<GemmCore_T, GemmCore_T::ISA>;
  static ProA proA;
  auto quanA =
      proA.createStorage(static_cast<int>(M), static_cast<int>(K), static_cast<int>(B->mBlockSize), B->IsAsym());
  return quanA.mSize;
}

}  // namespace bestla

using namespace bestla;

static bool NSSQ4GemmBatchDriver(size_t M, size_t N, size_t K, size_t BatchN,
                                 const NS_SQNBITS_GEMM_DATA_PACKED_PARAMS* DataParams, int8_t* WorkSpace,
                                 void* ThreadPool) {
  GetCPUDevice();
  bestla::ORTThreading orth(ThreadPool);
  bool processed = true;
  for (size_t i = 0; i < BatchN; i++) {
    auto ptr = bestla::storage::gemm::PackedWeightParser::deserialBuffer(DataParams[i].B);
    auto uptr = std::unique_ptr<bestla::storage::gemm::IWeightBase>(ptr);
    if (ptr) {
      auto NTile = gemm::CoreAttr::get_mask_val(ptr->mCoreId, gemm::CoreAttr::NTILE_MASK, gemm::CoreAttr::NTILE_SHIFT);
      auto PackRow = gemm::CoreAttr::get_packrow(ptr->mCoreId);
      auto CType = gemm::CoreAttr::get_comp(ptr->mCoreId);
      auto btype = static_cast<gemm::CompType>(gemm::CompTypeHelper::get_B(CType));
      if (ptr->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNInteger) {
        auto kptr = reinterpret_cast<bestla::storage::gemm::StorageWeightKBlockNInteger*>(ptr);
        if (btype == gemm::CompType::tFP32 && PackRow == 1) {
          if (NTile == bestla::tAVX512F::NTILE && _cd->AVX512F()) {
            bestla::NSSQ4GemmCompF32<bestla::tAVX512F>(M, N, K, DataParams[i].A, DataParams[i].lda, kptr,
                                                       DataParams[i].C, DataParams[i].ldc, WorkSpace, &orth);
          } else if (NTile == bestla::tAVX2::NTILE && _cd->AVX2()) {
            bestla::NSSQ4GemmCompF32<bestla::tAVX2>(M, N, K, DataParams[i].A, DataParams[i].lda, kptr, DataParams[i].C,
                                                    DataParams[i].ldc, WorkSpace, &orth);
          }
        }
        if (btype == gemm::CompType::tS8 && PackRow == 4) {
          if (NTile == bestla::tAMX_INT8_SS_KBlock::NTILE && _cd->AMX_INT8()) {
            bestla::NSSQ4GemmCompInt8<bestla::tAMX_INT8_SS_KBlock>(M, N, K, DataParams[i].A, DataParams[i].lda, kptr,
                                                                   DataParams[i].C, DataParams[i].ldc, WorkSpace,
                                                                   &orth);
          } else if (NTile == bestla::tAVX512_VNNI_KBlock::NTILE && _cd->AVX512_VNNI()) {
            bestla::NSSQ4GemmCompInt8<bestla::tAVX512_VNNI_KBlock>(M, N, K, DataParams[i].A, DataParams[i].lda, kptr,
                                                                   DataParams[i].C, DataParams[i].ldc, WorkSpace,
                                                                   &orth);
          } else if (NTile == bestla::tAVX_VNNI_KBlock::NTILE && _cd->AVX_VNNI()) {
            bestla::NSSQ4GemmCompInt8<bestla::tAVX_VNNI_KBlock>(M, N, K, DataParams[i].A, DataParams[i].lda, kptr,
                                                                DataParams[i].C, DataParams[i].ldc, WorkSpace, &orth);
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

static size_t NSSQ4GemmBatchWorkspaceSize(size_t M, size_t N, size_t K, size_t BatchN,
                                          const NS_SQNBITS_GEMM_DATA_PACKED_PARAMS* DataParams) {
  GetCPUDevice();
  size_t size = 0;
  for (size_t i = 0; i < BatchN; i++) {
    auto ptr = storage::gemm::PackedWeightParser::deserialBuffer(DataParams[i].B);
    auto uptr = std::unique_ptr<storage::gemm::IWeightBase>(ptr);
    if (ptr) {
      if (ptr->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNInteger) {
        auto kptr = reinterpret_cast<storage::gemm::StorageWeightKBlockNInteger*>(ptr);
        auto NTile =
            gemm::CoreAttr::get_mask_val(ptr->mCoreId, gemm::CoreAttr::NTILE_MASK, gemm::CoreAttr::NTILE_SHIFT);
        auto PackRow = gemm::CoreAttr::get_packrow(ptr->mCoreId);
        auto CType = gemm::CoreAttr::get_comp(ptr->mCoreId);
        auto btype = static_cast<gemm::CompType>(gemm::CompTypeHelper::get_B(CType));
        if (btype == gemm::CompType::tFP32 && PackRow == 1) {
          if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
            size = std::max(NSSQ4GemmCompF32WorkspaceSize<tAVX512F>(M, N, K, DataParams[i].A, DataParams[i].lda, kptr,
                                                                    DataParams[i].C, DataParams[i].ldc),
                            size);
          } else if (NTile == tAVX2::NTILE && _cd->AVX2()) {
            size = std::max(NSSQ4GemmCompF32WorkspaceSize<tAVX2>(M, N, K, DataParams[i].A, DataParams[i].lda, kptr,
                                                                 DataParams[i].C, DataParams[i].ldc),
                            size);
          }
        }
        if (btype == gemm::CompType::tS8 && PackRow == 4) {
          if (NTile == tAMX_INT8_SS_KBlock::NTILE && _cd->AMX_INT8()) {
            size = std::max(NSSQ4GemmCompInt8WorkspaceSize<tAMX_INT8_SS_KBlock>(
                                M, N, K, DataParams[i].A, DataParams[i].lda, kptr, DataParams[i].C, DataParams[i].ldc),
                            size);
          } else if (NTile == tAVX512_VNNI_KBlock::NTILE && _cd->AVX512_VNNI()) {
            size = std::max(NSSQ4GemmCompInt8WorkspaceSize<tAVX512_VNNI_KBlock>(
                                M, N, K, DataParams[i].A, DataParams[i].lda, kptr, DataParams[i].C, DataParams[i].ldc),
                            size);
          } else if (NTile == tAVX_VNNI_KBlock::NTILE && _cd->AVX_VNNI()) {
            size = std::max(NSSQ4GemmCompInt8WorkspaceSize<tAVX_VNNI_KBlock>(
                                M, N, K, DataParams[i].A, DataParams[i].lda, kptr, DataParams[i].C, DataParams[i].ldc),
                            size);
          }
        }
      }
    }
  }
  return size;
}

template <typename T>
static size_t NSQ4BuSize(size_t block_size, size_t N, size_t K, bool isAsym) {
  static T proB;
  auto stor = proB.createStorage(static_cast<int>(N), static_cast<int>(K), static_cast<int>(block_size),
                                 BTLA_DTYPE::S4_CLIP, BTLA_DTYPE::F32, BTLA_DTYPE::BF16, isAsym);
  // TODO(Yu) support more scale dtype
  return stor.mSize;
}

static bool NSQ4GemmUnPackB(float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb, void* ThreadPool) {
  auto ptr = storage::gemm::PackedWeightParser::deserialBuffer(PackedBuf);
  auto uptr = std::unique_ptr<storage::gemm::IWeightBase>(ptr);
  ORTThreading orth(ThreadPool);
  auto N_ = static_cast<int>(N);
  auto K_ = static_cast<int>(K);
  auto ldb_ = static_cast<int>(ldb);
  GetCPUDevice();
  if (ptr) {
    auto NTile = gemm::CoreAttr::get_mask_val(ptr->mCoreId, gemm::CoreAttr::NTILE_MASK, gemm::CoreAttr::NTILE_SHIFT);
    auto PackRow = gemm::CoreAttr::get_packrow(ptr->mCoreId);
    auto CType = gemm::CoreAttr::get_comp(ptr->mCoreId);
    auto btype = static_cast<gemm::CompType>(gemm::CompTypeHelper::get_B(CType));
    if (ptr->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNInteger) {
      auto wptr = reinterpret_cast<storage::gemm::StorageWeightKBlockNInteger*>(ptr);
      if (btype == gemm::CompType::tFP32 && PackRow == 1) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
          static tWeiNInt<tAVX512F, tAVX512F::ISA> proB;
          proB.unpackWeight(N_, K_, wptr, FpData, ldb_, &orth);
        } else if (NTile == tAVX2::NTILE && _cd->AVX2()) {
          static tWeiNInt<tAVX2, tAVX2::ISA> proB;
          proB.unpackWeight(N_, K_, wptr, FpData, ldb_, &orth);
        }
      }
      if (btype == gemm::CompType::tS8 && PackRow == 4) {
        if (NTile == tAMX_INT8_SS_KBlock::NTILE && _cd->AMX_INT8()) {
          static tWeiNInt<tAMX_INT8_SS_KBlock, tAMX_INT8_SS_KBlock::ISA> proB;
          proB.unpackWeight(N_, K_, wptr, FpData, ldb_, &orth);
        } else if (NTile == tAVX512_VNNI_KBlock::NTILE && _cd->AVX512_VNNI()) {
          static tWeiNInt<tAVX512_VNNI_KBlock, tAVX512_VNNI_KBlock::ISA> proB;
          proB.unpackWeight(N_, K_, wptr, FpData, ldb_, &orth);
        } else if (NTile == tAVX_VNNI_KBlock::NTILE && _cd->AVX_VNNI()) {
          static tWeiNInt<tAVX_VNNI_KBlock, tAVX_VNNI_KBlock::ISA> proB;
          proB.unpackWeight(N_, K_, wptr, FpData, ldb_, &orth);
        }
      }
    }
    return true;
  }
  return false;
}

template <typename T>
static void NSQ4GemmPackBImpl(void* PackedBuf, size_t BlkSize, const uint8_t* QData, const float* Scale,
                              const uint8_t* Zp, size_t N, size_t K, bool IsAsym, bool lastCall, size_t ldb,
                              void* ThreadPool) {
  static T proB;
  auto N_ = static_cast<int>(N);
  auto K_ = static_cast<int>(K);
  auto stor = proB.createStorage(N_, K_, static_cast<int>(BlkSize), BTLA_DTYPE::S4_CLIP, BTLA_DTYPE::F32,
                                 BTLA_DTYPE::BF16, IsAsym);
  stor.assign(reinterpret_cast<int8_t*>(PackedBuf));
  ORTThreading orth(ThreadPool);
  proB.packNbitsWeightQ4(N_, K_, IsAsym, QData, static_cast<int>(ldb), Scale, Zp, &stor, &orth);
  if (lastCall) {
    proB.reduceWeight(&stor, &orth);
  }
}

static size_t NSQ4GemmPackBSize(size_t N, size_t K, size_t BlkSize, bool isAsym, NS_SQNBIT_COMPUTE_TYPE CompType) {
  GetCPUDevice();
  if (K % BlkSize != 0) {
    return 0;
  }
  // from low precision to high precision
  switch (CompType) {
    case CompInt8:
      if (!isAsym) {  // asym int8 is not optimized, so fall through to others.
        if (_cd->AMX_INT8() && BlkSize % tAMX_INT8_SS_KBlock::KTILE == 0) {
          return NSQ4BuSize<tWeiNInt<tAMX_INT8_SS_KBlock, tAMX_INT8_SS_KBlock::ISA>>(BlkSize, N, K, isAsym);
        }
        if (_cd->AVX512_VNNI() && BlkSize % tAVX512_VNNI_KBlock::KTILE == 0) {
          return NSQ4BuSize<tWeiNInt<tAVX512_VNNI_KBlock, tAVX512_VNNI_KBlock::ISA>>(BlkSize, N, K, isAsym);
        }
        if (_cd->AVX_VNNI() && BlkSize % tAVX_VNNI_KBlock::KTILE == 0) {
          return NSQ4BuSize<tWeiNInt<tAVX_VNNI_KBlock, tAVX_VNNI_KBlock::ISA>>(BlkSize, N, K, isAsym);
        }
      }
      [[fallthrough]];
    case CompBf16:
    case CompFp16:
    case CompFp32:
    case CompUndef:
      if (_cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
        return NSQ4BuSize<tWeiNInt<tAVX512F, tAVX512F::ISA>>(BlkSize, N, K, isAsym);
      }
      if (_cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
        return NSQ4BuSize<tWeiNInt<tAVX2, tAVX2::ISA>>(BlkSize, N, K, isAsym);
      }
      [[fallthrough]];
    default:
      return 0;
  }
  return 0;
}

static bool NSQ4GemmPackB(void* PackedBuf, const uint8_t* QData, const float* Scale, const uint8_t* Zp, size_t N,
                          size_t K, size_t ldb, size_t BlkSize, bool isAsym, bool lastCall,
                          NS_SQNBIT_COMPUTE_TYPE CompType, void* ThreadPool) {
  GetCPUDevice();
  // explicit statement fall through.
  switch (CompType) {
    case CompInt8:
      if (!isAsym) {  // asym int8 is not optimized, so fall through to others.
        if (_cd->AMX_INT8() && BlkSize % tAMX_INT8_SS_KBlock::KTILE == 0) {
          NSQ4GemmPackBImpl<tWeiNInt<tAMX_INT8_SS_KBlock, tAMX_INT8_SS_KBlock::ISA>>(
              PackedBuf, BlkSize, QData, Scale, Zp, N, K, isAsym, lastCall, ldb, ThreadPool);
          return true;
        }
        if (_cd->AVX512_VNNI() && BlkSize % tAVX512_VNNI_KBlock::KTILE == 0) {
          NSQ4GemmPackBImpl<tWeiNInt<tAVX512_VNNI_KBlock, tAVX512_VNNI_KBlock::ISA>>(
              PackedBuf, BlkSize, QData, Scale, Zp, N, K, isAsym, lastCall, ldb, ThreadPool);
          return true;
        }
        if (_cd->AVX_VNNI() && BlkSize % tAVX_VNNI_KBlock::KTILE == 0) {
          NSQ4GemmPackBImpl<tWeiNInt<tAVX_VNNI_KBlock, tAVX_VNNI_KBlock::ISA>>(PackedBuf, BlkSize, QData, Scale, Zp, N,
                                                                               K, isAsym, lastCall, ldb, ThreadPool);
          return true;
        }
      }
      [[fallthrough]];
    case CompBf16:
    case CompFp16:
    case CompFp32:
    case CompUndef:
      if (_cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
        NSQ4GemmPackBImpl<tWeiNInt<tAVX512F, tAVX512F::ISA>>(PackedBuf, BlkSize, QData, Scale, Zp, N, K, isAsym,
                                                             lastCall, ldb, ThreadPool);
        return true;
      }
      if (_cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
        NSQ4GemmPackBImpl<tWeiNInt<tAVX2, tAVX2::ISA>>(PackedBuf, BlkSize, QData, Scale, Zp, N, K, isAsym, lastCall,
                                                       ldb, ThreadPool);
        return true;
      }
      [[fallthrough]];
    default:
      return false;
  }
  return false;
}

size_t NSNBitsGemmPackBSize(size_t N, size_t K, size_t BlkSize, int nbits, bool isAsym,
                            NS_SQNBIT_COMPUTE_TYPE CompType) {
  if (nbits == 4) {
    auto jsize = NSQ4GemmPackBSize(N, K, BlkSize, isAsym, CompType);
    if (jsize) {
      return jsize;
    }
  }
  return 0;
}

void NSNBitsGemmPackB(void* PackedBuf, const uint8_t* QData, const float* Scale, const uint8_t* Zp, size_t N, size_t K,
                      size_t ldb, size_t BlkSize, int nbits, bool isAsym, bool lastCall,
                      NS_SQNBIT_COMPUTE_TYPE CompType, void* ThreadPool) {
  if (nbits == 4) {
    if (NSQ4GemmPackB(PackedBuf, QData, Scale, Zp, N, K, ldb, BlkSize, isAsym, lastCall, CompType, ThreadPool)) {
      return;
    }
  }
}

void NSNBitsGemmUnPackB(float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb, void* ThreadPool) {
  // only nbits=4 can be packed, so not necessary to check the nbits in DataParams
  if (NSQ4GemmUnPackB(FpData, PackedBuf, N, K, ldb, ThreadPool)) {
    return;
  }
}

size_t NSSQNBitsGemmBatchWorkspaceSize(const size_t M, const size_t N, const size_t K, const size_t BatchN,
                                       const NS_SQNBITS_GEMM_DATA_PACKED_PARAMS* DataParams) {
  // only nbits=4 can be packed, so not necessary to check the nbits in DataParams
  return NSSQ4GemmBatchWorkspaceSize(M, N, K, BatchN, DataParams);
}

void NSSQNBitsGemmBatchPackedB(const size_t M, const size_t N, const size_t K, const size_t BatchN,
                               const NS_SQNBITS_GEMM_DATA_PACKED_PARAMS* DataParams, void* WorkSpace,
                               void* ThreadPool) {
  // Prepare system config for AMX instructions
  auto p = Platform::get();
  (void)(p);
  // only nbits=4 can be packed, so not necessary to check the nbits in DataParams
  if (NSSQ4GemmBatchDriver(M, N, K, BatchN, DataParams, reinterpret_cast<int8_t*>(WorkSpace), ThreadPool)) {
    // PackedWeight is created by bestla
    return;
  }
}
