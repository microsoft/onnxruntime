//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#pragma once
#include <immintrin.h>
#include <cassert>

#include "jit_blas.h"
#include "jit_blas_gemm.h"
#include "jit_blas_utils.h"
#include "jit_blas_storage.h"
#include "jit_blas_device.h"
#include "jit_blas_parallel.h"
#include "kernel_wrapper.h"

namespace jblas {
namespace prologue_a {
namespace gemm {
template <class _GemmCore_T, JBLAS_ISA ISA_T>
class ActivationBase {
 public:
  using AType = typename _GemmCore_T::AType;
  using SRCType = AType;
  struct Param {
    const AType* A;
    int lda;
  };
  ActivationBase() {}

  JBLAS_CODE getActivation(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                           int k_offset, void* tmpcache, size_t cachesize) {
    auto aptr = const_cast<AType*>(_param.A);
    if (k_size % _GemmCore_T::KTILE == 0) {
      *dstptr = aptr + m_offset * _param.lda + k_offset;
      *dststep = _param.lda;
      return JblasSuccess;
    } else {
      auto k_pad = utils::padto(k_size, _GemmCore_T::KTILE);
      *dststep = k_pad;
      return kernel::wrapper::Memcpy2D::forward<ISA_T, AType, AType>(aptr + m_offset * _param.lda + k_offset, *dstptr,
                                                                     m_size, k_size, _param.lda, k_pad);
    }
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T, typename SRC_T>
class ActivationConverter {
 public:
  using AType = typename _GemmCore_T::AType;
  using SRCType = SRC_T;
  struct Param {
    const SRC_T* A;
    int lda;
  };
  ActivationConverter() {}

  JBLAS_CODE getActivation(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                           int k_offset, void* tmpcache, size_t cachesize) {
    auto aptr = const_cast<SRC_T*>(_param.A);
    auto k_pad = utils::padto(k_size, _GemmCore_T::KTILE);
    *dststep = k_pad;
    if constexpr (std::is_same_v<AType, utils::bf16> && std::is_same_v<SRC_T, float>) {
      return kernel::wrapper::Memcpy2DFp32CvtBf16::forward<ISA_T>(aptr + m_offset * _param.lda + k_offset, *dstptr,
                                                                  m_size, k_size, _param.lda * sizeof(SRC_T),
                                                                  k_pad * sizeof(AType), true);
    } else if constexpr (std::is_same_v<AType, utils::fp16> && std::is_same_v<SRC_T, float>) {
      return kernel::wrapper::Memcpy2DFp32CvtFp16::forward<ISA_T>(aptr + m_offset * _param.lda + k_offset, *dstptr,
                                                                  m_size, k_size, _param.lda * sizeof(SRC_T),
                                                                  k_pad * sizeof(AType), true);
    } else if constexpr (std::is_same_v<AType, float> && std::is_same_v<SRC_T, utils::bf16>) {
      return kernel::wrapper::Memcpy2DBf16CvtFp32::forward<ISA_T>(aptr + m_offset * _param.lda + k_offset, *dstptr,
                                                                  m_size, k_size, _param.lda * sizeof(SRC_T),
                                                                  k_pad * sizeof(AType), true);
    } else {
      assert(0);
    }
    return JblasNotSupport;
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
using ActivationConverterFp32 = ActivationConverter<_GemmCore_T, ISA_T, float>;
template <class _GemmCore_T, JBLAS_ISA ISA_T>
using ActivationConverterBf16 = ActivationConverter<_GemmCore_T, ISA_T, utils::bf16>;

template <class _GemmCore_T, JBLAS_ISA ISA_T, typename SRC_T>
class ActivationKBlockQuantize {
 public:
  using AType = typename _GemmCore_T::AType;
  using SType = float;
  using QParam = storage::gemm::StorageQuantActivation;
  using SRCType = SRC_T;
  struct Param {
    const SRC_T* A;
    int lda;
    QParam* quan;
  };
  using Parallel = jblas::parallel::Scheduler2D;
  using ThreadProblem = jblas::parallel::ThreadProblem2D;

  inline QParam createStorage(int m, int k, int kblock, bool hasreduce) {
    QParam tmp;
    int kpad = utils::padto(k, _GemmCore_T::KTILE);
    tmp.resize(m, kpad, kblock == -1 ? kpad : kblock, JBLAS_DTYPE::U8, JBLAS_DTYPE::F32, JBLAS_DTYPE::U8,
               JBLAS_DTYPE::F32, std::is_same_v<AType, uint8_t>, hasreduce);
    return tmp;
  }

  void run(const Param& _param, ThreadProblem& thdp) {
    auto quan = _param.quan;
    if (thdp.valid) {
      // min max
      auto srcptr = const_cast<SRC_T*>(_param.A) + thdp.loc[0] * _param.lda + thdp.loc[1];
      auto thdqptr = quan->template APtr<AType>() + thdp.loc[0] * quan->lda + thdp.loc[1];
      auto blk_offset = thdp.loc[0] * quan->mCStep + thdp.loc[1] / quan->kblock;
      auto thdsptr = quan->template SPtr<float>() + blk_offset;
      auto thdzptr = quan->template ZPtr<AType>() + blk_offset;
      auto thdrptr = quan->RPtr<float>() == nullptr ? nullptr : quan->RPtr<float>() + blk_offset;
      if constexpr (std::is_same_v<AType, uint8_t>) {
        kernel::wrapper::QuantizeU8ColBlock::template forward<ISA_T, SRC_T>(
            thdp.size[0], thdp.size[1], srcptr, _param.lda, thdqptr, quan->lda, thdsptr, quan->mCStep, thdzptr,
            quan->kblock, thdrptr);
      }
      if constexpr (std::is_same_v<AType, int8_t>) {
        kernel::wrapper::QuantizeS8ColBlock::template forward<ISA_T, SRC_T>(thdp.size[0], thdp.size[1], srcptr,
                                                                            _param.lda, thdqptr, quan->lda, thdsptr,
                                                                            quan->mCStep, quan->kblock, thdrptr);
      }
    }
  }

  JBLAS_CODE quantize(const Param& _param, int m, int k, jblas::parallel::IThreading* threading) {
    auto paral = Parallel({threading->num_threads(), m, k, 1, _param.quan->kblock});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp{tidx};
      paral.getIndex(thdp);
      if (thdp.valid) run(_param, thdp);
    });
    return JblasSuccess;
  }

 public:  // Runtime get by launcher
  JBLAS_CODE getActivation(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                           int k_offset, void* tmpcache, size_t cachesize) {
    (void)m_size;
    (void)k_size;
    auto quan = _param.quan;
    auto aptr = quan->template APtr<AType>();
    *dstptr = aptr + m_offset * _param.lda + k_offset;
    *dststep = _param.lda;
    return JblasSuccess;
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
using ActivationF32KBlockQuantize = ActivationKBlockQuantize<_GemmCore_T, ISA_T, float>;
template <class _GemmCore_T, JBLAS_ISA ISA_T>
using ActivationBf16KBlockQuantize = ActivationKBlockQuantize<_GemmCore_T, ISA_T, utils::bf16>;

template <class _GemmCore_T, JBLAS_ISA ISA_T, typename SRC_T>
class ActivationKBlockBase : public ActivationBase<_GemmCore_T, ISA_T> {
 public:
  using AType = typename _GemmCore_T::AType;
  using SType = storage::gemm::StorageReduce;
  using SRCType = SRC_T;
  using Param = typename ActivationBase<_GemmCore_T, ISA_T>::Param;
  using Parallel = jblas::parallel::Scheduler2D;
  using ThreadProblem = jblas::parallel::ThreadProblem2D;

  inline SType createStorage(int m, int k, int kblock) {
    SType tmp;
    tmp.resize(m, k, kblock == -1 ? k : kblock, JBLAS_DTYPE::F32);
    return tmp;
  }

  void run(const Param& _param, SType* stor, int m, int k, ThreadProblem& thdp) {
    if (thdp.valid) {
      // min max
      auto srcptr = const_cast<SRC_T*>(_param.A) + thdp.loc[0] * _param.lda + thdp.loc[1];
      auto blk_offset = thdp.loc[0] * stor->lda + thdp.loc[1] / stor->kblock;
      auto thdrptr = stor->template get<float>() + blk_offset;
      auto ret = kernel::wrapper::ColBlockReduceSum::template forward<ISA_T, SRC_T>(
          srcptr, _param.lda, thdp.size[0], thdp.size[1], stor->kblock, thdrptr, stor->lda);
      assert(ret == JblasSuccess);
    }
  }

  JBLAS_CODE reduce(const Param& _param, SType* stor, int m, int k, jblas::parallel::IThreading* threading) {
    auto paral = Parallel({threading->num_threads(), m, k, 1, stor->kblock});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp{tidx};
      paral.getIndex(thdp);
      if (thdp.valid) run(_param, stor, m, k, thdp);
    });
    return JblasSuccess;
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
using ActivationKBlockBaseF32 = ActivationKBlockBase<_GemmCore_T, ISA_T, float>;
}  // namespace gemm
}  // namespace prologue_a
}  // namespace jblas
