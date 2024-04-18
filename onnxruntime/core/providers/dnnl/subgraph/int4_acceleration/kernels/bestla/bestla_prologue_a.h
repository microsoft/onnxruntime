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
#include <type_traits>

#include "bestla.h"
#include "bestla_device.h"
#include "bestla_gemm.h"
#include "bestla_parallel.h"
#include "bestla_storage.h"
#include "bestla_utils.h"
#include "kernel_wrapper.h"

namespace bestla {
namespace prologue_a {
namespace gemm {

template <typename AType>
struct ParamActivationBase {
  const AType* A;
  int lda;
};
template <class _GemmCore_T, BTLA_ISA ISA_T>
class ActivationBase {
 public:
  using AType = typename _GemmCore_T::AType;
  using SRCType = AType;
  using Param = ParamActivationBase<AType>;
  BTLA_CODE getActivation(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                          int k_offset, void* tmpcache, size_t cachesize) {
    auto aptr = const_cast<AType*>(_param.A) + m_offset * _param.lda + k_offset;
    auto alignedptr = utils::cpu_pointer_align(aptr);
    bool use_rawptr = k_size % _GemmCore_T::KTILE == 0 && m_size >= _GemmCore_T::MTILE;
    use_rawptr = use_rawptr && (alignedptr == aptr);
    if (use_rawptr) {
      *dstptr = aptr;
      *dststep = _param.lda;
      return BTLA_CODE::Success;
    } else {
      auto k_pad = utils::padto(k_size, _GemmCore_T::KTILE);
      *dststep = k_pad;
      return kernel::wrapper::Memcpy2D::forward<BTLA_ISA::NoSIMD, AType, AType>(aptr, *dstptr, m_size, k_size,
                                                                                _param.lda, k_pad);
    }
  }
};

template <class _GemmCore_T, BTLA_ISA ISA_T, typename SRC_T>
class ActivationConverter : public ActivationBase<_GemmCore_T, ISA_T> {
 public:
  using AType = typename _GemmCore_T::AType;
  using SRCType = SRC_T;
  using Param = ParamActivationBase<SRC_T>;
  BTLA_CODE getActivation(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
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
    } else if constexpr (std::is_same_v<AType, SRC_T>) {
      return ActivationBase<_GemmCore_T, ISA_T>::getActivation(dstptr, dststep, {_param.A, _param.lda}, m_size, k_size,
                                                               m_offset, k_offset, tmpcache, cachesize);
    } else {
      assert(0);
    }
    return BTLA_CODE::NotSupport;
  }
};

template <class _GemmCore_T, BTLA_ISA ISA_T>
using ActivationConverterFp32 = ActivationConverter<_GemmCore_T, ISA_T, float>;
template <class _GemmCore_T, BTLA_ISA ISA_T>
using ActivationConverterBf16 = ActivationConverter<_GemmCore_T, ISA_T, utils::bf16>;

template <typename AType>
struct ParamActivationKBlockQuantize : ParamActivationBase<AType> {
  storage::gemm::StorageQuantActivation* quan;
};
template <class _GemmCore_T, BTLA_ISA ISA_T, typename SRC_T>
class ActivationKBlockQuantize {
 public:
  using AType = typename _GemmCore_T::AType;
  using SType = float;
  using QParam = storage::gemm::StorageQuantActivation;
  using SRCType = SRC_T;
  using Param = ParamActivationKBlockQuantize<SRC_T>;
  using Parallel = parallel::Scheduler2D;
  using ThreadProblem = parallel::ThreadProblem2D;

  inline Parallel createParallel(int nthreads, const utils::GemmProblem& prbm) {
    return Parallel({
        nthreads, prbm.dims[1],  // m
        prbm.dims[3],            // k
        1,
        prbm.dims[4]  // kblock
    });
  }

  inline QParam createStorage(int m, int k, int kblock, bool hasreduce) {
    QParam tmp;
    int kpad = utils::padto(k, _GemmCore_T::KTILE);
    int mpad = utils::padto(m, _GemmCore_T::MTILE);
    tmp.resize(mpad, kpad, m, k, kblock == -1 ? kpad : kblock, BTLA_DTYPE::U8, BTLA_DTYPE::F32, BTLA_DTYPE::U8,
               BTLA_DTYPE::F32, std::is_same_v<AType, uint8_t>, hasreduce);
    return tmp;
  }

  void run(const Param& _param, ThreadProblem& thdp) {
    auto quan = _param.quan;
    if (thdp.valid) {
      // min max
      auto srcptr = const_cast<SRC_T*>(_param.A) + thdp.loc[0] * _param.lda + thdp.loc[1];
      auto thdqptr = quan->template APtr<AType>() + thdp.loc[0] * quan->mKPad + thdp.loc[1];
      auto blk_offset = thdp.loc[0] * quan->CStep() + thdp.loc[1] / quan->mBlockSize;
      auto thdsptr = quan->template SPtr<float>() + blk_offset;
      auto thdzptr = quan->template ZPtr<AType>() + blk_offset;
      auto thdrptr = quan->template RPtr<float>() == nullptr ? nullptr : quan->template RPtr<float>() + blk_offset;
      if constexpr (std::is_same_v<AType, uint8_t>) {
        kernel::wrapper::QuantizeU8ColBlock::template forward<ISA_T, SRC_T>(
            thdp.size[0], thdp.size[1], srcptr, _param.lda, thdqptr, quan->mKPad, thdsptr, quan->CStep(), thdzptr,
            quan->mBlockSize, thdrptr);
      }
      if constexpr (std::is_same_v<AType, int8_t>) {
        kernel::wrapper::QuantizeS8ColBlock::template forward<ISA_T, SRC_T>(thdp.size[0], thdp.size[1], srcptr,
                                                                            _param.lda, thdqptr, quan->mKPad, thdsptr,
                                                                            quan->CStep(), quan->mBlockSize, thdrptr);
      }
    }
  }

  BTLA_CODE quantize(const Param& _param, int m, int k, parallel::IThreading* threading) {
    auto paral = Parallel({threading->num_threads(), m, k, 1, _param.quan->mBlockSize});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp{tidx};
      paral.getIndex(thdp);
      if (thdp.valid) run(_param, thdp);
    });
    return BTLA_CODE::Success;
  }

 public:  // Runtime get by launcher
  BTLA_CODE getActivation(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                          int k_offset, void* tmpcache, size_t cachesize) {
    (void)m_size;
    (void)k_size;
    auto quan = _param.quan;
    auto aptr = quan->template APtr<AType>();
    *dstptr = aptr + m_offset * quan->mKPad + k_offset;
    *dststep = quan->mKPad;
    return BTLA_CODE::Success;
  }

  BTLA_CODE getZp(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset, int k_offset,
                  void* tmpcache, size_t cachesize) {
    auto quan = _param.quan;
    auto aptr = quan->template ZPtr<AType>();
    if (aptr == nullptr) {  // optional
      *dstptr = nullptr;
      return BTLA_CODE::Success;
    }
    int kele = utils::updiv(k_size, quan->mBlockSize);
    *dststep = kele;
    kernel::ref::memcpy2d(aptr + m_offset * quan->CStep() + k_offset / quan->mBlockSize, *dstptr, m_size,
                          kele * sizeof(AType), quan->CStep() * sizeof(AType), kele * sizeof(AType));
    return BTLA_CODE::Success;
  }

  BTLA_CODE getScale(float** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                     int k_offset, void* tmpcache, size_t cachesize) {
    auto quan = _param.quan;
    auto aptr = quan->template SPtr<float>();
    int kele = utils::updiv(k_size, quan->mBlockSize);
    *dststep = kele;
    kernel::ref::memcpy2d(aptr + m_offset * quan->CStep() + k_offset / quan->mBlockSize, *dstptr, m_size,
                          kele * sizeof(float), quan->CStep() * sizeof(float), kele * sizeof(float));
    return BTLA_CODE::Success;
  }

  BTLA_CODE getReduce(float** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                      int k_offset, void* tmpcache, size_t cachesize) {
    auto quan = _param.quan;
    auto aptr = quan->template RPtr<float>();
    int kele = utils::updiv(k_size, quan->mBlockSize);
    *dststep = kele;
    kernel::ref::memcpy2d(aptr + m_offset * quan->CStep() + k_offset / quan->mBlockSize, *dstptr, m_size,
                          kele * sizeof(float), quan->CStep() * sizeof(float), kele * sizeof(float));
    return BTLA_CODE::Success;
  }
};

template <class _GemmCore_T, BTLA_ISA ISA_T>
using ActivationF32KBlockQuantize = ActivationKBlockQuantize<_GemmCore_T, ISA_T, float>;
template <class _GemmCore_T, BTLA_ISA ISA_T>
using ActivationBf16KBlockQuantize = ActivationKBlockQuantize<_GemmCore_T, ISA_T, utils::bf16>;

template <typename AType>
struct ParamActivationKBlockBase : ParamActivationBase<AType> {
  storage::gemm::StorageReduce* reduce;
};
template <class _GemmCore_T, BTLA_ISA ISA_T, typename SRC_T>
class ActivationKBlockBase : public ActivationConverter<_GemmCore_T, ISA_T, SRC_T> {
 public:
  using AType = typename _GemmCore_T::AType;
  using SType = storage::gemm::StorageReduce;
  using SRCType = SRC_T;
  using Param = ParamActivationKBlockBase<SRC_T>;
  using Parallel = parallel::Scheduler2D;
  using ThreadProblem = parallel::ThreadProblem2D;

  inline Parallel createParallel(int nthreads, const utils::GemmProblem& prbm) {
    return Parallel({
        nthreads, prbm.dims[1],  // m
        prbm.dims[3],            // k
        1,
        prbm.dims[4]  // kblock
    });
  }
  inline SType createStorage(int m, int k, int kblock) {
    SType tmp;
    tmp.resize(m, k, kblock == -1 ? k : kblock, BTLA_DTYPE::F32);
    return tmp;
  }

  void run(const Param& _param, ThreadProblem& thdp) {
    auto stor = _param.reduce;
    if (thdp.valid) {
      // min max
      auto srcptr = const_cast<SRC_T*>(_param.A) + thdp.loc[0] * _param.lda + thdp.loc[1];
      auto blk_offset = thdp.loc[0] * stor->lda + thdp.loc[1] / stor->kblock;
      auto thdrptr = stor->template RPtr<float>() + blk_offset;
      auto ret = kernel::wrapper::ColBlockReduceSum::template forward<ISA_T, SRC_T>(
          srcptr, _param.lda, thdp.size[0], thdp.size[1], stor->kblock, thdrptr, stor->lda);
      assert(ret == BTLA_CODE::Success);
    }
  }

  BTLA_CODE reduce(const Param& _param, int m, int k, int kblock, parallel::IThreading* threading) {
    auto paral = Parallel({threading->num_threads(), m, k, 1, kblock});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp{tidx};
      paral.getIndex(thdp);
      if (thdp.valid) run(_param, thdp);
    });
    return BTLA_CODE::Success;
  }

  BTLA_CODE getActivation(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                          int k_offset, void* tmpcache, size_t cachesize) {
    return ActivationConverter<_GemmCore_T, ISA_T, SRC_T>::getActivation(
        dstptr, dststep, {_param.A, _param.lda}, m_size, k_size, m_offset, k_offset, tmpcache, cachesize);
  }

  BTLA_CODE getReduce(float** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                      int k_offset, void* tmpcache, size_t cachesize) {
    auto reduce = _param.reduce;
    auto aptr = reduce->template RPtr<float>();
    int kele = utils::updiv(k_size, reduce->kblock);
    *dststep = kele;
    kernel::ref::memcpy2d(aptr + m_offset * reduce->lda + k_offset / reduce->kblock, *dstptr, m_size,
                          kele * sizeof(float), reduce->lda * sizeof(float), kele * sizeof(float));
    return BTLA_CODE::Success;
  }
};

template <class _GemmCore_T, BTLA_ISA ISA_T>
using ActivationKBlockBaseF32 = ActivationKBlockBase<_GemmCore_T, ISA_T, float>;

template <typename AType>
struct ParamShuffleActivationKBlockBase : ParamActivationKBlockBase<AType> {
  int* indices = nullptr;
  storage::gemm::StorageReorderActivation* reordered = nullptr;
};
template <class _GemmCore_T, BTLA_ISA ISA_T, typename SRC_T>
class ShuffleActivationKBlockBase : public ActivationKBlockBase<_GemmCore_T, ISA_T, SRC_T> {
 public:
  using AType = typename _GemmCore_T::AType;
  using RedType = storage::gemm::StorageReduce;
  using RAType = storage::gemm::StorageReorderActivation;
  using SRCType = SRC_T;
  using Param = ParamShuffleActivationKBlockBase<SRC_T>;
  using Parallel = parallel::Scheduler2D;
  using ThreadProblem = parallel::ThreadProblem2D;
  inline RAType createReorderStorage(int m, int k, int kblock) {
    RAType tmp(_GemmCore_T::ID);
    int kpad = utils::padto(k, _GemmCore_T::KTILE);
    int mpad = utils::padto(m, _GemmCore_T::MTILE);
    tmp.resize(mpad, kpad, m, k, kblock == -1 ? kpad : kblock, utils::bestla_dtype<SRC_T>);
    return tmp;
  }

  inline RedType createReduceStorage(int m, int k, int kblock) {
    RedType tmp;
    tmp.resize(m, k, kblock == -1 ? k : kblock, BTLA_DTYPE::F32);
    return tmp;
  }

  void run(const Param& _param, ThreadProblem& thdp) {
    auto stor = _param.reduce;
    auto reordered = _param.reordered;
    if (thdp.valid) {
      auto srcptr = const_cast<SRC_T*>(_param.A) + thdp.loc[0] * _param.lda + thdp.loc[1];
      if (reordered && _param.indices) {
        auto rptr = reordered->template APtr<SRC_T>() + thdp.loc[0] * reordered->mKPad + thdp.loc[1];
        auto ret =
            kernel::ref::shuffle_activation(const_cast<SRC_T*>(_param.A), rptr, thdp.size[0], thdp.size[1], thdp.loc[0],
                                            thdp.loc[1], _param.indices, _param.lda, reordered->mKPad);
        srcptr = rptr;
      }
      if (stor) {
        // min max
        auto blk_offset = thdp.loc[0] * stor->lda + thdp.loc[1] / stor->kblock;
        auto thdrptr = stor->template RPtr<float>() + blk_offset;
        auto ret = kernel::wrapper::ColBlockReduceSum::template forward<ISA_T, SRC_T>(
            srcptr, _param.lda, thdp.size[0], thdp.size[1], stor->kblock, thdrptr, stor->lda);
        assert(ret == BTLA_CODE::Success);
      }
    }
  }

  BTLA_CODE preprocess(const Param& _param, int m, int k, int kblock, parallel::IThreading* threading) {
    auto paral = Parallel({threading->num_threads(), m, k, 1, kblock});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp{tidx};
      paral.getIndex(thdp);
      run(_param, thdp);
    });
    return BTLA_CODE::Success;
  }

  BTLA_CODE getActivation(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                          int k_offset, void* tmpcache, size_t cachesize) {
    if (_param.indices == nullptr) {
      return ActivationConverter<_GemmCore_T, ISA_T, SRC_T>::getActivation(
          dstptr, dststep, {_param.A, _param.lda}, m_size, k_size, m_offset, k_offset, tmpcache, cachesize);
    } else {
      return ActivationConverter<_GemmCore_T, ISA_T, SRC_T>::getActivation(
          dstptr, dststep, {_param.reordered->template APtr<SRC_T>(), _param.reordered->mKPad}, m_size, k_size,
          m_offset, k_offset, tmpcache, cachesize);
    }
  }
};

template <class _GemmCore_T, BTLA_ISA ISA_T>
using ShuffleActivationKBlockBaseF32 = ShuffleActivationKBlockBase<_GemmCore_T, ISA_T, float>;
template <class _GemmCore_T, BTLA_ISA ISA_T>
using ShuffleActivationKBlockBaseBf16 = ShuffleActivationKBlockBase<_GemmCore_T, ISA_T, utils::bf16>;

template <typename AType>
struct ParamShuffleActivationKBlockQuantize : ParamActivationKBlockQuantize<AType> {
  int* indices = nullptr;
  storage::gemm::StorageReorderActivation* reordered = nullptr;
};
template <class _GemmCore_T, BTLA_ISA ISA_T, typename SRC_T>
class ShuffleActivationKBlockQuantize : public ActivationKBlockQuantize<_GemmCore_T, ISA_T, SRC_T> {
 public:
  using AType = typename _GemmCore_T::AType;
  using SType = float;
  using QParam = storage::gemm::StorageQuantActivation;
  using RAType = storage::gemm::StorageReorderActivation;
  using SRCType = SRC_T;
  using Param = ParamShuffleActivationKBlockQuantize<SRC_T>;
  using Parallel = parallel::Scheduler2D;
  using ThreadProblem = parallel::ThreadProblem2D;

  inline QParam createQuantStorage(int m, int k, int kblock, bool hasreduce) {
    QParam tmp;
    int kpad = utils::padto(k, _GemmCore_T::KTILE);
    int mpad = utils::padto(m, _GemmCore_T::MTILE);
    tmp.resize(mpad, kpad, m, k, kblock == -1 ? kpad : kblock, BTLA_DTYPE::U8, BTLA_DTYPE::F32, BTLA_DTYPE::U8,
               BTLA_DTYPE::F32, std::is_same_v<AType, uint8_t>, hasreduce);
    return tmp;
  }

  inline RAType createReorderStorage(int m, int k, int kblock) {
    RAType tmp(_GemmCore_T::ID);
    int kpad = utils::padto(k, _GemmCore_T::KTILE);
    int mpad = utils::padto(m, _GemmCore_T::MTILE);
    tmp.resize(mpad, kpad, m, k, kblock == -1 ? kpad : kblock, utils::bestla_dtype<SRC_T>);
    return tmp;
  }

  BTLA_CODE quantize(const Param& _param, int m, int k, parallel::IThreading* threading) {
    auto srcptr = const_cast<SRC_T*>(_param.A);
    if (_param.indices) {
      auto shuffle_src = _param.reordered->template APtr<SRC_T>();
      threading->parallel_for([&](int tidx) {
        auto enable_thr = threading->num_threads();
        auto align_m = m / enable_thr;
        auto process_m = (tidx + 1) == enable_thr ? (m - tidx * align_m) : align_m;
        kernel::ref::shuffle_activation(const_cast<SRC_T*>(_param.A), shuffle_src + tidx * align_m * k, process_m, k,
                                        tidx * align_m, 0, _param.indices, k, k);
      });
      srcptr = shuffle_src;
    }
    ActivationKBlockQuantize<_GemmCore_T, ISA_T, SRC_T>::quantize({srcptr, k, _param.quan}, m, k, threading);
    return BTLA_CODE::Success;
  }
};

template <class _GemmCore_T, BTLA_ISA ISA_T>
using ShuffleActivationKBlockQuantizeF32 = ShuffleActivationKBlockQuantize<_GemmCore_T, ISA_T, float>;
template <class _GemmCore_T, BTLA_ISA ISA_T>
using ShuffleActivationKBlockQuantizeBf16 = ShuffleActivationKBlockQuantize<_GemmCore_T, ISA_T, utils::bf16>;
}  // namespace gemm
}  // namespace prologue_a
}  // namespace bestla
