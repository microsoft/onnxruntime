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

#include "jit_base.hpp"
#include "jit_blas.h"
#include "jit_blas_gemm.h"
#include "jit_blas_utils.h"
#include "kernel_wrapper.h"

namespace jblas {
namespace prologue {
enum class WeightPrologueType : int {
  Undef = -1,
  Begin = 0,
  WeightPack = Begin,
  End,
};
class PackedWeight {
 public:
  PackedWeight(jblas::gemm::GemmCoreType type) {
    mNPad = 0;
    mKPad = 0;
    mSize = 0;
    mCoreType = type;
  }

  virtual ~PackedWeight() {}

  void resize(int NPad, int KPad) {
    mNPad = NPad;
    mKPad = KPad;
  }

  virtual size_t getSerializedSize() {
    size_t totalsize = 0;
    totalsize += sizeof(mSize);
    totalsize += sizeof(mCoreType);
    totalsize += sizeof(mType);
    totalsize += sizeof(mNPad);
    totalsize += sizeof(mKPad);
    totalsize += getDataSerializedSize();
    return totalsize;
  }

  virtual void serializeToBuffer(void* buf) {
    auto wptr = reinterpret_cast<int8_t*>(buf);
    mSize = getSerializedSize();
    utils::serialize(wptr, mSize);
    utils::serialize(wptr, mCoreType);
    utils::serialize(wptr, mType);
    utils::serialize(wptr, mNPad);
    utils::serialize(wptr, mKPad);
    serializeDataToBuffer(wptr);
  }

  virtual void deserializeBuffer(void* buf, int memalloc) {
    auto rptr = reinterpret_cast<int8_t*>(buf);
    mSize = utils::deserialize<size_t>(rptr);
    mCoreType = utils::deserialize<jblas::gemm::GemmCoreType>(rptr);
    mType = utils::deserialize<int>(rptr);
    mNPad = utils::deserialize<int>(rptr);
    mKPad = utils::deserialize<int>(rptr);
    deserializeDataBuffer(rptr, memalloc);
  }
  size_t mSize = 0;
  jblas::gemm::GemmCoreType mCoreType = jblas::gemm::GemmCoreType::Undef;
  int mType = -1;
  int mNPad = 0, mKPad = 0;
  static int constexpr TypeOffset = sizeof(mSize) + sizeof(mCoreType);

 protected:
  virtual size_t getDataSerializedSize() = 0;
  virtual void serializeDataToBuffer(void* buf) = 0;
  virtual void deserializeDataBuffer(void* buf, int memalloc) = 0;
};

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
                           int k_offset) {
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
    return JblasSuccess;
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
                           int k_offset) {
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

template <typename QT_T, typename ST_T>
class StorageQuantActivation {
 public:
  QT_T *mWPtr, *mZPtr;
  ST_T* mSPtr;
  int lda, lds;
  void resize(int m, int lda, int lds, int8_t* ptr) {
    if (ptr) {
      setPtr(m, lda, lds, ptr);
    } else {
      auto size = getSize(m, lda, lds);
      mBuffer.resize(size);
      setPtr(m, lda, lds, mBuffer.data());
    }
  }

  static size_t getSize(int m, int lda, int lds) {
    size_t total = 0;
    total = size_t(m) * lda * sizeof(QT_T) + (size_t)m * lds * (sizeof(QT_T) + sizeof(ST_T));
    return total;
  }

  void setPtr(int m, int _lda, int _lds, int8_t* ptr) {
    lds = _lds;
    lda = _lda;
    mWPtr = (QT_T*)ptr;
    mZPtr = (QT_T*)(mWPtr + m * lda);
    mSPtr = (ST_T*)(mZPtr + m * lds);
  }

  utils::avector<int8_t> mBuffer;
};

template <typename QT_T, typename ST_T>
class StorageQuantActivationKblock : public StorageQuantActivation<QT_T, ST_T> {
 public:
  int kblock;
  void resize(int m, int k, int _kblock, int8_t* ptr) {
    if (ptr) {
      setPtr(m, k, _kblock, ptr);
    } else {
      auto size = getSize(m, k, _kblock);
      this->mBuffer.resize(size);
      setPtr(m, k, _kblock, this->mBuffer.data());
    }
  }

  static size_t getSize(int m, int k, int kblock) {
    int lda = k;
    int lds = utils::updiv(k, kblock);
    return StorageQuantActivation<QT_T, ST_T>::getSize(m, lda, lds);
  }

  void setPtr(int m, int k, int _kblock, int8_t* ptr) {
    kblock = _kblock;
    int lda = k;
    int lds = utils::updiv(k, kblock);
    StorageQuantActivation<QT_T, ST_T>::setPtr(m, lda, lds, ptr);
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T, typename SRC_T>
class ActivationU8KBlockQuantize {
 public:
  using AType = typename _GemmCore_T::AType;
  using SType = float;
  using QParam = StorageQuantActivationKblock<AType, SType>;
  using SRCType = SRC_T;
  struct Param {
    const SRC_T* A;
    int lda;
    QParam* quan;
  };
  using Parallel = utils::parallel::Parallel2DRowMajorColBlock;

  Parallel createParallel(int m, int k, int kblock) {
    Parallel _paral;
    auto cb = utils::CpuBase();
    _paral.update(m, k, 1, 16, kblock, cb.mNumThreads);
    return _paral;
  }

  size_t getWorkSpaceSize(int m, int k, int kblock) {
    int kpad = utils::padto(k, _GemmCore_T::KTILE);
    size_t totalsize = QParam::getSize(m, kpad, kblock);
    return totalsize;
  }

  QParam* createStorage(int m, int k, int kblock, int8_t* workspace) {
    auto ptr = new QParam;
    int kpad = utils::padto(k, _GemmCore_T::KTILE);
    ptr->resize(m, kpad, kblock, workspace);
    return ptr;
  }

  void launch(const Param& _param, int tidx, Parallel& para) {
    int colidx, rowidx, rowsize, colsize;
    int blkidx, idxinblk;
    para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize, &blkidx, &idxinblk);
    auto quan = _param.quan;
    if (rowsize > 0 && colsize > 0) {
      // min max
      auto srcptr = const_cast<SRC_T*>(_param.A) + rowidx * _param.lda + colidx;
      int rowremain = utils::remainsize(rowidx, para.mRows, rowsize);
      int colremain = utils::remainsize(colidx, para.mCols, colsize);
      auto thdqptr = quan->mWPtr + rowidx * quan->lda + colidx;
      auto thdsptr = quan->mSPtr + rowidx * quan->lds + blkidx;
      auto thdzptr = quan->mZPtr + rowidx * quan->lds + blkidx;
      kernel::wrapper::QuantizeU8ColBlock::template forward<ISA_T, SRC_T>(
          rowremain, colremain, srcptr, _param.lda, thdqptr, quan->lda, thdsptr, quan->lds, thdzptr, para.mColBlock);
    }
  }

  void quantize(const Param& _param, int m, int k) {
    utils::parallel::Parallel2DRowMajorColBlock paral = createParallel(m, k, _param.quan->kblock);
    if (paral.mThdsPerBlock == 1) {  // no barrier
#pragma omp parallel
      {
        int tidx = omp_get_thread_num();
        launch(_param, tidx, paral);
      }
    } else {
      assert(0);
    }
  }

  JBLAS_CODE getActivation(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                           int k_offset) {
    auto quan = _param.quan;
    auto aptr = const_cast<AType*>(quan->mWPtr);
    *dstptr = aptr + m_offset * _param.lda + k_offset;
    *dststep = _param.lda;
    return JblasSuccess;
  }

  JBLAS_CODE getScale(SType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                      int k_offset) {
    auto quan = _param.quan;
    auto ptr = const_cast<SType*>(quan->mSPtr);
    *dstptr = ptr + m_offset * quan->lds + k_offset / quan->kblock;
    *dststep = quan->lds;
    return JblasSuccess;
  }

  static inline JBLAS_CODE getZp(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size,
                                 int m_offset, int k_offset) {
    auto quan = _param.quan;
    *dstptr = &quan->mZPtr[(m_offset)*quan->lds + k_offset / quan->kblock];
    *dststep = quan->lds;
    return JblasSuccess;
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
using ActivationF32U8KBlockQuantize = ActivationU8KBlockQuantize<_GemmCore_T, ISA_T, float>;
template <class _GemmCore_T, JBLAS_ISA ISA_T>
using ActivationBf16U8KBlockQuantize = ActivationU8KBlockQuantize<_GemmCore_T, ISA_T, utils::bf16>;

template <class _GemmCore_T, JBLAS_ISA ISA_T, typename SRC_T>
class ActivationAsymU8Quantize {
 public:
  using AType = typename _GemmCore_T::AType;
  using SType = float;
  using QParam = StorageQuantActivation<AType, SType>;
  using SRCType = SRC_T;
  struct Param {
    const SRC_T* A;
    int lda;
    QParam* Q;
  };
  using Parallel = utils::parallel::Parallel2DRowMajor;

  Parallel createParallel(int m, int k) {
    Parallel _paral;
    auto cb = utils::CpuBase();
    _paral.update(m, k, 1, k, cb.mNumThreads);
    return _paral;
  }

  size_t getWorkSpaceSize(int m, int k) {
    int kpad = utils::padto(k, _GemmCore_T::KTILE);
    size_t totalsize = QParam::getSize(m, kpad, 1);
    return totalsize;
  }

  QParam* createStorage(int m, int k, int8_t* workspace) {
    auto ptr = new QParam;
    int kpad = utils::padto(k, _GemmCore_T::KTILE);
    ptr->resize(m, kpad, 1, workspace);
    return ptr;
  }

  void launch(const Param& _param, int tidx, Parallel& para) {
    int colidx, rowidx, rowsize, colsize;
    para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
    auto quan = _param.Q;
    if (rowsize > 0 && colsize > 0) {
      // min max
      auto srcptr = const_cast<SRC_T*>(_param.A) + rowidx * _param.lda + colidx;
      int rowremain = utils::remainsize(rowidx, para.mRows, rowsize);
      int colremain = utils::remainsize(colidx, para.mCols, colsize);
      auto thdqptr = quan->mWPtr + rowidx * quan->lda + colidx;
      auto thdsptr = quan->mSPtr + rowidx * quan->lds;
      auto thdzptr = quan->mZPtr + rowidx * quan->lds;
      kernel::wrapper::QuantizeU8ColBlock::template forward<ISA_T, SRC_T>(
          rowremain, colremain, srcptr, _param.lda, thdqptr, quan->lda, thdsptr, quan->lds, thdzptr, para.mCols);
    }
  }

  JBLAS_CODE quantize(const Param& _param, int m, int k) {
    auto paral = createParallel(m, k);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      launch(_param, tidx, paral);
    }
    return JblasSuccess;
  }

  JBLAS_CODE getActivation(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                           int k_offset) {
    auto aptr = const_cast<AType*>(_param.Q->mWPtr);
    *dstptr = aptr + m_offset * _param.Q->lda + k_offset;
    *dststep = _param.Q->lda;
    return JblasSuccess;
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
using ActivationFp32AsymU8Quantize = ActivationAsymU8Quantize<_GemmCore_T, ISA_T, float>;
template <class _GemmCore_T, JBLAS_ISA ISA_T>
using ActivationBf16AsymU8Quantize = ActivationAsymU8Quantize<_GemmCore_T, ISA_T, utils::bf16>;

template <class _GemmCore_T, JBLAS_ISA ISA_T, typename SRC_T>
class ActivationS8KBlockQuantize {
 public:
  using AType = typename _GemmCore_T::AType;
  using SType = float;
  using QParam = StorageQuantActivationKblock<AType, SType>;
  using Parallel = utils::parallel::Parallel2DRowMajorColBlock;
  using SRCType = SRC_T;

  struct Param {
    const SRC_T* A;
    int lda;
    QParam* quan;
  };

  Parallel createParallel(int m, int k, int kblock) {
    Parallel _paral;
    auto cb = utils::CpuBase();
    _paral.update(m, k, 1, 16, kblock, cb.mNumThreads);
    return _paral;
  }

  size_t getWorkSpaceSize(int m, int k, int kblock) {
    int kpad = utils::padto(k, _GemmCore_T::KTILE);
    size_t totalsize = QParam::getSize(m, kpad, kblock);
    return totalsize;
  }

  QParam* createStorage(int m, int k, int kblock, int8_t* workspace) {
    auto ptr = new QParam;
    int kpad = utils::padto(k, _GemmCore_T::KTILE);
    ptr->resize(m, kpad, kblock, workspace);
    return ptr;
  }

  void launch(const Param& _param, int tidx, Parallel& para) {
    int colidx, rowidx, rowsize, colsize;
    int blkidx, idxinblk;
    para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize, &blkidx, &idxinblk);
    auto quan = _param.quan;
    if (rowsize > 0 && colsize > 0) {
      // min max
      auto srcptr = const_cast<SRC_T*>(_param.A) + rowidx * _param.lda + colidx;
      int rowremain = utils::remainsize(rowidx, para.mRows, rowsize);
      int colremain = utils::remainsize(colidx, para.mCols, colsize);
      auto thdqptr = quan->mWPtr + rowidx * quan->lda + colidx;
      auto thdsptr = quan->mSPtr + rowidx * quan->lds + blkidx;
      kernel::wrapper::QuantizeS8ColBlock::template forward<ISA_T, SRC_T>(
          rowremain, colremain, srcptr, _param.lda, thdqptr, quan->lda, thdsptr, quan->lds, para.mColBlock);
    }
  }

  void quantize(const Param& _param, int m, int k) {
    utils::parallel::Parallel2DRowMajorColBlock paral = createParallel(m, k, _param.quan->kblock);
    if (paral.mThdsPerBlock == 1) {  // no barrier
#pragma omp parallel
      {
        int tidx = omp_get_thread_num();
        launch(_param, tidx, paral);
      }
    } else {
      assert(0);
    }
  }

  JBLAS_CODE getActivation(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                           int k_offset) {
    auto quan = _param.quan;
    auto aptr = const_cast<AType*>(quan->mWPtr);
    *dstptr = aptr + m_offset * quan->lda + k_offset;
    *dststep = quan->lda;
    return JblasSuccess;
  }

  JBLAS_CODE getScale(SType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                      int k_offset) {
    auto quan = _param.quan;
    auto ptr = const_cast<SType*>(quan->mSPtr);
    *dstptr = ptr + m_offset * quan->lds + k_offset / quan->kblock;
    *dststep = quan->lds;
    return JblasSuccess;
  }

  static inline JBLAS_CODE getZp(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size,
                                 int m_offset, int k_offset) {
    return JblasSuccess;
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
using ActivationF32S8KBlockQuantize = ActivationS8KBlockQuantize<_GemmCore_T, ISA_T, float>;
template <class _GemmCore_T, JBLAS_ISA ISA_T>
using ActivationBf16S8KBlockQuantize = ActivationS8KBlockQuantize<_GemmCore_T, ISA_T, utils::bf16>;

template <class _GemmCore_T, JBLAS_ISA ISA_T, typename SRC_T>
class ActivationSymS8Quantize {
 public:
  using AType = typename _GemmCore_T::AType;
  using SType = float;
  using QParam = StorageQuantActivation<AType, SType>;
  using SRCType = SRC_T;
  struct Param {
    const SRC_T* A;
    int lda;
    QParam* Q;
  };
  using Parallel = utils::parallel::Parallel2DRowMajor;

  Parallel createParallel(int m, int k) {
    Parallel _paral;
    auto cb = utils::CpuBase();
    _paral.update(m, k, 1, k, cb.mNumThreads);
    return _paral;
  }

  size_t getWorkSpaceSize(int m, int k) {
    int kpad = utils::padto(k, _GemmCore_T::KTILE);
    size_t totalsize = QParam::getSize(m, kpad, 1);
    return totalsize;
  }

  QParam* createStorage(int m, int k, int8_t* workspace) {
    auto ptr = new QParam;
    int kpad = utils::padto(k, _GemmCore_T::KTILE);
    ptr->resize(m, kpad, 1, workspace);
    return ptr;
  }

  void launch(const Param& _param, int tidx, Parallel& para) {
    int colidx, rowidx, rowsize, colsize;
    para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
    auto quan = _param.Q;
    if (rowsize > 0 && colsize > 0) {
      // min max
      auto srcptr = const_cast<SRC_T*>(_param.A) + rowidx * _param.lda + colidx;
      int rowremain = utils::remainsize(rowidx, para.mRows, rowsize);
      int colremain = utils::remainsize(colidx, para.mCols, colsize);
      auto thdqptr = quan->mWPtr + rowidx * quan->lda + colidx;
      auto thdsptr = quan->mSPtr + rowidx * quan->lds;
      auto thdzptr = quan->mZPtr + rowidx * quan->lds;
      kernel::wrapper::QuantizeS8ColBlock::template forward<ISA_T, SRC_T>(
          rowremain, colremain, srcptr, _param.lda, thdqptr, quan->lda, thdsptr, quan->lds, para.mCols);
    }
  }

  JBLAS_CODE quantize(const Param& _param, int m, int k) {
    auto paral = createParallel(m, k);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      launch(_param, tidx, paral);
    }
    return JblasSuccess;
  }

  JBLAS_CODE getActivation(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                           int k_offset) {
    auto aptr = const_cast<AType*>(_param.Q->mWPtr);
    *dstptr = aptr + m_offset * _param.Q->lda + k_offset;
    *dststep = _param.Q->lda;
    return JblasSuccess;
  }
};

template <typename T, JBLAS_ISA ISA_T>
class WeightBase {
 public:
  static void transposeWeight(const int Row, const int Col, const T* src, const int ld_src, T* dst, const int ld_dst) {
    utils::parallel::Parallel2DRowMajor _para;
    utils::CpuBase cb;
    _para.update(Row, Col, 16, 16, cb.mNumThreads);
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      int colidx, rowidx, rowsize, colsize;
      _para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        // rowremain: src valid size. rowsize: padded size
        int rowremain = utils::remainsize(rowidx, Row, rowsize);
        int colremain = utils::remainsize(colidx, Col, colsize);
        kernel::wrapper::Transpose2D<T>::template forward<ISA_T>(
            src + rowidx * ld_src + colidx, dst + rowidx + colidx * ld_dst, rowremain, colremain, ld_src, ld_dst);
      }
    }
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
using ActivationFp32SymS8Quantize = ActivationSymS8Quantize<_GemmCore_T, ISA_T, float>;
template <class _GemmCore_T, JBLAS_ISA ISA_T>
using ActivationBf16SymS8Quantize = ActivationSymS8Quantize<_GemmCore_T, ISA_T, utils::bf16>;

// Storage class has real weight memory, PackedWeight provides interface
class StorageWeight : public prologue::PackedWeight {
 public:
  StorageWeight(jblas::gemm::GemmCoreType _type) : PackedWeight(_type) {
    mRawPtr = NULL;
    mRawSize = 0;
    mType = int(WeightPrologueType::WeightPack);
  }

  void resize(int NPad, int KPad, int8_t* ptr) {
    mNPad = NPad;
    mKPad = KPad;
    mRawSize = getSize(NPad, KPad, jblas::gemm::getWeightSize(mCoreType));
    if (ptr) {
      mRawPtr = ptr;
    } else {
      mBuffer.resize((size_t)NPad * KPad * jblas::gemm::getWeightSize(mCoreType));
      mRawPtr = mBuffer.data();
    }
  }

  static size_t getSize(size_t NPad, size_t KPad, size_t EleBytes) { return NPad * KPad * EleBytes; }

  template <typename WT>
  inline WT* getPtr() const {
    return reinterpret_cast<WT*>(mRawPtr);
  }

  template <typename WT>
  inline size_t getSize() const {
    return mRawSize / sizeof(WT);
  }

  int8_t* mRawPtr;
  size_t mRawSize;

 protected:
  virtual size_t getDataSerializedSize() override {
    size_t totalsize = 0;
    totalsize += sizeof(mRawSize);
    totalsize += mRawSize;
    return totalsize;
  }
  virtual void serializeDataToBuffer(void* buf) override {
    auto wptr = reinterpret_cast<int8_t*>(buf);
    utils::serialize(wptr, mRawSize);
    std::memcpy(mRawPtr, wptr, mRawSize);
  }
  virtual void deserializeDataBuffer(void* buf, int memalloc) override {
    auto rptr = reinterpret_cast<int8_t*>(buf);
    size_t rsize = utils::deserialize<size_t>(rptr);
    if (memalloc) {
      mBuffer.resize(rsize);
      std::memcpy(mBuffer.data(), rptr, rsize);
      mRawPtr = mBuffer.data();
      mRawSize = mBuffer.size();
    } else {
      mRawPtr = (int8_t*)rptr;
      mRawSize = rsize;
    }
  }
  utils::aligned_vector<int8_t> mBuffer;
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class WeightPack : public WeightBase<typename _GemmCore_T::BType, ISA_T> {
 public:
  using WType = typename _GemmCore_T::BType;
  struct Param {
    const WType* B;
    const int ldb;
    StorageWeight* packedW;
  };
  using Parallel = utils::parallel::Parallel2DRowMajor;

  Parallel createParallel(int n, int k) {
    Parallel _paral;
    auto cb = utils::CpuBase();
    _paral.update(k, n, _GemmCore_T::KTILE, _GemmCore_T::NTILE, cb.mNumThreads);
    return _paral;
  }

  size_t getWorkSpaceSize(int n, int k) {
    int KPad = utils::padto(k, _GemmCore_T::KTILE);
    int NPad = utils::padto(n, _GemmCore_T::NTILE);
    return StorageWeight::getSize(NPad, KPad, jblas::gemm::getWeightSize(_GemmCore_T::TYPE));
  }

  StorageWeight* createStorage(int n, int k, int8_t* workspace) {
    int KPad = utils::padto(k, _GemmCore_T::KTILE);
    int NPad = utils::padto(n, _GemmCore_T::NTILE);
    auto ptr = new StorageWeight(_GemmCore_T::TYPE);
    ptr->resize(NPad, KPad, workspace);
    return ptr;
  }

  void packWeightTranspose(const int N, const int K, const Param& _param) {
    utils::aligned_vector<WType> B_NT(N * K);
    WeightBase<WType, ISA_T>::transposeWeight(N, K, _param.B, _param.ldb, B_NT.data(), N);
    return packWeight(N, K, {B_NT.data(), N, _param.packedW});
  }

  // from KxN int8 symmetric weight to packed N//NtilexKPadxNTile int4 weight
  void packWeight(const int N, const int K, const Param& _param) {
    auto _para = createParallel(N, K);
    utils::CpuBase cb;
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      launch(_param, tidx, _para);
    }
  }

  void launch(const Param& _param, int tidx, const Parallel& para) {
    int colidx, rowidx, rowsize, colsize;
    para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
    auto packedw = _param.packedW;
    if (rowsize > 0 && colsize > 0) {
      int rowremain = utils::remainsize(rowidx, para.mRows,
                                        rowsize);  // rowremain: src valid size. rowsize: padded size
      int colremain = utils::remainsize(colidx, para.mCols, colsize);
      const auto src = _param.B + rowidx * _param.ldb + colidx;
      const auto dst = packedw->template getPtr<WType>() + rowidx * _GemmCore_T::NTILE + colidx * packedw->mKPad;
      using PaddingInterleaveMNWType = kernel::wrapper::PaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW>;
      auto ret = PaddingInterleaveMNWType::template forward<ISA_T>(  //
          src, dst, rowremain, colremain, rowsize, colsize, _param.ldb, packedw->mKPad);
      assert(ret == JblasSuccess);
    }
  }

  inline JBLAS_CODE getWeight(WType** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                              const Param param) {
    auto wptr = param.packedW;
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->template getPtr<WType>() + n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    kernel::wrapper::Memcpy2D::template forward<ISA_T, WType, WType>(
        bptr, *dstptr, n_size / _GemmCore_T::NTILE, _GemmCore_T::NTILE * k_size, _GemmCore_T::NTILE * KPad,
        _GemmCore_T::NTILE * k_size);
    *dststep = k_size;
    return JblasSuccess;
  }
};

}  // namespace gemm
class PackedWeightParser {
 public:
  static PackedWeight* deserialBuffer(void* serialized_buf, int memalloc = 0) {
    auto rptr = reinterpret_cast<int8_t*>(serialized_buf);
    rptr += PackedWeight::TypeOffset;
    int mType = utils::deserialize<int>(rptr);
    if (mType >= int(WeightPrologueType::Begin) && mType < int(WeightPrologueType::End)) {
      rptr = reinterpret_cast<int8_t*>(serialized_buf);
      auto type = static_cast<WeightPrologueType>(mType);
      switch (type) {
        case jblas::prologue::WeightPrologueType::WeightPack: {
          auto ptr = new gemm::StorageWeight(jblas::gemm::GemmCoreType::Undef);
          ptr->deserializeBuffer(rptr, memalloc);
          return ptr;
        }
        default:
          return nullptr;
      }
    }
    return nullptr;
  }
};

}  // namespace prologue
}  // namespace jblas
