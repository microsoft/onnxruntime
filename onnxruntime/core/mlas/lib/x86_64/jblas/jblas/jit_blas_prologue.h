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
enum class PrologueBIDs : int {
  Undef = -1,
  Begin = 0,
  WeightPack = Begin,
  End,
};

class ISerialObject {
 protected:
  virtual size_t getSerializedSize() = 0;

  virtual void serializeToBuffer(int8_t*& wptr) = 0;

  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) = 0;
};

class ISerializable : public ISerialObject {
 public:
  virtual ~ISerializable() = default;

  virtual void assign(int8_t* buf) = 0;

  virtual void serialize(int8_t* wptr) = 0;

  virtual void deserialize(int8_t* rptr) = 0;
  size_t mSize = 0;

 protected:
  virtual size_t getSerializedSize() override {
    size_t totalsize = 0;
    totalsize += sizeof(mSize);
    return totalsize;
  }
  virtual void serializeToBuffer(int8_t*& wptr) override { utils::serialize(wptr, mSize); }
  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) override {
    if (!map_buf) {
      mSize = utils::deserialize<size_t>(rptr);
    } else {
      utils::serialize<size_t>(rptr, mSize);
    }
  }
};

class ISerialBuffer : public ISerialObject {
 public:
  template <typename T>
  inline constexpr T* get() {
    return (T*)mBufPtr;
  };
  template <typename T>
  inline size_t size() {
    return mBufSize / sizeof(T);
  };

  void resize(size_t bytes) { mBufSize = bytes; }

 protected:
  virtual size_t getSerializedSize() override {
    size_t totalsize = 0;
    totalsize += sizeof(mBufSize);
    totalsize += mBufSize;
    return totalsize;
  }
  virtual void serializeToBuffer(int8_t*& wptr) override {
    utils::serialize(wptr, mBufSize);
    if (wptr != (int8_t*)mBufPtr) {
      std::memcpy(wptr, mBufPtr, mBufSize);
    }
    wptr += mBufSize;
  }
  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) override {
    if (!map_buf) {
      mBufSize = utils::deserialize<size_t>(rptr);
    } else {
      utils::serialize<size_t>(rptr, mBufSize);
    }
    mBufPtr = (int8_t*)rptr;
    rptr += mBufSize;
  }

  int8_t* mBufPtr = NULL;
  size_t mBufSize = 0;
};

namespace gemm {

class WeightBase : public ISerializable {
 public:
  int mPrologueID = -1;
  // To many dummy codes to make these read-only, just use these members
  jblas::gemm::GemmCoreType mCoreType = jblas::gemm::GemmCoreType::Undef;
  int mNPad = 0, mKPad = 0;

  WeightBase(jblas::gemm::GemmCoreType _type) {
    mNPad = 0;
    mKPad = 0;
    mCoreType = _type;
  }

  // bytes offset to mPrologueID
  static constexpr inline size_t offset() { return sizeof(mSize); }

 protected:
  void resize(int NPad, int KPad) {
    mNPad = NPad;
    mKPad = KPad;
  }

  virtual size_t getSerializedSize() { return ISerializable::getSerializedSize() + getMiscSize(); }

  virtual void serializeToBuffer(int8_t*& wptr) {
    ISerializable::serializeToBuffer(wptr);
    utils::serialize(wptr, mPrologueID);
    utils::serialize(wptr, mCoreType);
    utils::serialize(wptr, mNPad);
    utils::serialize(wptr, mKPad);
  }

  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) {
    ISerializable::deserializeBuffer(rptr, map_buf);
    if (!map_buf) {
      mPrologueID = utils::deserialize<int>(rptr);
      mCoreType = utils::deserialize<jblas::gemm::GemmCoreType>(rptr);
      mNPad = utils::deserialize<int>(rptr);
      mKPad = utils::deserialize<int>(rptr);
    } else {
      utils::serialize<int>(rptr, mPrologueID);
      utils::serialize<jblas::gemm::GemmCoreType>(rptr, mCoreType);
      utils::serialize<int>(rptr, mNPad);
      utils::serialize<int>(rptr, mKPad);
    }
  }

  inline constexpr size_t getMiscSize() {
    size_t totalsize = 0;
    totalsize += sizeof(mPrologueID);
    totalsize += sizeof(mCoreType);
    totalsize += sizeof(mNPad);
    totalsize += sizeof(mKPad);
    return totalsize;
  }
};

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

template <typename SCA_T, typename DST_T, typename RED_T>
class StorageWeightCorrection : public ISerialObject {
 public:
  SCA_T* mSPtr = nullptr;
  DST_T* mZPtr = nullptr;
  RED_T* mRPtr = nullptr;
  size_t mCSize = 0;
  int mCStep = 0;
  bool mIsAsym = false;
  bool mHasReduce = false;

  size_t resize(int NPad, int KBlks, bool _is_asym = false, bool _has_reduce = true) {
    mIsAsym = _is_asym;
    mHasReduce = _has_reduce;
    mCStep = NPad;
    mCSize = (size_t)NPad * KBlks;
    return getSerializedSize();
  }

 protected:
  inline constexpr size_t getMiscSize() {
    size_t totalsize = 0;
    totalsize += sizeof(mIsAsym);
    totalsize += sizeof(mHasReduce);
    totalsize += sizeof(mCStep);
    totalsize += sizeof(mCSize);
    return totalsize;
  }
  virtual size_t getSerializedSize() override {
    size_t totalsize = getMiscSize();
    totalsize += mCSize * sizeof(mSPtr[0]);
    if (mIsAsym) totalsize += mCSize * sizeof(mZPtr[0]);
    if (mHasReduce) totalsize += mCSize * sizeof(mRPtr[0]);
    return totalsize;
  }
  virtual void serializeToBuffer(int8_t*& wptr) override {
    utils::serialize(wptr, mIsAsym);
    utils::serialize(wptr, mHasReduce);
    utils::serialize(wptr, mCStep);
    utils::serialize(wptr, mCSize);
    if (wptr != (int8_t*)mSPtr) {
      std::memcpy(wptr, mSPtr, mCSize * sizeof(mSPtr[0]));
    }
    wptr += mCSize * sizeof(mSPtr[0]);
    if (mIsAsym) {
      if (wptr != (int8_t*)mZPtr) {
        std::memcpy(wptr, mZPtr, mCSize * sizeof(mZPtr[0]));
      }
      wptr += mCSize * sizeof(mZPtr[0]);
    }
    if (mHasReduce) {
      if (wptr != (int8_t*)mRPtr) {
        std::memcpy(wptr, mRPtr, mCSize * sizeof(mRPtr[0]));
      }
      wptr += mCSize * sizeof(mRPtr[0]);
    }
  }
  virtual void deserializeBuffer(int8_t*& rptr, bool locate_buf) override {
    if (!locate_buf) {
      mIsAsym = utils::deserialize<bool>(rptr);
      mHasReduce = utils::deserialize<bool>(rptr);
      mCStep = utils::deserialize<int>(rptr);
      mCSize = utils::deserialize<size_t>(rptr);
    } else {
      utils::serialize<bool>(rptr, mIsAsym);
      utils::serialize<bool>(rptr, mHasReduce);
      utils::serialize<int>(rptr, mCStep);
      utils::serialize<size_t>(rptr, mCSize);
    }
    mSPtr = reinterpret_cast<SCA_T*>(rptr);
    rptr += mCSize * sizeof(mSPtr[0]);
    if (mIsAsym) {
      mZPtr = reinterpret_cast<DST_T*>(rptr);
      rptr += mCSize * sizeof(mZPtr[0]);
    }
    if (mHasReduce) {
      mRPtr = reinterpret_cast<RED_T*>(rptr);
      rptr += mCSize * sizeof(mRPtr[0]);
    }
  }
};

template <typename SCA_T, typename DST_T, typename RED_T>
class StorageActivationCorrection : public StorageWeightCorrection<SCA_T, DST_T, RED_T> {
 public:
  size_t resize(int _m, int _kblks, bool _is_asym = false, bool _has_reduce = true) {
    this->mIsAsym = _is_asym;
    this->mHasReduce = _has_reduce;
    this->mCStep = _kblks;
    this->mCSize = (size_t)_m * _kblks;
    return this->getSerializedSize();
  }
};

template <typename QT_T, typename ST_T>
class StorageQuantActivation : public ISerializable,
                               public ISerialBuffer,
                               public StorageActivationCorrection<ST_T, QT_T, float> {
 public:
  using CorrectionType = StorageActivationCorrection<ST_T, QT_T, float>;
  int m = 0, lda = 0, kblock = 1;
  size_t resize(int _m, int _lda, int _kblock, bool is_asym) {
    kblock = _kblock;
    lda = _lda;
    m = _m;
    CorrectionType::resize(_m, utils::updiv(_lda, _kblock), is_asym, true);
    size_t bufsize = size_t(m) * lda * sizeof(QT_T);
    ISerialBuffer::resize(bufsize);
    mSize = getSerializedSize();
    return mSize;
  }

  inline QT_T* APtr() { return get<QT_T>(); }

  virtual void assign(int8_t* buf) override {
    ISerializable::deserializeBuffer(buf, true);
    deserializeBuffer(buf, true);
    ISerialBuffer::deserializeBuffer(buf, true);
    CorrectionType::deserializeBuffer(buf, true);
  }

  virtual void serialize(int8_t* wptr) {
    ISerializable::serializeToBuffer(wptr);
    serializeToBuffer(wptr);
    ISerialBuffer::serializeToBuffer(wptr);
    CorrectionType::serializeToBuffer(wptr);
  }

  virtual void deserialize(int8_t* rptr) override {
    ISerializable::deserializeBuffer(rptr, false);
    deserializeBuffer(rptr, false);
    ISerialBuffer::deserializeBuffer(rptr, false);
    CorrectionType::deserializeBuffer(rptr, false);
  }

 protected:
  virtual size_t getSerializedSize() {
    return ISerializable::getSerializedSize() + getMiscSize() + ISerialBuffer::getSerializedSize() +
           CorrectionType::getSerializedSize();
  }

  virtual void serializeToBuffer(int8_t*& wptr) {
    utils::serialize(wptr, m);
    utils::serialize(wptr, lda);
    utils::serialize(wptr, kblock);
  }

  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) {
    if (!map_buf) {
      m = utils::deserialize<int>(rptr);
      lda = utils::deserialize<int>(rptr);
      kblock = utils::deserialize<int>(rptr);
    } else {
      utils::serialize(rptr, m);
      utils::serialize(rptr, lda);
      utils::serialize(rptr, kblock);
    }
  }

  inline constexpr size_t getMiscSize() {
    size_t totalsize = 0;
    totalsize += sizeof(m);
    totalsize += sizeof(lda);
    totalsize += sizeof(kblock);
    return totalsize;
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T, typename SRC_T>
class ActivationU8KBlockQuantize {
 public:
  using AType = typename _GemmCore_T::AType;
  using SType = float;
  using QParam = StorageQuantActivation<AType, SType>;
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

  inline QParam createStorage(int m, int k, int kblock) {
    QParam tmp;
    int kpad = utils::padto(k, _GemmCore_T::KTILE);
    tmp.resize(m, kpad, kblock, true);
    return tmp;
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
      auto thdqptr = quan->APtr() + rowidx * quan->lda + colidx;
      auto thdsptr = quan->mSPtr + rowidx * quan->mCStep + blkidx;
      auto thdzptr = quan->mZPtr + rowidx * quan->mCStep + blkidx;
      kernel::wrapper::QuantizeU8ColBlock::template forward<ISA_T, SRC_T>(
          rowremain, colremain, srcptr, _param.lda, thdqptr, quan->lda, thdsptr, quan->mCStep, thdzptr, para.mColBlock);
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
    (void)m_size;
    (void)k_size;
    auto quan = _param.quan;
    auto aptr = const_cast<AType*>(quan->APtr());
    *dstptr = aptr + m_offset * _param.lda + k_offset;
    *dststep = _param.lda;
    return JblasSuccess;
  }

  JBLAS_CODE getScale(SType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                      int k_offset) {
    (void)m_size;
    (void)k_size;
    auto quan = _param.quan;
    auto ptr = const_cast<SType*>(quan->mSPtr);
    *dstptr = ptr + m_offset * quan->mCStep + k_offset / quan->kblock;
    *dststep = quan->mCStep;
    return JblasSuccess;
  }

  static inline JBLAS_CODE getZp(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size,
                                 int m_offset, int k_offset) {
    (void)m_size;
    (void)k_size;
    auto quan = _param.quan;
    *dstptr = &quan->mZPtr[(m_offset)*quan->mCStep + k_offset / quan->kblock];
    *dststep = quan->mCStep;
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

  inline QParam createStorage(int m, int k) {
    QParam tmp;
    int kpad = utils::padto(k, _GemmCore_T::KTILE);
    tmp.resize(m, kpad, kpad, true);
    return tmp;
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
      auto thdqptr = quan->APtr() + rowidx * quan->lda + colidx;
      auto thdsptr = quan->mSPtr + rowidx * quan->mCStep;
      auto thdzptr = quan->mZPtr + rowidx * quan->mCStep;
      kernel::wrapper::QuantizeU8ColBlock::template forward<ISA_T, SRC_T>(
          rowremain, colremain, srcptr, _param.lda, thdqptr, quan->lda, thdsptr, quan->mCStep, thdzptr, para.mCols);
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
    auto aptr = _param.Q->APtr();
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
  using QParam = StorageQuantActivation<AType, SType>;
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

  inline QParam createStorage(int m, int k, int kblock) {
    QParam tmp;
    int kpad = utils::padto(k, _GemmCore_T::KTILE);
    tmp.resize(m, kpad, kblock, false);
    return tmp;
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
      auto thdqptr = quan->APtr() + rowidx * quan->lda + colidx;
      auto thdsptr = quan->mSPtr + rowidx * quan->mCStep + blkidx;
      kernel::wrapper::QuantizeS8ColBlock::template forward<ISA_T, SRC_T>(
          rowremain, colremain, srcptr, _param.lda, thdqptr, quan->lda, thdsptr, quan->mCStep, para.mColBlock);
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
    (void)m_size;
    (void)k_size;
    auto quan = _param.quan;
    auto aptr = const_cast<AType*>(quan->APtr());
    *dstptr = aptr + m_offset * quan->lda + k_offset;
    *dststep = quan->lda;
    return JblasSuccess;
  }

  JBLAS_CODE getScale(SType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                      int k_offset) {
    (void)m_size;
    (void)k_size;
    auto quan = _param.quan;
    auto ptr = const_cast<SType*>(quan->mSPtr);
    *dstptr = ptr + m_offset * quan->mCStep + k_offset / quan->kblock;
    *dststep = quan->mCStep;
    return JblasSuccess;
  }

  static inline JBLAS_CODE getZp(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size,
                                 int m_offset, int k_offset) {
    (void)m_size;
    (void)k_size;
    (void)dstptr;
    (void)dststep;
    (void)_param;
    (void)m_offset;
    (void)k_offset;
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

  inline QParam createStorage(int m, int k) {
    QParam tmp;
    int kpad = utils::padto(k, _GemmCore_T::KTILE);
    tmp.resize(m, kpad, kpad, false);
    return tmp;
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
      auto thdqptr = quan->APtr() + rowidx * quan->lda + colidx;
      auto thdsptr = quan->mSPtr + rowidx * quan->mCStep;
      kernel::wrapper::QuantizeS8ColBlock::template forward<ISA_T, SRC_T>(
          rowremain, colremain, srcptr, _param.lda, thdqptr, quan->lda, thdsptr, quan->mCStep, para.mCols);
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
    auto aptr = const_cast<AType*>(_param.Q->APtr());
    *dstptr = aptr + m_offset * _param.Q->lda + k_offset;
    *dststep = _param.Q->lda;
    return JblasSuccess;
  }
};

template <typename T, JBLAS_ISA ISA_T>
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

template <class _GemmCore_T, JBLAS_ISA ISA_T>
using ActivationFp32SymS8Quantize = ActivationSymS8Quantize<_GemmCore_T, ISA_T, float>;
template <class _GemmCore_T, JBLAS_ISA ISA_T>
using ActivationBf16SymS8Quantize = ActivationSymS8Quantize<_GemmCore_T, ISA_T, utils::bf16>;

// Storage class has real weight memory, PackedWeight provides interface
class StoragePackedWeight : public WeightBase, public ISerialBuffer {
 public:
  StoragePackedWeight(jblas::gemm::GemmCoreType _type) : WeightBase(_type) {
    mPrologueID = int(PrologueBIDs::WeightPack);
  }

  size_t resize(int NPad, int KPad) {
    WeightBase::resize(NPad, KPad);
    auto bsize = (size_t)NPad * KPad * jblas::gemm::getWeightSize(mCoreType);
    ISerialBuffer::resize(bsize);
    mSize = WeightBase::getSerializedSize() + ISerialBuffer::getSerializedSize();
    return mSize;
  }

  virtual void assign(int8_t* buf) override {
    WeightBase::deserializeBuffer(buf, true);
    ISerialBuffer::deserializeBuffer(buf, true);
  }

  virtual void serialize(int8_t* wptr) {
    WeightBase::serializeToBuffer(wptr);
    ISerialBuffer::serializeToBuffer(wptr);
  }

  virtual void deserialize(int8_t* rptr) override {
    WeightBase::deserializeBuffer(rptr, false);
    ISerialBuffer::deserializeBuffer(rptr, false);
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class WeightPack {
 public:
  using WType = typename _GemmCore_T::BType;
  struct Param {
    const WType* B;
    const int ldb;
    StoragePackedWeight* packedW;
  };
  using Parallel = utils::parallel::Parallel2DRowMajor;

  Parallel createParallel(int n, int k) {
    Parallel _paral;
    auto cb = utils::CpuBase();
    _paral.update(k, n, _GemmCore_T::KTILE, _GemmCore_T::NTILE, cb.mNumThreads);
    return _paral;
  }

  StoragePackedWeight createStorage(int n, int k) {
    int KPad = utils::padto(k, _GemmCore_T::KTILE);
    int NPad = utils::padto(n, _GemmCore_T::NTILE);
    StoragePackedWeight tmp(_GemmCore_T::TYPE);
    tmp.resize(NPad, KPad);
    return tmp;
  }

  void packWeightTranspose(const int N, const int K, const Param& _param) {
    utils::aligned_vector<WType> B_NT(N * K);
    transposeWeight<WType, ISA_T>(N, K, _param.B, _param.ldb, B_NT.data(), N);
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
      const auto dst = packedw->template get<WType>() + rowidx * _GemmCore_T::NTILE + colidx * packedw->mKPad;
      using PaddingInterleaveMNWType = kernel::wrapper::PaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW>;
      auto ret = PaddingInterleaveMNWType::template forward<ISA_T>(  //
          src, dst, rowremain, colremain, rowsize, colsize, _param.ldb, packedw->mKPad);
      assert(ret == JblasSuccess);
      (void)ret;
    }
  }

  inline JBLAS_CODE getWeight(WType** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                              const Param param) {
    auto wptr = param.packedW;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->template get<WType>() + n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    kernel::wrapper::Memcpy2D::template forward<ISA_T, WType, WType>(
        bptr, *dstptr, n_size / _GemmCore_T::NTILE, _GemmCore_T::NTILE * k_size, _GemmCore_T::NTILE * KPad,
        _GemmCore_T::NTILE * k_size);
    *dststep = k_size;
    return JblasSuccess;
  }
};

class PackedWeightParser {
 public:
  static gemm::WeightBase* deserialBuffer(void* serialized_buf) {
    auto rptr = reinterpret_cast<int8_t*>(serialized_buf);
    rptr += WeightBase::offset();
    int mProID = utils::deserialize<int>(rptr);
    if (mProID >= int(PrologueBIDs::Begin) && mProID < int(PrologueBIDs::End)) {
      rptr = reinterpret_cast<int8_t*>(serialized_buf);
      auto type = static_cast<PrologueBIDs>(mProID);
      switch (type) {
        case jblas::prologue::PrologueBIDs::WeightPack: {
          auto ptr = new gemm::StoragePackedWeight(jblas::gemm::GemmCoreType::Undef);
          ptr->deserialize(rptr);
          return ptr;
        }
        default:
          return nullptr;
      }
    }
    return nullptr;
  }
};
}  // namespace gemm
}  // namespace prologue
}  // namespace jblas
