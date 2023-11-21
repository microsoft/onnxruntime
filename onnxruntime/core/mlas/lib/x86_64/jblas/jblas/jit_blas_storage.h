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
#include "jit_base.h"
#include "jit_blas.h"
#include "jit_blas_gemm.h"
#include "jit_blas_utils.h"

namespace jblas {
namespace storage {

constexpr size_t Alignment = 64;
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
    return reinterpret_cast<T*>(mBufPtr);
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
    totalsize += mBufSize + Alignment;
    return totalsize;
  }
  virtual void serializeToBuffer(int8_t*& wptr) override {
    utils::serialize(wptr, mBufSize);
    wptr = utils::pointer_align<Alignment>(wptr);
    if (wptr != mBufPtr) {
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
    rptr = utils::pointer_align<Alignment>(rptr);
    mBufPtr = rptr;
    rptr += mBufSize;
  }

  int8_t* mBufPtr = NULL;
  size_t mBufSize = 0;
};
namespace gemm {
// Storage classes for GEMM cases:
// Weight K*N
// Activation M*K

class WeightBase : public storage::ISerializable {
 public:
  JBLAS_PROLOGUEB_IDS mPrologueID = JBLAS_PROLOGUEB_IDS::Undef;
  uint32_t mCoreId = 0;
  JBLAS_DTYPE mDType = JBLAS_DTYPE::F32;
  int mNPad = 0, mKPad = 0;
  int mN = 0, mK = 0;

  WeightBase(uint32_t _id) { mCoreId = _id; }

  // bytes offset to mPrologueID
  static constexpr inline size_t offset() { return sizeof(mSize); }

 protected:
  void resize(int NPad, int KPad, int N, int K, JBLAS_DTYPE dtype) {
    mNPad = NPad;
    mKPad = KPad;
    mN = N;
    mK = K;
    mDType = dtype;
  }

  virtual size_t getSerializedSize() { return ISerializable::getSerializedSize() + getMiscSize(); }

  virtual void serializeToBuffer(int8_t*& wptr) {
    ISerializable::serializeToBuffer(wptr);
    utils::serialize(wptr, mPrologueID);
    utils::serialize(wptr, mCoreId);
    utils::serialize(wptr, mNPad);
    utils::serialize(wptr, mKPad);
    utils::serialize(wptr, mN);
    utils::serialize(wptr, mK);
    utils::serialize(wptr, mDType);
  }

  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) {
    ISerializable::deserializeBuffer(rptr, map_buf);
    if (!map_buf) {
      mPrologueID = utils::deserialize<JBLAS_PROLOGUEB_IDS>(rptr);
      mCoreId = utils::deserialize<uint32_t>(rptr);
      mNPad = utils::deserialize<int>(rptr);
      mKPad = utils::deserialize<int>(rptr);
      mN = utils::deserialize<int>(rptr);
      mK = utils::deserialize<int>(rptr);
      mDType = utils::deserialize<JBLAS_DTYPE>(rptr);
    } else {
      utils::serialize<JBLAS_PROLOGUEB_IDS>(rptr, mPrologueID);
      utils::serialize<uint32_t>(rptr, mCoreId);
      utils::serialize<int>(rptr, mNPad);
      utils::serialize<int>(rptr, mKPad);
      utils::serialize<int>(rptr, mN);
      utils::serialize<int>(rptr, mK);
      utils::serialize<JBLAS_DTYPE>(rptr, mDType);
    }
  }

  inline constexpr size_t getMiscSize() {
    size_t totalsize = 0;
    totalsize += sizeof(mPrologueID);
    totalsize += sizeof(mCoreId);
    totalsize += sizeof(mNPad);
    totalsize += sizeof(mKPad);
    totalsize += sizeof(mN);
    totalsize += sizeof(mK);
    totalsize += sizeof(mDType);
    return totalsize;
  }
};

class WeightKBlockBase : public WeightBase {
 public:
  int mBlockSize = 1;
  WeightKBlockBase(uint32_t _id) : WeightBase(_id) {}
  void resize(int NPad, int KPad, int Block, int N, int K, JBLAS_DTYPE dtype) {
    WeightBase::resize(NPad, KPad, N, K, dtype);
    mBlockSize = Block;
  }

 protected:
  virtual size_t getSerializedSize() {
    size_t totalsize = WeightBase::getSerializedSize() + getMiscSize();
    return totalsize;
  }

  virtual void serializeToBuffer(int8_t*& wptr) {
    WeightBase::serializeToBuffer(wptr);
    utils::serialize(wptr, mBlockSize);
  }

  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) {
    WeightBase::deserializeBuffer(rptr, map_buf);
    if (!map_buf) {
      mBlockSize = utils::deserialize<int>(rptr);
    } else {
      utils::serialize(rptr, mBlockSize);
    }
  }

  inline constexpr size_t getMiscSize() {
    size_t totalsize = sizeof(mBlockSize);
    return totalsize;
  }
};

class StorageQuantCorrection : public ISerialObject {
  // ser
 public:
  size_t mCSize = 0;
  int mCStep = 0;
  bool mIsAsym = false;
  bool mHasReduce = false;
  JBLAS_DTYPE mScaT = JBLAS_DTYPE::F32, mZpT = JBLAS_DTYPE::F32, mRedT = JBLAS_DTYPE::F32;

 protected:
  int8_t* mSPtr = nullptr;
  int8_t* mZPtr = nullptr;
  int8_t* mRPtr = nullptr;

  // non-ser
 public:
  int mScaEleSize = 0, mZpEleSize = 0, mRedEleSize = 0;

 public:
  template <typename T>
  inline T* SPtr() {
    return (T*)mSPtr;
  }

  template <typename T>
  inline T* ZPtr() {
    return (T*)mZPtr;
  }

  template <typename T>
  inline T* RPtr() {
    return (T*)mRPtr;
  }

  size_t resize(int Rows, int Step, JBLAS_DTYPE scalet, JBLAS_DTYPE zpt, JBLAS_DTYPE redt, bool _is_asym,
                bool _has_reduce) {
    mScaT = scalet;
    mZpT = zpt;
    mRedT = redt;
    updateSize();
    mIsAsym = _is_asym;
    mHasReduce = _has_reduce;
    mCStep = Step;
    mCSize = static_cast<size_t>(Rows) * Step;
    return getSerializedSize();
  }

 protected:
  inline void updateSize() {
    mScaEleSize = int(utils::jblas_dtype_size(mScaT));
    mZpEleSize = int(utils::jblas_dtype_size(mZpT));
    mRedEleSize = int(utils::jblas_dtype_size(mRedT));
  }

  inline constexpr size_t getMiscSize() {
    size_t totalsize = 0;
    totalsize += sizeof(mScaT);
    totalsize += sizeof(mZpT);
    totalsize += sizeof(mRedT);
    totalsize += sizeof(mIsAsym);
    totalsize += sizeof(mHasReduce);
    totalsize += sizeof(mCStep);
    totalsize += sizeof(mCSize);
    return totalsize;
  }
  virtual size_t getSerializedSize() override {
    size_t totalsize = getMiscSize();
    totalsize += mCSize * mScaEleSize + Alignment;
    if (mIsAsym) totalsize += mCSize * mZpEleSize + Alignment;
    if (mHasReduce) totalsize += mCSize * mRedEleSize + Alignment;
    return totalsize;
  }
  virtual void serializeToBuffer(int8_t*& wptr) override {
    utils::serialize(wptr, mScaT);
    utils::serialize(wptr, mZpT);
    utils::serialize(wptr, mRedT);
    utils::serialize(wptr, mIsAsym);
    utils::serialize(wptr, mHasReduce);
    utils::serialize(wptr, mCStep);
    utils::serialize(wptr, mCSize);
    wptr = utils::pointer_align<Alignment>(wptr);
    if (wptr != mSPtr) {
      std::memcpy(wptr, mSPtr, mScaEleSize);
    }
    wptr += mCSize * mScaEleSize;
    if (mIsAsym) {
      wptr = utils::pointer_align<Alignment>(wptr);
      if (wptr != mZPtr) {
        std::memcpy(wptr, mZPtr, mZpEleSize);
      }
      wptr += mCSize * mZpEleSize;
    }
    if (mHasReduce) {
      wptr = utils::pointer_align<Alignment>(wptr);
      if (wptr != mRPtr) {
        std::memcpy(wptr, mRPtr, mCSize * mRedEleSize);
      }
      wptr += mCSize * mRedEleSize;
    }
  }
  virtual void deserializeBuffer(int8_t*& rptr, bool locate_buf) override {
    if (!locate_buf) {
      mScaT = utils::deserialize<JBLAS_DTYPE>(rptr);
      mZpT = utils::deserialize<JBLAS_DTYPE>(rptr);
      mRedT = utils::deserialize<JBLAS_DTYPE>(rptr);
      updateSize();
      mIsAsym = utils::deserialize<bool>(rptr);
      mHasReduce = utils::deserialize<bool>(rptr);
      mCStep = utils::deserialize<int>(rptr);
      mCSize = utils::deserialize<size_t>(rptr);
    } else {
      utils::serialize<JBLAS_DTYPE>(rptr, mScaT);
      utils::serialize<JBLAS_DTYPE>(rptr, mZpT);
      utils::serialize<JBLAS_DTYPE>(rptr, mRedT);
      utils::serialize<bool>(rptr, mIsAsym);
      utils::serialize<bool>(rptr, mHasReduce);
      utils::serialize<int>(rptr, mCStep);
      utils::serialize<size_t>(rptr, mCSize);
    }
    rptr = utils::pointer_align<Alignment>(rptr);
    mSPtr = rptr;
    rptr += mCSize * mScaEleSize;
    if (mIsAsym) {
      rptr = utils::pointer_align<Alignment>(rptr);
      mZPtr = rptr;
      rptr += mCSize * mZpEleSize;
    }
    if (mHasReduce) {
      rptr = utils::pointer_align<Alignment>(rptr);
      mRPtr = rptr;
      rptr += mCSize * mRedEleSize;
    }
  }
};

class StorageReduce : public ISerializable, public ISerialBuffer {
 public:
  using CorrectionType = StorageQuantCorrection;
  int m = 0, k = 0, lda = 0, kblock = 1;
  size_t resize(int _m, int _k, int _kblock, JBLAS_DTYPE redt) {
    kblock = _kblock;
    m = _m;
    k = _k;
    lda = utils::updiv(_k, _kblock);
    size_t bufsize = static_cast<size_t>(m) * lda * utils::jblas_dtype_size(redt);
    ISerialBuffer::resize(bufsize);
    mSize = getSerializedSize();
    return mSize;
  }
  template <typename QT_T>
  inline QT_T* APtr() {
    return get<QT_T>();
  }

  virtual void assign(int8_t* buf) override {
    ISerializable::deserializeBuffer(buf, true);
    deserializeBuffer(buf, true);
    ISerialBuffer::deserializeBuffer(buf, true);
  }

  virtual void serialize(int8_t* wptr) {
    ISerializable::serializeToBuffer(wptr);
    serializeToBuffer(wptr);
    ISerialBuffer::serializeToBuffer(wptr);
  }

  virtual void deserialize(int8_t* rptr) override {
    ISerializable::deserializeBuffer(rptr, false);
    deserializeBuffer(rptr, false);
    ISerialBuffer::deserializeBuffer(rptr, false);
  }

 protected:
  virtual size_t getSerializedSize() {
    return ISerializable::getSerializedSize() + getMiscSize() + ISerialBuffer::getSerializedSize();
  }

  virtual void serializeToBuffer(int8_t*& wptr) {
    utils::serialize(wptr, m);
    utils::serialize(wptr, k);
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
      utils::serialize(rptr, k);
      utils::serialize(rptr, lda);
      utils::serialize(rptr, kblock);
    }
  }

  inline constexpr size_t getMiscSize() {
    size_t totalsize = 0;
    totalsize += sizeof(m);
    totalsize += sizeof(k);
    totalsize += sizeof(lda);
    totalsize += sizeof(kblock);
    return totalsize;
  }
};

class StorageQuantActivation : public ISerializable, public ISerialBuffer, public StorageQuantCorrection {
 public:
  using CorrectionType = StorageQuantCorrection;
  int m = 0, lda = 0, kblock = 1;
  size_t resize(int _m, int _lda, int _kblock, JBLAS_DTYPE buft, JBLAS_DTYPE scalet, JBLAS_DTYPE zpt, JBLAS_DTYPE redt,
                bool is_asym, bool has_reduce) {
    kblock = _kblock;
    lda = _lda;
    m = _m;
    CorrectionType::resize(_m, utils::updiv(_lda, _kblock), scalet, zpt, redt, is_asym, has_reduce);
    size_t bufsize = static_cast<size_t>(m) * lda * utils::jblas_dtype_size(buft);
    ISerialBuffer::resize(bufsize);
    mSize = getSerializedSize();
    return mSize;
  }
  template <typename QT_T>
  inline QT_T* APtr() {
    return get<QT_T>();
  }

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

class StoragePackedWeight : public WeightBase, public ISerialBuffer {
 public:
  StoragePackedWeight(uint32_t _id) : WeightBase(_id) { mPrologueID = JBLAS_PROLOGUEB_IDS::WeightPack; }

  size_t resize(int NPad, int KPad, int N, int K, JBLAS_DTYPE dtype) {
    WeightBase::resize(NPad, KPad, N, K, dtype);
    auto bsize = static_cast<size_t>(NPad) * KPad * jblas::utils::jblas_dtype_size(dtype);
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

class Buffer8Bit : public ISerialBuffer {
 public:
  void resize(size_t size) { ISerialBuffer::resize(size); }
  inline int8_t* WPtr() { return get<int8_t>(); }
};

class Buffer4Bit : public ISerialBuffer {
 public:
  void resize(size_t size) { ISerialBuffer::resize(utils::updiv(size, 2)); }
  inline utils::bit4x2* WPtr() { return get<utils::bit4x2>(); }
};

class StorageWeightKBlockS8 : public WeightKBlockBase, public Buffer8Bit, public StorageQuantCorrection {
 public:
  using InfoType = WeightKBlockBase;
  using QWeightType = Buffer8Bit;
  using CorrectionType = StorageQuantCorrection;
  StorageWeightKBlockS8(uint32_t _type) : WeightKBlockBase(_type) { mPrologueID = JBLAS_PROLOGUEB_IDS::WeightKBlockS8; }

  size_t resize(int NPad, int KPad, int Block, int N, int K, JBLAS_DTYPE scalet, JBLAS_DTYPE redt, bool IsAsym) {
    JBLAS_DTYPE zpt = JBLAS_DTYPE::S8;
    InfoType::resize(NPad, KPad, Block, N, K, JBLAS_DTYPE::S8);
    QWeightType::resize(static_cast<size_t>(NPad) * KPad);
    int nk_scale = utils::updiv(KPad, Block);
    auto gemm_comp = jblas::gemm::CoreAttr::get_mask_val(mCoreId, jblas::gemm::CoreAttr::COMP_MASK,
                                                         jblas::gemm::CoreAttr::COMP_SHIFT);
    CorrectionType::resize(nk_scale, NPad, scalet, zpt, redt, IsAsym,
                           gemm_comp >= static_cast<uint32_t>(jblas::gemm::CompType::COMP_INT_START));
    mSize = InfoType::getSerializedSize() + QWeightType::getSerializedSize() + CorrectionType::getSerializedSize();
    return mSize;
  }

  virtual void assign(int8_t* buf) override {
    InfoType::deserializeBuffer(buf, true);
    QWeightType::deserializeBuffer(buf, true);
    CorrectionType::deserializeBuffer(buf, true);
  }

  virtual void serialize(int8_t* wptr) {
    InfoType::serializeToBuffer(wptr);
    QWeightType::serializeToBuffer(wptr);
    CorrectionType::serializeToBuffer(wptr);
  }

  virtual void deserialize(int8_t* rptr) override {
    InfoType::deserializeBuffer(rptr, false);
    QWeightType::deserializeBuffer(rptr, false);
    CorrectionType::deserializeBuffer(rptr, false);
  }
};

class StorageWeightKBlockS4 : public WeightKBlockBase, public Buffer4Bit, public StorageQuantCorrection {
 public:
  using InfoType = WeightKBlockBase;
  using QWeightType = Buffer4Bit;
  using CorrectionType = StorageQuantCorrection;
  StorageWeightKBlockS4(uint32_t _type) : WeightKBlockBase(_type) { mPrologueID = JBLAS_PROLOGUEB_IDS::WeightKBlockS4; }

  size_t resize(int NPad, int KPad, int Block, int N, int K, JBLAS_DTYPE s4t, JBLAS_DTYPE scalet, JBLAS_DTYPE redt,
                bool IsAsym) {
    JBLAS_DTYPE zpt = JBLAS_DTYPE::S8;
    InfoType::resize(NPad, KPad, Block, N, K, s4t);
    QWeightType::resize(static_cast<size_t>(NPad) * KPad);
    int nk_scale = utils::updiv(KPad, Block);
    auto gemm_comp = jblas::gemm::CoreAttr::get_mask_val(mCoreId, jblas::gemm::CoreAttr::COMP_MASK,
                                                         jblas::gemm::CoreAttr::COMP_SHIFT);
    CorrectionType::resize(nk_scale, NPad, scalet, zpt, redt, IsAsym,
                           gemm_comp >= static_cast<uint32_t>(jblas::gemm::CompType::COMP_INT_START));
    mSize = InfoType::getSerializedSize() + QWeightType::getSerializedSize() + CorrectionType::getSerializedSize();
    return mSize;
  }

  virtual void assign(int8_t* buf) override {
    InfoType::deserializeBuffer(buf, true);
    QWeightType::deserializeBuffer(buf, true);
    CorrectionType::deserializeBuffer(buf, true);
  }

  virtual void serialize(int8_t* wptr) {
    InfoType::serializeToBuffer(wptr);
    QWeightType::serializeToBuffer(wptr);
    CorrectionType::serializeToBuffer(wptr);
  }

  virtual void deserialize(int8_t* rptr) override {
    InfoType::deserializeBuffer(rptr, false);
    QWeightType::deserializeBuffer(rptr, false);
    CorrectionType::deserializeBuffer(rptr, false);
  }
};

class StorageWeightKBlockF4 : public StorageWeightKBlockS4 {
 public:
  StorageWeightKBlockF4(uint32_t _type) : StorageWeightKBlockS4(_type) {
    mPrologueID = JBLAS_PROLOGUEB_IDS::WeightKBlockF4;
  }

  size_t resize(int NPad, int KPad, int Block, int N, int K, JBLAS_DTYPE f4t, JBLAS_DTYPE scalet) {
    StorageWeightKBlockS4::InfoType::resize(NPad, KPad, Block, N, K, f4t);
    StorageWeightKBlockS4::QWeightType::resize((size_t)NPad * KPad);
    int nk_scale = utils::updiv(KPad, Block);
    StorageWeightKBlockS4::CorrectionType::resize(nk_scale, NPad, scalet, JBLAS_DTYPE::S8, JBLAS_DTYPE::F32, false,
                                                  false);
    mSize = StorageWeightKBlockS4::InfoType::getSerializedSize() +
            StorageWeightKBlockS4::QWeightType::getSerializedSize() +
            StorageWeightKBlockS4::CorrectionType::getSerializedSize();
    return mSize;
  }
};

class PackedWeightParser {
 public:
  static gemm::WeightBase* deserialBuffer(void* serialized_buf) {
    auto rptr = reinterpret_cast<int8_t*>(serialized_buf);
    rptr += WeightBase::offset();
    int mProID = utils::deserialize<int>(rptr);
    WeightBase* ptr = NULL;
    if (mProID >= int(JBLAS_PROLOGUEB_IDS::Begin) && mProID < int(JBLAS_PROLOGUEB_IDS::End)) {
      rptr = reinterpret_cast<int8_t*>(serialized_buf);
      auto type = static_cast<JBLAS_PROLOGUEB_IDS>(mProID);
      switch (type) {
        case JBLAS_PROLOGUEB_IDS::WeightPack:
          ptr = new gemm::StoragePackedWeight(0);
          break;
        case JBLAS_PROLOGUEB_IDS::WeightKBlockS8:
          ptr = new gemm::StorageWeightKBlockS8(0);
          break;
        case JBLAS_PROLOGUEB_IDS::WeightKBlockS4:
          ptr = new gemm::StorageWeightKBlockS4(0);
          break;
        case JBLAS_PROLOGUEB_IDS::WeightKBlockF4:
          ptr = new gemm::StorageWeightKBlockF4(0);
          break;
        default:
          break;
      }
      if (ptr) {
        ptr->deserialize(rptr);
      }
    }
    return ptr;
  }
};
}  // namespace gemm
}  // namespace storage
}  // namespace jblas
