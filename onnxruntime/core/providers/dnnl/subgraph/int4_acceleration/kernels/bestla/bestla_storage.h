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
#include "bestla.h"
#include "bestla_gemm.h"
#include "bestla_utils.h"

namespace bestla {
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

template <int ALIGN>
class ObjectAlignedBuffer : public ISerialObject {
 public:
  template <typename T>
  inline constexpr T* get() const {
    return reinterpret_cast<T*>(mBufPtr);
  }
  template <typename T>
  inline size_t size() {
    return mBufSize / sizeof(T);
  }

  void resize(size_t bytes) { mBufSize = bytes; }

  // ser
  int8_t* mBufPtr = nullptr;
  size_t mBufSize = 0;
  size_t mBufOffset = 0;

  virtual size_t getSerializedSize() override {
    size_t totalsize = 0;
    totalsize += sizeof(mBufSize);
    totalsize += sizeof(mBufOffset);
    totalsize += mBufSize + ALIGN;
    return totalsize;
  }
  virtual void serializeToBuffer(int8_t*& wptr) override {
    utils::serialize(wptr, mBufSize);
    auto tmpptr = wptr + sizeof(mBufOffset);
    mBufOffset = utils::pointer_align<ALIGN>(tmpptr) - tmpptr;
    utils::serialize(wptr, mBufOffset);
    wptr += mBufOffset;
    if (wptr != mBufPtr) {
      std::memcpy(wptr, mBufPtr, mBufSize);
    }
    wptr += mBufSize;
  }
  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) override {
    if (!map_buf) {
      mBufSize = utils::deserialize<size_t>(rptr);
      mBufOffset = utils::deserialize<size_t>(rptr);
    } else {
      utils::serialize<size_t>(rptr, mBufSize);
      auto tmpptr = rptr + sizeof(mBufOffset);
      mBufOffset = utils::pointer_align<ALIGN>(tmpptr) - tmpptr;
      utils::serialize(rptr, mBufOffset);
    }
    rptr += mBufOffset;
    mBufPtr = rptr;
    rptr += mBufSize;
  }
};

template <int ALIGN>
class ObjectOptionalBuffer : public ObjectAlignedBuffer<ALIGN> {
 public:
  void resize(size_t bytes) {
    ObjectAlignedBuffer<ALIGN>::resize(bytes);
    mNotEmpty = bytes > 0;
  }

  // ser
  bool mNotEmpty{false};

  virtual size_t getSerializedSize() override {
    size_t totalsize = 0;
    totalsize += sizeof(mNotEmpty);
    if (mNotEmpty) {
      totalsize += ObjectAlignedBuffer<ALIGN>::getSerializedSize();
    }
    return totalsize;
  }
  virtual void serializeToBuffer(int8_t*& wptr) override {
    utils::serialize(wptr, mNotEmpty);
    if (mNotEmpty) {
      ObjectAlignedBuffer<ALIGN>::serializeToBuffer(wptr);
    }
  }
  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) override {
    if (!map_buf) {
      mNotEmpty = utils::deserialize<bool>(rptr);
    } else {
      utils::serialize<bool>(rptr, mNotEmpty);
    }
    if (mNotEmpty) {
      ObjectAlignedBuffer<ALIGN>::deserializeBuffer(rptr, map_buf);
    }
  }
};

namespace gemm {

class ObjectQuantCorrection : public ISerialObject {
  // ser
 public:
  size_t mCSize = 0;
  int mCStep = 0;
  BTLA_DTYPE mScaT = BTLA_DTYPE::F32, mZpT = BTLA_DTYPE::F32, mRedT = BTLA_DTYPE::F32;
  ObjectAlignedBuffer<Alignment> mScaleBuf;
  ObjectOptionalBuffer<Alignment> mZpBuf, mRedBuf;

  // non-ser
 public:
  int mScaEleSize = 0, mZpEleSize = 0, mRedEleSize = 0;

  size_t resize(int Rows, int Step, BTLA_DTYPE scalet, BTLA_DTYPE zpt, BTLA_DTYPE redt, bool _is_asym,
                bool _has_reduce) {
    mScaT = scalet;
    mZpT = zpt;
    mRedT = redt;
    updateSize();
    mCStep = Step;
    mCSize = static_cast<size_t>(Rows) * Step;
    mScaleBuf.resize(mCSize * mScaEleSize);
    if (_is_asym) {
      mZpBuf.resize(mCSize * mZpEleSize);
    }
    if (_has_reduce) {
      mRedBuf.resize(mCSize * mRedEleSize);
    }
    return getSerializedSize();
  }

  virtual size_t getSerializedSize() override {
    size_t totalsize = getMiscSize();
    totalsize += mScaleBuf.getSerializedSize();
    totalsize += mZpBuf.getSerializedSize();
    totalsize += mRedBuf.getSerializedSize();
    return totalsize;
  }
  virtual void serializeToBuffer(int8_t*& wptr) override {
    utils::serialize(wptr, mScaT);
    utils::serialize(wptr, mZpT);
    utils::serialize(wptr, mRedT);
    utils::serialize(wptr, mCStep);
    utils::serialize(wptr, mCSize);
    mScaleBuf.serializeToBuffer(wptr);
    mZpBuf.serializeToBuffer(wptr);
    mRedBuf.serializeToBuffer(wptr);
  }
  virtual void deserializeBuffer(int8_t*& rptr, bool locate_buf) override {
    if (!locate_buf) {
      mScaT = utils::deserialize<BTLA_DTYPE>(rptr);
      mZpT = utils::deserialize<BTLA_DTYPE>(rptr);
      mRedT = utils::deserialize<BTLA_DTYPE>(rptr);
      updateSize();
      mCStep = utils::deserialize<int>(rptr);
      mCSize = utils::deserialize<size_t>(rptr);
    } else {
      utils::serialize<BTLA_DTYPE>(rptr, mScaT);
      utils::serialize<BTLA_DTYPE>(rptr, mZpT);
      utils::serialize<BTLA_DTYPE>(rptr, mRedT);
      utils::serialize<int>(rptr, mCStep);
      utils::serialize<size_t>(rptr, mCSize);
    }
    mScaleBuf.deserializeBuffer(rptr, locate_buf);
    mZpBuf.deserializeBuffer(rptr, locate_buf);
    mRedBuf.deserializeBuffer(rptr, locate_buf);
  }

 protected:
  inline void updateSize() {
    mScaEleSize = int(utils::bestla_dtype_size(mScaT));
    mZpEleSize = int(utils::bestla_dtype_size(mZpT));
    mRedEleSize = int(utils::bestla_dtype_size(mRedT));
  }

  inline constexpr size_t getMiscSize() {
    size_t totalsize = 0;
    totalsize += sizeof(mScaT);
    totalsize += sizeof(mZpT);
    totalsize += sizeof(mRedT);
    totalsize += sizeof(mCStep);
    totalsize += sizeof(mCSize);
    return totalsize;
  }
};

class IWeightBase : public storage::ISerializable {
 public:
  BTLA_PROLOGUEB_IDS mPrologueID = BTLA_PROLOGUEB_IDS::Undef;
  uint64_t mCoreId = 0;
  BTLA_DTYPE mDType = BTLA_DTYPE::F32;
  int mNPad = 0, mKPad = 0;
  int mN = 0, mK = 0;

  IWeightBase(uint64_t _id) { mCoreId = _id; }

  // bytes offset to mPrologueID
  static constexpr inline size_t offset() { return sizeof(mSize); }

 protected:
  void resize(int NPad, int KPad, int N, int K, BTLA_DTYPE dtype) {
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
      mPrologueID = utils::deserialize<BTLA_PROLOGUEB_IDS>(rptr);
      mCoreId = utils::deserialize<uint64_t>(rptr);
      mNPad = utils::deserialize<int>(rptr);
      mKPad = utils::deserialize<int>(rptr);
      mN = utils::deserialize<int>(rptr);
      mK = utils::deserialize<int>(rptr);
      mDType = utils::deserialize<BTLA_DTYPE>(rptr);
    } else {
      utils::serialize<BTLA_PROLOGUEB_IDS>(rptr, mPrologueID);
      utils::serialize<uint64_t>(rptr, mCoreId);
      utils::serialize<int>(rptr, mNPad);
      utils::serialize<int>(rptr, mKPad);
      utils::serialize<int>(rptr, mN);
      utils::serialize<int>(rptr, mK);
      utils::serialize<BTLA_DTYPE>(rptr, mDType);
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

class IWeightKBlockBase : public IWeightBase {
 public:
  int mBlockSize = 1;
  IWeightKBlockBase(uint64_t _id) : IWeightBase(_id) {}
  void resize(int NPad, int KPad, int Block, int N, int K, BTLA_DTYPE dtype) {
    IWeightBase::resize(NPad, KPad, N, K, dtype);
    mBlockSize = Block;
  }

 protected:
  virtual size_t getSerializedSize() {
    size_t totalsize = IWeightBase::getSerializedSize() + getMiscSize();
    return totalsize;
  }

  virtual void serializeToBuffer(int8_t*& wptr) {
    IWeightBase::serializeToBuffer(wptr);
    utils::serialize(wptr, mBlockSize);
  }

  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) {
    IWeightBase::deserializeBuffer(rptr, map_buf);
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

class IActivationBase : public storage::ISerializable {
 public:
  BTLA_PROLOGUEB_IDS mPrologueID = BTLA_PROLOGUEB_IDS::Undef;
  uint64_t mCoreId = 0;
  BTLA_DTYPE mDType = BTLA_DTYPE::F32;
  int mMPad = 0, mKPad = 0;
  int mM = 0, mK = 0;

  IActivationBase(uint64_t _id) { mCoreId = _id; }

  // bytes offset to mPrologueID
  static constexpr inline size_t offset() { return sizeof(mSize); }

 protected:
  void resize(int NPad, int KPad, int N, int K, BTLA_DTYPE dtype) {
    mMPad = NPad;
    mKPad = KPad;
    mM = N;
    mK = K;
    mDType = dtype;
  }

  virtual size_t getSerializedSize() { return ISerializable::getSerializedSize() + getMiscSize(); }

  virtual void serializeToBuffer(int8_t*& wptr) {
    ISerializable::serializeToBuffer(wptr);
    utils::serialize(wptr, mPrologueID);
    utils::serialize(wptr, mCoreId);
    utils::serialize(wptr, mMPad);
    utils::serialize(wptr, mKPad);
    utils::serialize(wptr, mM);
    utils::serialize(wptr, mK);
    utils::serialize(wptr, mDType);
  }

  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) {
    ISerializable::deserializeBuffer(rptr, map_buf);
    if (!map_buf) {
      mPrologueID = utils::deserialize<BTLA_PROLOGUEB_IDS>(rptr);
      mCoreId = utils::deserialize<uint64_t>(rptr);
      mMPad = utils::deserialize<int>(rptr);
      mKPad = utils::deserialize<int>(rptr);
      mM = utils::deserialize<int>(rptr);
      mK = utils::deserialize<int>(rptr);
      mDType = utils::deserialize<BTLA_DTYPE>(rptr);
    } else {
      utils::serialize<BTLA_PROLOGUEB_IDS>(rptr, mPrologueID);
      utils::serialize<uint64_t>(rptr, mCoreId);
      utils::serialize<int>(rptr, mMPad);
      utils::serialize<int>(rptr, mKPad);
      utils::serialize<int>(rptr, mM);
      utils::serialize<int>(rptr, mK);
      utils::serialize<BTLA_DTYPE>(rptr, mDType);
    }
  }

  inline constexpr size_t getMiscSize() {
    size_t totalsize = 0;
    totalsize += sizeof(mPrologueID);
    totalsize += sizeof(mCoreId);
    totalsize += sizeof(mMPad);
    totalsize += sizeof(mKPad);
    totalsize += sizeof(mM);
    totalsize += sizeof(mK);
    totalsize += sizeof(mDType);
    return totalsize;
  }
};

class IActivationKBlockBase : public IActivationBase {
 public:
  int mBlockSize = 1;
  IActivationKBlockBase(uint64_t _id) : IActivationBase(_id) {}
  void resize(int MPad, int KPad, int Block, int N, int K, BTLA_DTYPE dtype) {
    IActivationBase::resize(MPad, KPad, N, K, dtype);
    mBlockSize = Block;
  }

 protected:
  virtual size_t getSerializedSize() {
    size_t totalsize = IActivationBase::getSerializedSize() + getMiscSize();
    return totalsize;
  }

  virtual void serializeToBuffer(int8_t*& wptr) {
    IActivationBase::serializeToBuffer(wptr);
    utils::serialize(wptr, mBlockSize);
  }

  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) {
    IActivationBase::deserializeBuffer(rptr, map_buf);
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

class StoragePackedWeight : public IWeightBase {
 public:
  ObjectAlignedBuffer<Alignment> mWBuf;
  StoragePackedWeight(uint64_t _id) : IWeightBase(_id) { mPrologueID = BTLA_PROLOGUEB_IDS::WeightPack; }

  size_t resize(int NPad, int KPad, int N, int K, BTLA_DTYPE dtype) {
    IWeightBase::resize(NPad, KPad, N, K, dtype);
    auto bsize = static_cast<size_t>(NPad) * KPad * utils::bestla_dtype_size(dtype);
    mWBuf.resize(bsize);
    mSize = IWeightBase::getSerializedSize() + mWBuf.getSerializedSize();
    mSize = utils::padto(mSize, Alignment);
    return mSize;
  }

  template <typename T>
  inline constexpr T* WPtr() const {
    return mWBuf.get<T>();
  }

  virtual void assign(int8_t* buf) override {
    IWeightBase::deserializeBuffer(buf, true);
    mWBuf.deserializeBuffer(buf, true);
  }

  virtual void serialize(int8_t* wptr) {
    IWeightBase::serializeToBuffer(wptr);
    mWBuf.serializeToBuffer(wptr);
  }

  virtual void deserialize(int8_t* rptr) override {
    IWeightBase::deserializeBuffer(rptr, false);
    mWBuf.deserializeBuffer(rptr, false);
  }
};

class StorageReduce : public ISerializable {
 public:
  using CorrectionType = ObjectQuantCorrection;
  int m = 0, k = 0, lda = 0, kblock = 1;
  ObjectAlignedBuffer<Alignment> mRedBuf;
  size_t resize(int _m, int _k, int _kblock, BTLA_DTYPE redt) {
    kblock = _kblock;
    m = _m;
    k = _k;
    lda = utils::updiv(_k, _kblock);
    size_t bufsize = static_cast<size_t>(m) * lda * utils::bestla_dtype_size(redt);
    mRedBuf.resize(bufsize);
    mSize = getSerializedSize();
    mSize = utils::padto(mSize, Alignment);
    return mSize;
  }
  template <typename QT_T>
  inline QT_T* RPtr() {
    return mRedBuf.get<QT_T>();
  }

  virtual void assign(int8_t* buf) override {
    ISerializable::deserializeBuffer(buf, true);
    deserializeBuffer(buf, true);
    mRedBuf.deserializeBuffer(buf, true);
  }

  virtual void serialize(int8_t* wptr) {
    ISerializable::serializeToBuffer(wptr);
    serializeToBuffer(wptr);
    mRedBuf.serializeToBuffer(wptr);
  }

  virtual void deserialize(int8_t* rptr) override {
    ISerializable::deserializeBuffer(rptr, false);
    deserializeBuffer(rptr, false);
    mRedBuf.deserializeBuffer(rptr, false);
  }

 protected:
  virtual size_t getSerializedSize() {
    return ISerializable::getSerializedSize() + getMiscSize() + mRedBuf.getSerializedSize();
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

class StorageReorderActivation : public IActivationKBlockBase {
 public:
  ObjectAlignedBuffer<Alignment> mABuf;
  StorageReorderActivation(uint64_t _id) : IActivationKBlockBase(_id) { mPrologueID = BTLA_PROLOGUEB_IDS::WeightPack; }

  size_t resize(int MPad, int KPad, int M, int K, int KBlock, BTLA_DTYPE dtype) {
    IActivationKBlockBase::resize(MPad, KPad, KBlock, M, K, dtype);
    auto bsize = static_cast<size_t>(MPad) * KPad * utils::bestla_dtype_size(dtype);
    mABuf.resize(bsize);
    mSize = IActivationKBlockBase::getSerializedSize() + mABuf.getSerializedSize();
    mSize = utils::padto(mSize, Alignment);
    return mSize;
  }

  template <typename T>
  inline constexpr T* APtr() const {
    return mABuf.get<T>();
  }

  virtual void assign(int8_t* buf) override {
    IActivationKBlockBase::deserializeBuffer(buf, true);
    mABuf.deserializeBuffer(buf, true);
  }

  virtual void serialize(int8_t* wptr) {
    IActivationKBlockBase::serializeToBuffer(wptr);
    mABuf.serializeToBuffer(wptr);
  }

  virtual void deserialize(int8_t* rptr) override {
    IActivationKBlockBase::deserializeBuffer(rptr, false);
    mABuf.deserializeBuffer(rptr, false);
  }
};

class StorageQuantActivation : public IActivationKBlockBase {
 public:
  using CorrectionType = ObjectQuantCorrection;
  CorrectionType mCorrection;
  ObjectAlignedBuffer<Alignment> mQBuf;
  StorageQuantActivation(uint64_t _id = 0) : IActivationKBlockBase(_id) {
    mPrologueID = BTLA_PROLOGUEB_IDS::WeightPack;
  }

  size_t resize(int _mpad, int _kpad, int _m, int _k, int _kblock, BTLA_DTYPE buft, BTLA_DTYPE scalet, BTLA_DTYPE zpt,
                BTLA_DTYPE redt, bool is_asym, bool has_reduce) {
    IActivationKBlockBase::resize(_mpad, _kpad, _kblock, _m, _k, buft);
    mCorrection.resize(_mpad, utils::updiv(_kpad, _kblock), scalet, zpt, redt, is_asym, has_reduce);
    size_t bufsize = static_cast<size_t>(_mpad) * _kpad * utils::bestla_dtype_size(buft);
    mQBuf.resize(bufsize);
    mSize = getSerializedSize();
    mSize = utils::padto(mSize, Alignment);
    return mSize;
  }
  template <typename QT_T>
  inline constexpr QT_T* APtr() {
    return mQBuf.get<QT_T>();
  }

  template <typename QT_T>
  inline constexpr QT_T* ZPtr() {
    return mCorrection.mZpBuf.get<QT_T>();
  }

  template <typename QT_T>
  inline constexpr QT_T* SPtr() {
    return mCorrection.mScaleBuf.get<QT_T>();
  }

  template <typename QT_T>
  inline constexpr QT_T* RPtr() {
    return mCorrection.mRedBuf.get<QT_T>();
  }

  inline constexpr BTLA_DTYPE RDtype() { return mCorrection.mRedT; }
  inline constexpr BTLA_DTYPE ZDtype() { return mCorrection.mZpT; }
  inline constexpr BTLA_DTYPE SDtype() { return mCorrection.mScaT; }
  inline constexpr bool IsAsym() { return mCorrection.mZpBuf.mNotEmpty; }
  inline constexpr bool HasReduce() { return mCorrection.mRedBuf.mNotEmpty; }
  inline constexpr size_t CSize() { return mCorrection.mCSize; }
  inline constexpr int CStep() { return mCorrection.mCStep; }

  virtual void assign(int8_t* buf) override {
    IActivationKBlockBase::deserializeBuffer(buf, true);
    deserializeBuffer(buf, true);
  }

  virtual void serialize(int8_t* wptr) {
    IActivationKBlockBase::serializeToBuffer(wptr);
    serializeToBuffer(wptr);
  }

  virtual void deserialize(int8_t* rptr) override {
    IActivationKBlockBase::deserializeBuffer(rptr, false);
    deserializeBuffer(rptr, false);
  }

 protected:
  virtual size_t getSerializedSize() {
    return ISerializable::getSerializedSize() + getMiscSize() + mQBuf.getSerializedSize() +
           mCorrection.getSerializedSize();
  }

  virtual void serializeToBuffer(int8_t*& wptr) {
    mQBuf.serializeToBuffer(wptr);
    mCorrection.serializeToBuffer(wptr);
  }

  virtual void deserializeBuffer(int8_t*& rptr, bool map_buf) {
    mQBuf.deserializeBuffer(rptr, map_buf);
    mCorrection.deserializeBuffer(rptr, map_buf);
  }

  inline constexpr size_t getMiscSize() {
    size_t totalsize = 0;
    return totalsize;
  }
};

class StorageWeightKBlockNInteger : public IWeightKBlockBase {
 public:
  using InfoType = IWeightKBlockBase;
  using QWeightType = ObjectAlignedBuffer<Alignment>;
  using CorrectionType = ObjectQuantCorrection;
  QWeightType mQBuf;
  CorrectionType mCorrection;
  ObjectOptionalBuffer<Alignment> mShuffleIndices;
  StorageWeightKBlockNInteger(uint64_t _type) : IWeightKBlockBase(_type) {
    mPrologueID = BTLA_PROLOGUEB_IDS::WeightKBlockNInteger;
  }

  size_t resize(int NPad, int KPad, int Block, int N, int K, BTLA_DTYPE qtype, BTLA_DTYPE scalet, BTLA_DTYPE redt,
                bool IsAsym) {
    BTLA_DTYPE zpt = BTLA_DTYPE::S8;
    InfoType::resize(NPad, KPad, Block, N, K, qtype);
    auto bits = utils::bestla_dtype_bits(qtype);
    auto elesize = static_cast<size_t>(NPad) * KPad;
    auto bytes = utils::updiv(elesize * bits, 8);  // add 3bits, 5btis, 7bits size calculation here
    mQBuf.resize(bytes);
    int nk_scale = utils::updiv(KPad, Block);
    auto gemm_comp = bestla::gemm::CoreAttr::get_comp(mCoreId);
    auto is_cint = bestla::gemm::CompTypeHelper::is_integer(gemm_comp);
    mCorrection.resize(nk_scale, NPad, scalet, zpt, redt, IsAsym, is_cint);
    update_size();
    return mSize;
  }

  void enable_shuffle() {
    auto indicessize = mK * sizeof(int);
    mShuffleIndices.resize(indicessize);
    update_size();
  }

  inline constexpr BTLA_DTYPE RDtype() { return mCorrection.mRedT; }
  inline constexpr BTLA_DTYPE ZDtype() { return mCorrection.mZpT; }
  inline constexpr BTLA_DTYPE SDtype() { return mCorrection.mScaT; }
  inline constexpr bool IsAsym() { return mCorrection.mZpBuf.mNotEmpty; }
  inline constexpr bool HasReduce() { return mCorrection.mRedBuf.mNotEmpty; }
  inline constexpr size_t CSize() { return mCorrection.mCSize; }
  inline constexpr int CStep() { return mCorrection.mCStep; }

  template <typename T>
  inline constexpr size_t WSize() {
    return mQBuf.size<T>();
  }

  template <typename T>
  inline constexpr T* WPtr() const {
    return mQBuf.get<T>();
  }

  template <typename T>
  inline constexpr T* SPtr() {
    return mCorrection.mScaleBuf.get<T>();
  }

  template <typename T>
  inline constexpr T* ZPtr() {
    return mCorrection.mZpBuf.get<T>();
  }

  template <typename T>
  inline constexpr T* RPtr() {
    return mCorrection.mRedBuf.get<T>();
  }

  inline constexpr int* ShfIndice() { return mShuffleIndices.get<int>(); }

  void update_size() {
    mSize = InfoType::getSerializedSize() + mQBuf.getSerializedSize() + mCorrection.getSerializedSize() +
            mShuffleIndices.getSerializedSize();
    mSize = utils::padto(mSize, Alignment);
  }

  virtual void assign(int8_t* buf) override {
    InfoType::deserializeBuffer(buf, true);
    mQBuf.deserializeBuffer(buf, true);
    mCorrection.deserializeBuffer(buf, true);
    mShuffleIndices.deserializeBuffer(buf, true);
  }

  virtual void serialize(int8_t* wptr) {
    InfoType::serializeToBuffer(wptr);
    mQBuf.serializeToBuffer(wptr);
    mCorrection.serializeToBuffer(wptr);
    mShuffleIndices.serializeToBuffer(wptr);
  }

  virtual void deserialize(int8_t* rptr) override {
    InfoType::deserializeBuffer(rptr, false);
    mQBuf.deserializeBuffer(rptr, false);
    mCorrection.deserializeBuffer(rptr, false);
    mShuffleIndices.deserializeBuffer(rptr, false);
  }
};

class StorageWeightKBlockNFloat : public StorageWeightKBlockNInteger {
 public:
  StorageWeightKBlockNFloat(uint64_t _type) : StorageWeightKBlockNInteger(_type) {
    mPrologueID = BTLA_PROLOGUEB_IDS::WeightKBlockNFloat;
  }

  size_t resize(int NPad, int KPad, int Block, int N, int K, BTLA_DTYPE ftype, BTLA_DTYPE scalet) {
    StorageWeightKBlockNInteger::InfoType::resize(NPad, KPad, Block, N, K, ftype);
    auto bits = utils::bestla_dtype_bits(ftype);
    auto elesize = static_cast<size_t>(NPad) * KPad;
    auto bytes = utils::updiv(elesize * bits, 8);  // add fp6 size calculation here
    StorageWeightKBlockNInteger::mQBuf.resize(bytes);
    int nk_scale = utils::updiv(KPad, Block);
    StorageWeightKBlockNInteger::mCorrection.resize(nk_scale, NPad, scalet, BTLA_DTYPE::EleBitsUndef,
                                                    BTLA_DTYPE::EleBitsUndef, false, false);
    mSize = StorageWeightKBlockNInteger::InfoType::getSerializedSize() +
            StorageWeightKBlockNInteger::mQBuf.getSerializedSize() +
            StorageWeightKBlockNInteger::mCorrection.getSerializedSize();
    mSize = utils::padto(mSize, Alignment);
    return mSize;
  }
};

class PackedWeightParser {
 public:
  static gemm::IWeightBase* deserialBuffer(const void* serialized_buf) {
    if (serialized_buf == nullptr) {
      return nullptr;
    }
    auto tmpptr = const_cast<void*>(serialized_buf);
    auto rptr = reinterpret_cast<int8_t*>(tmpptr);
    rptr += IWeightBase::offset();
    int mProID = utils::deserialize<int>(rptr);
    IWeightBase* ptr = nullptr;
    if (mProID >= int(BTLA_PROLOGUEB_IDS::Begin) && mProID < int(BTLA_PROLOGUEB_IDS::End)) {
      rptr = reinterpret_cast<int8_t*>(tmpptr);
      auto type = static_cast<BTLA_PROLOGUEB_IDS>(mProID);
      switch (type) {
        case BTLA_PROLOGUEB_IDS::WeightPack:
          ptr = new gemm::StoragePackedWeight(0);
          break;
        case BTLA_PROLOGUEB_IDS::WeightKBlockNInteger:
          ptr = new gemm::StorageWeightKBlockNInteger(0);
          break;
        case BTLA_PROLOGUEB_IDS::WeightKBlockNFloat:
          ptr = new gemm::StorageWeightKBlockNFloat(0);
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
}  // namespace bestla
