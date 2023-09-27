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
#include "jit_blas_wrapper.h"
#include "kernel_wrapper.h"

namespace jblas {
namespace prologue {
namespace weight_comp {
class PackedWeightKBlock : public prologue::PackedWeight {
 public:
  PackedWeightKBlock(jblas::gemm::GemmCoreType _type) : PackedWeight(_type) {}
  void resize(int NPad, int KPad, int Block) {
    PackedWeight::resize(NPad, KPad);
    mBlockSize = Block;
  }

  virtual size_t getSerializedSize() override {
    size_t totalsize = 0;
    totalsize += sizeof(mSize);
    totalsize += sizeof(mCoreType);
    totalsize += sizeof(mType);
    totalsize += sizeof(mNPad);
    totalsize += sizeof(mKPad);
    totalsize += sizeof(mBlockSize);
    totalsize += getDataSerializedSize();
    return totalsize;
  }

  virtual void serializeToBuffer(void* buf) override {
    auto wptr = reinterpret_cast<int8_t*>(buf);
    mSize = getSerializedSize();
    utils::serialize(wptr, mSize);
    utils::serialize(wptr, mCoreType);
    utils::serialize(wptr, mType);
    utils::serialize(wptr, mNPad);
    utils::serialize(wptr, mKPad);
    utils::serialize(wptr, mBlockSize);
    serializeDataToBuffer(wptr);
  }

  virtual void deserializeBuffer(void* buf, int memalloc) override {
    auto rptr = reinterpret_cast<int8_t*>(buf);
    mSize = utils::deserialize<size_t>(rptr);
    mCoreType = utils::deserialize<jblas::gemm::GemmCoreType>(rptr);
    mType = utils::deserialize<int>(rptr);
    mNPad = utils::deserialize<int>(rptr);
    mKPad = utils::deserialize<int>(rptr);
    mBlockSize = utils::deserialize<int>(rptr);
    deserializeDataBuffer(rptr, memalloc);
  }

  int mBlockSize = 1;
};
namespace gemm_kblcok {

enum class WeightCompType : int {
  Begin = int(prologue::WeightPrologueType::End),
  WeightS4ClipScaleFp32 = Begin,
  WeightS4ClipScaleBf16,
  WeightS4FullRangeScaleFp32,
  WeightS4FullRangeScaleBf16,
  WeightS8ScaleFp32,
  WeightFp4BnbScaleFp32,
  WeightFp4E2M1ScaleFp32,
  WeightNf4ScaleFp32,
  WeightS8ScaleFp32PerChannelN,
  WeightS4ClipScaleFp32PerChannelN,
  End,
};

class StorageWeight8Bit {
 public:
  void resize(int NPad, int KPad) {
    mWeights.resize((size_t)NPad * KPad);
    mWPtr = mWeights.data();
    mWSize = mWeights.size();
  }

  int8_t* mWPtr = NULL;
  size_t mWSize = 0;

 protected:
  size_t myDataSerializedSize() {
    size_t totalsize = 0;
    totalsize += sizeof(mWSize);
    totalsize += mWSize * sizeof(mWPtr[0]);
    return totalsize;
  }
  void mySerializeDataToBuffer(int8_t*& wptr) {
    utils::serialize(wptr, mWSize);
    for (size_t i = 0; i < mWSize; i++) {
      utils::serialize(wptr, mWPtr[i]);
    }
  }
  void myDeserializeDataBuffer(int8_t*& rptr, int memalloc) {
    size_t rsize = utils::deserialize<size_t>(rptr);
    if (memalloc) {
      mWeights.resize(rsize);
      std::memcpy(mWeights.data(), rptr, rsize * sizeof(mWeights[0]));
      mWPtr = mWeights.data();
      mWSize = mWeights.size();
    } else {
      mWPtr = (int8_t*)rptr;
      mWSize = rsize;
    }
    rptr += rsize * sizeof(mWeights[0]);
  }
  utils::aligned_vector<int8_t> mWeights;
};

class StorageWeight4Bit {
 public:
  void resize(int NPad, int KPad) {
    mWeights.resize(utils::updiv((size_t)NPad * KPad, 2));
    mWPtr = mWeights.data();
    mWSize = mWeights.size();
  }

  utils::bit4x2* mWPtr = NULL;
  size_t mWSize = 0;

 protected:
  size_t myDataSerializedSize() {
    size_t totalsize = 0;
    totalsize += sizeof(mWSize);
    totalsize += mWSize * sizeof(mWPtr[0]);
    return totalsize;
  }
  void mySerializeDataToBuffer(int8_t*& wptr) {
    utils::serialize(wptr, mWSize);
    for (size_t i = 0; i < mWSize; i++) {
      utils::serialize(wptr, mWPtr[i]);
    }
  }
  void myDeserializeDataBuffer(int8_t*& rptr, int memalloc) {
    size_t rsize = utils::deserialize<size_t>(rptr);
    if (memalloc) {
      mWeights.resize(rsize);
      std::memcpy(mWeights.data(), rptr, rsize * sizeof(mWeights[0]));
      mWPtr = mWeights.data();
      mWSize = mWeights.size();
    } else {
      mWPtr = (utils::bit4x2*)rptr;
      mWSize = rsize;
    }
    rptr += rsize * sizeof(mWeights[0]);
  }
  utils::aligned_vector<utils::bit4x2> mWeights;
};

template <typename SRC_T, typename DST_T, typename RED_T>
class StorageSimpleCorrection {
 public:
  SRC_T* mSPtr = nullptr;
  DST_T* mZPtr = nullptr;
  RED_T* mRPtr = nullptr;
  size_t mSSize = 0;
  int mStep = 0;

  void resize(int NPad, int KBlks, bool _is_sym = true, bool _has_reduce = true) {
    isSym = _is_sym;
    hasReduce = _has_reduce;
    mStep = NPad;
    mScales.resize((size_t)NPad * KBlks);
    mSPtr = mScales.data();
    if (!is_sym()) {
      mZeroPoints.resize((size_t)NPad * KBlks);
      mZPtr = mZeroPoints.data();
    } else {
      mZPtr = nullptr;
    }
    if (has_reduce()) {
      mReduce.resize((size_t)NPad * KBlks);
      mRPtr = mReduce.data();
    } else {
      mRPtr = nullptr;
    }
    mSSize = mScales.size();
  }

  void init_scales(size_t _size, SRC_T* _scales, int memalloc = 0) {
    if (memalloc) {
      mScales.resize(_size);
      std::memcpy(mScales.data(), _scales, _size * sizeof(SRC_T));
      mSPtr = mScales.data();
    } else {
      mSPtr = _scales;
    }
  }

  void init_zp(size_t _size, DST_T* _zeroPoints, int memalloc = 0) {
    if (memalloc) {
      mZeroPoints.resize(_size);
      std::memcpy(mZeroPoints.data(), _zeroPoints, _size * sizeof(DST_T));
      mZPtr = mZeroPoints.data();
    } else {
      mZPtr = _zeroPoints;
    }
  }

  void init_reduce(size_t _size, RED_T* _ptr, int memalloc = 0) {
    if (memalloc) {
      mReduce.resize(_size);
      std::memcpy(mReduce.data(), _ptr, _size * sizeof(RED_T));
      mRPtr = mReduce.data();
    } else {
      mRPtr = _ptr;
    }
  }

  constexpr bool is_sym() { return isSym; }
  constexpr bool has_reduce() { return hasReduce; }
  inline SRC_T* get_scales() { return mSPtr; }
  inline DST_T* get_zps() { return mZPtr; }
  inline RED_T* get_reduce() { return mRPtr; }
  inline size_t size() { return mSSize; }
  inline int get_step() { return mStep; }

 protected:
  size_t myDataSerializedSize() {
    size_t totalsize = 0;
    totalsize += sizeof(isSym);
    totalsize += sizeof(hasReduce);
    totalsize += sizeof(mStep);
    totalsize += sizeof(mSSize);
    totalsize += mSSize * sizeof(mSPtr[0]);
    if (!is_sym()) totalsize += mSSize * sizeof(mZPtr[0]);
    if (has_reduce()) totalsize += mSSize * sizeof(mRPtr[0]);
    return totalsize;
  }
  void mySerializeDataToBuffer(int8_t*& wptr) {
    utils::serialize(wptr, isSym);
    utils::serialize(wptr, hasReduce);
    utils::serialize(wptr, mStep);
    utils::serialize(wptr, mSSize);
    for (size_t i = 0; i < mSSize; i++) {
      utils::serialize(wptr, mSPtr[i]);
    }
    if (!is_sym()) {
      for (size_t i = 0; i < mSSize; i++) {
        utils::serialize(wptr, mZPtr[i]);
      }
    }
    if (has_reduce()) {
      for (size_t i = 0; i < mSSize; i++) {
        utils::serialize(wptr, mRPtr[i]);
      }
    }
  }
  void myDeserializeDataBuffer(int8_t*& rptr, int memalloc) {
    isSym = utils::deserialize<bool>(rptr);
    hasReduce = utils::deserialize<bool>(rptr);
    mStep = utils::deserialize<int>(rptr);
    mSSize = utils::deserialize<size_t>(rptr);
    auto src_rptr = reinterpret_cast<SRC_T*>(rptr);
    init_scales(mSSize, src_rptr, memalloc);
    rptr += mSSize * sizeof(mScales[0]);
    if (!is_sym()) {
      auto dst_rptr = reinterpret_cast<DST_T*>(rptr);
      init_zp(mSSize, dst_rptr, memalloc);
      rptr += mSSize * sizeof(mZeroPoints[0]);
    }
    if (has_reduce()) {
      auto dst_rptr = reinterpret_cast<RED_T*>(rptr);
      init_reduce(mSSize, dst_rptr, memalloc);
      rptr += mSSize * sizeof(mZeroPoints[0]);
    }
  }
  bool isSym = true;
  bool hasReduce = false;

  utils::aligned_vector<SRC_T> mScales;
  utils::aligned_vector<DST_T> mZeroPoints;
  utils::aligned_vector<RED_T> mReduce;
};

class StorageWeightS8ScaleFp32 : public prologue::weight_comp::PackedWeightKBlock,
                                 public StorageWeight8Bit,
                                 public StorageSimpleCorrection<float, int8_t, float> {
 public:
  using QWeightType = StorageWeight8Bit;
  using CorrectionType = StorageSimpleCorrection<float, int8_t, float>;
  StorageWeightS8ScaleFp32(jblas::gemm::GemmCoreType _type) : prologue::weight_comp::PackedWeightKBlock(_type) {
    mType = static_cast<int>(WeightCompType::WeightS8ScaleFp32);
  }

  void resize(int NPad, int KPad, int Block, bool IsSym = true) {
    PackedWeightKBlock::resize(NPad, KPad, Block);
    QWeightType::resize(NPad, KPad);
    int nk_scale = utils::updiv(KPad, Block);
    CorrectionType::resize(NPad, nk_scale, IsSym,
                           true);  // create reduce space for weight which may use int8 compute type
  }

 protected:
  virtual size_t getDataSerializedSize() override {
    size_t totalsize = QWeightType::myDataSerializedSize() + CorrectionType::myDataSerializedSize();
    return totalsize;
  }
  virtual void serializeDataToBuffer(void* buf) override {
    auto wptr = reinterpret_cast<int8_t*>(buf);
    QWeightType::mySerializeDataToBuffer(wptr);
    CorrectionType::mySerializeDataToBuffer(wptr);
  }
  virtual void deserializeDataBuffer(void* buf, int memalloc) override {
    auto rptr = reinterpret_cast<int8_t*>(buf);
    QWeightType::myDeserializeDataBuffer(rptr, memalloc);
    CorrectionType::myDeserializeDataBuffer(rptr, memalloc);
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class WeightS8ScaleFp32 {
 public:
  struct Param {
    const prologue::PackedWeight* packedW;
  };
  using StorageWeight = StorageWeightS8ScaleFp32;
  using SType = float;
  using WeightBaseFloat = jblas::prologue::gemm::WeightBase<float, ISA_T>;
  using Parallel = utils::parallel::Parallel2DRowMajor;
  virtual PackedWeight* createStorage(const int N, const int K, int blocksize, bool is_sym = true) {
    int KPad = utils::padto(K, _GemmCore_T::KTILE);
    int NPad = utils::padto(N, _GemmCore_T::NTILE);
    auto ptr = new StorageWeight(_GemmCore_T::TYPE);
    ptr->resize(NPad, KPad, blocksize <= 0 ? K : blocksize, is_sym);
    return ptr;
  }

  Parallel createParallel(const int N, const int K, const int blocksize) {
    assert(0);
    return Parallel();  // no runtime parallel forward
  }
  // only for compilation, weight compression prologue doesn't get any benefit from runtime compression
  void launch(const Param& _param, int tidx, Parallel& _para) {
    // no runtime parallel forward
    assert(0);
  }

  // from K*N fp32 weight to packed N//NtilexKPadxNTile weight
  virtual void packTransposeWeight(const int N, const int K, const float* B, const int ldb, PackedWeight* stor,
                                   bool is_sym = true) {
    utils::aligned_vector<float> B_NT(N * K);
    WeightBaseFloat::transposeWeight(N, K, B, ldb, B_NT.data(), N);
    packWeight(N, K, B_NT.data(), N, stor, is_sym);
  }

  // from packed N//NtilexKPadxNTile int8 weight to KxN f32 weight
  virtual void unpackTransposeWeight(const int N, const int K, PackedWeight* stor, float* B, const int ldb) {
    utils::aligned_vector<float> B_NT(N * K);
    unpackWeight(N, K, stor, B_NT.data(), N);
    WeightBaseFloat::transposeWeight(K, N, B_NT.data(), N, B, ldb);
  }

  // from KxN f32 weight to packed N//NtilexKPadxNTile int8 weight
  virtual void packWeight(const int N, const int K, const float* B, const int ldb, PackedWeight* stor,
                          bool is_sym = true) {
    utils::aligned_vector<int8_t> tmpq(N * K);
    auto ptr = reinterpret_cast<PackedWeightKBlock*>(stor);
    int nk_scale = utils::updiv(K, ptr->mBlockSize);
    StorageSimpleCorrection<float, int8_t, float> corr;
    corr.resize(N, nk_scale, is_sym, false);
    quantizeWeight(N, K, B, ldb, ptr->mBlockSize, tmpq.data(), corr.get_scales(), corr.get_zps());
    packQWeight(N, K, tmpq.data(), ldb, corr.get_scales(), corr.get_zps(), stor);
  }

  virtual void unpackWeight(const int N, const int K, PackedWeight* stor, float* B, const int ldb) {
    utils::parallel::Parallel2DRowMajor _para;
    utils::CpuBase cb;
    _para.update(K, N, _GemmCore_T::KTILE, _GemmCore_T::NTILE, cb.mNumThreads);
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      int colidx, rowidx, rowsize, colsize;
      _para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        int rowremain = utils::remainsize(rowidx, K,
                                          rowsize);  // rowremain: src valid size. rowsize: padded size
        int colremain = utils::remainsize(colidx, N, colsize);
        std::vector<float> dequant(rowsize * colsize);
        int dststep = 0;
        auto dstptr = dequant.data();
        auto rowpad = utils::padto(rowremain, _GemmCore_T::KTILE);
        auto colpad = utils::padto(colremain, _GemmCore_T::NTILE);
        getWeight(&dstptr, &dststep, rowpad, colpad, rowidx, colidx, {stor});
        kernel::wrapper::RevertPaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW>::template forward<ISA_T>(
            dstptr, B + rowidx * ldb + colidx, rowremain, colremain, rowpad, colpad, dststep, ldb);
      }
    }
  }

  virtual void unpackWeight(const int N, const int K, PackedWeight* stor, int8_t* B, const int ldb) {
    utils::parallel::Parallel2DRowMajor _para;
    utils::CpuBase cb;
    _para.update(K, N, _GemmCore_T::KTILE, _GemmCore_T::NTILE, cb.mNumThreads);
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      int colidx, rowidx, rowsize, colsize;
      _para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        int rowremain = utils::remainsize(rowidx, K,
                                          rowsize);  // rowremain: src valid size. rowsize: padded size
        int colremain = utils::remainsize(colidx, N, colsize);
        std::vector<int8_t> dequant(rowsize * colsize);
        int dststep = 0;
        auto dstptr = dequant.data();
        auto rowpad = utils::padto(rowremain, _GemmCore_T::KTILE);
        auto colpad = utils::padto(colremain, _GemmCore_T::NTILE);
        getWeight(&dstptr, &dststep, rowpad, colpad, rowidx, colidx, {stor});
        kernel::wrapper::RevertPaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW>::template forward<ISA_T>(
            dstptr, B + rowidx * ldb + colidx, rowremain, colremain, rowpad, colpad, dststep, ldb);
      }
    }
  }

  virtual void packQWeight(const int N, const int K, const int8_t* B, const int ldb, const float* scales,
                           const int8_t* zero_points, PackedWeight* ptr) {
    auto stor = reinterpret_cast<StorageWeight*>(ptr);
    int rawnk_scale = utils::updiv(K, stor->mBlockSize);
    int nk_scale = utils::updiv(stor->mKPad, stor->mBlockSize);
#pragma omp parallel for
    for (int i = 0; i < nk_scale; i++) {  // padding copy
      if (i < rawnk_scale) {
        std::memcpy(stor->mSPtr + i * stor->mNPad, scales + i * N, N * sizeof(scales[0]));
        if (zero_points != nullptr) {
          std::memcpy(stor->mZPtr + i * stor->mNPad, zero_points + i * N, N * sizeof(zero_points[0]));
        }
      } else {
        std::memset(stor->mSPtr + i * stor->mNPad, 0, stor->mNPad * sizeof(stor->mSPtr[0]));
        if (zero_points != nullptr) {
          std::memset(stor->mZPtr + i * stor->mNPad, 0, stor->mNPad * sizeof(zero_points[0]));
        }
      }
    }

    reorderWeight(N, K, B, ldb, stor->mWPtr);
    if (stor->has_reduce()) {
      utils::avector<float> deq(K * N);
      unpackWeight(N, K, stor, deq.data(), N);
      reduceWeight(N, K, stor->mBlockSize, deq.data(), ldb, stor->mRPtr, stor->mNPad);
    }
  }

  void reduceWeight(const int N, const int K, const int KBlock, const float* B, const int ldb, float* rptr,
                    const int ldr) {
    utils::parallel::Parallel2DRowMajor _para;
    utils::CpuBase cb;
    _para.update(K, N, KBlock, 16, cb.mNumThreads);
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      int colidx, rowidx, rowsize, colsize;
      _para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        int colremain = utils::remainsize(colidx, N, colsize);
        const auto src = B + rowidx * ldb + colidx;
        const auto dst = rptr + colidx + rowidx / KBlock * ldr;
        using RowReduceSum = kernel::wrapper::RowReduceSum<float>;
        for (int i = 0; i < rowsize; i += KBlock) {
          int rowremain = utils::remainsize(rowidx + i, K, KBlock);
          auto ret = RowReduceSum::template forward<ISA_T>(  //
              src + i * ldb, ldb, rowremain, colremain, dst + i / KBlock * ldr);
          assert(ret == JblasSuccess);
          (void)ret;
        }
      }
    }
  }

  virtual inline JBLAS_CODE getWeight(float** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                      const Param& _param) {
    auto wptr = reinterpret_cast<const StorageWeight*>(_param.packedW);
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->mWPtr + n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      if constexpr (_GemmCore_T::PACK_ROW == 1) {
        kernel::wrapper::DecompressKBlockS8F32::forward<ISA_T, float>(
            bptr + i * KPad, *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
            _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW,
            _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, wptr->mSPtr + n_offset + i,
            wptr->mZPtr != nullptr ? wptr->mZPtr + n_offset + i : nullptr, k_offset, wptr->mBlockSize, NPad);
      } else {
        kernel::wrapper::DecompressKBlockS8FP32PackRow::forward<ISA_T, float>(
            bptr + i * KPad, *dstptr + i * k_size, k_size, _GemmCore_T::NTILE, _GemmCore_T::NTILE, _GemmCore_T::NTILE,
            wptr->mSPtr + n_offset + i, wptr->mZPtr != nullptr ? wptr->mZPtr + n_offset + i : nullptr, k_offset,
            wptr->mBlockSize, NPad, _GemmCore_T::PACK_ROW);
      }
    }
    *dststep = k_size;
    return JblasSuccess;
  }

  virtual inline JBLAS_CODE getWeight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                      const Param& _param) {
    auto wptr = reinterpret_cast<const StorageWeight*>(_param.packedW);
    auto KPad = wptr->mKPad;
    auto bptr = wptr->mWPtr + n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    kernel::wrapper::Memcpy2D::template forward<ISA_T, int8_t, int8_t>(
        bptr, *dstptr, n_size / _GemmCore_T::NTILE, _GemmCore_T::NTILE * k_size, _GemmCore_T::NTILE * KPad,
        _GemmCore_T::NTILE * k_size);
    *dststep = k_size;
    return JblasSuccess;
  }

  virtual JBLAS_CODE getScale(float** dstptr, int* dststep, int n_size, int k_size, int n_offset, int k_offset,
                              const Param& _param) {
    auto wptr = reinterpret_cast<const StorageWeight*>(_param.packedW);
    auto NPad = wptr->mNPad;
    *dstptr = wptr->mSPtr + n_offset + k_offset / wptr->mBlockSize * NPad;
    *dststep = NPad;
    return JblasSuccess;
  }

  virtual JBLAS_CODE getReduce(float** dstptr, int* dststep, int n_size, int k_size, int n_offset, int k_offset,
                               const Param& _param) {
    auto wptr = reinterpret_cast<const StorageWeight*>(_param.packedW);
    auto NPad = wptr->mNPad;
    *dstptr = wptr->mRPtr + n_offset + k_offset / wptr->mBlockSize * NPad;
    *dststep = NPad;
    return JblasSuccess;
  }

  virtual JBLAS_CODE getZp(int8_t** dstptr, int* dststep, int n_size, int k_size, int n_offset, int k_offset,
                           const Param& _param) {
    auto wptr = reinterpret_cast<const StorageWeight*>(_param.packedW);
    auto NPad = wptr->mNPad;
    *dstptr = wptr->mZPtr == nullptr ? nullptr : wptr->mZPtr + n_offset + k_offset / wptr->mBlockSize * NPad;
    *dststep = NPad;
    return JblasSuccess;
  }

 protected:
  virtual void quantRowBlock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                             float* scales, int8_t* zero_points, int blocksize) {
    kernel::wrapper::QuantizeSignIntRowBlock::forward<ISA_T, S8>(srcptr, dstptr, row, col, ld_src, ld_dst, scales,
                                                                 zero_points, blocksize);
  }

  void quantizeWeight(const int N, const int K, const float* B, const int ldb, int blocksize, int8_t* qB, float* scales,
                      int8_t* zero_points) {
    utils::parallel::Parallel2DRowMajor _para;
    utils::CpuBase cb;
    int bsize = blocksize == -1 ? K : blocksize;
    _para.update(K, N, bsize, 16, cb.mNumThreads);
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      int colidx, rowidx, rowsize, colsize;
      _para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        int rowremain = utils::remainsize(rowidx, K,
                                          rowsize);  // rowremain: src valid size. rowsize: padded size
        int colremain = utils::remainsize(colidx, N, colsize);
        quantRowBlock(B + rowidx * ldb + colidx, qB + rowidx * N + colidx, rowremain, colremain, ldb, N,
                      scales + rowidx / bsize * N + colidx,
                      zero_points == nullptr ? zero_points : zero_points + rowidx / bsize * N + colidx, bsize);
      }
    }
  }

  void reorderWeight(const int N, const int K, const int8_t* B, const int ldb, int8_t* dstptr) {
    utils::parallel::Parallel2DRowMajor _para;
    utils::CpuBase cb;
    _para.update(K, N, _GemmCore_T::KTILE, _GemmCore_T::NTILE, cb.mNumThreads);
    int KPad = utils::padto(K, _GemmCore_T::KTILE);
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      int colidx, rowidx, rowsize, colsize;
      _para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        int rowremain = utils::remainsize(rowidx, K,
                                          rowsize);  // rowremain: src valid size. rowsize: padded size
        int colremain = utils::remainsize(colidx, N, colsize);
        const auto src = B + rowidx * ldb + colidx;
        const auto dst = dstptr + rowidx * _GemmCore_T::NTILE + colidx * KPad;
        using PaddingInterleaveMNWType =
            kernel::wrapper::PaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW>;
        auto ret = PaddingInterleaveMNWType::template forward<ISA_T>(  //
            src, dst, rowremain, colremain, rowsize, colsize, ldb, KPad);
        assert(ret == JblasSuccess);
        (void)ret;
      }
    }
  }
};

class StorageWeightS8ScaleFp32PerChannelN : public StorageWeightS8ScaleFp32 {
 public:
  using Parent = StorageWeightS8ScaleFp32;
  StorageWeightS8ScaleFp32PerChannelN(jblas::gemm::GemmCoreType _type) : StorageWeightS8ScaleFp32(_type) {
    mType = static_cast<int>(WeightCompType::WeightS8ScaleFp32PerChannelN);
  }

  void resize(int NPad, int KPad, int K, bool IsSym = true) {
    PackedWeightKBlock::resize(NPad, KPad, K);
    StorageWeight8Bit::resize(NPad, KPad);
    Parent::CorrectionType::resize(NPad, 1, IsSym, true);  // force block number==1, updiv(KPad,K) may get 2
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class WeightS8ScaleFp32PerChannelN : public WeightS8ScaleFp32<_GemmCore_T, ISA_T> {
 public:
  using Parent = WeightS8ScaleFp32<_GemmCore_T, ISA_T>;
  using Param = typename Parent::Param;
  using StorageWeight = StorageWeightS8ScaleFp32PerChannelN;
  using SType = float;
  using WeightBaseFloat = jblas::prologue::gemm::WeightBase<float, ISA_T>;
  using Parallel = utils::parallel::Parallel2DRowMajor;
  virtual PackedWeight* createStorage(const int N, const int K, bool is_sym = true) {
    int KPad = utils::padto(K, _GemmCore_T::KTILE);
    int NPad = utils::padto(N, _GemmCore_T::NTILE);
    auto ptr = new StorageWeight(_GemmCore_T::TYPE);
    ptr->resize(NPad, KPad, K, is_sym);
    return ptr;
  }

  Parallel createParallel(const int N, const int K) {
    return Parallel();  // no runtime parallel forward
  }

  virtual void packQWeight(const int N, const int K, const int8_t* B, const int ldb, const float* scales,
                           const int8_t* zero_points, PackedWeight* ptr) override {
    auto stor = reinterpret_cast<StorageWeight*>(ptr);
    std::memcpy(stor->mSPtr, scales, N * sizeof(scales[0]));
    if (zero_points != nullptr) {
      std::memcpy(stor->mZPtr, zero_points, N * sizeof(zero_points[0]));
    }
    WeightS8ScaleFp32<_GemmCore_T, ISA_T>::reorderWeight(N, K, B, ldb, stor->mWPtr);
    utils::avector<float> deq(K * N);
    WeightS8ScaleFp32<_GemmCore_T, ISA_T>::unpackWeight(N, K, stor, deq.data(), N);
    WeightS8ScaleFp32<_GemmCore_T, ISA_T>::reduceWeight(N, K, K, deq.data(), ldb, stor->mRPtr, stor->mNPad);
  }
};

class StorageWeightS4ScaleFp32 : public prologue::weight_comp::PackedWeightKBlock,
                                 public StorageWeight4Bit,
                                 public StorageSimpleCorrection<float, int8_t, float> {
 public:
  using CorrectionType = StorageSimpleCorrection<float, int8_t, float>;
  StorageWeightS4ScaleFp32(jblas::gemm::GemmCoreType _gemm_core_type, JBLAS_SIGN_INT_TYPE _s4_type = S4_UNDEF)
      : prologue::weight_comp::PackedWeightKBlock(_gemm_core_type) {
    switch (_s4_type) {
      case S4_CLIP:
        mType = static_cast<int>(WeightCompType::WeightS4ClipScaleFp32);
        break;
      case S4_FULLRANGE:
        mType = static_cast<int>(WeightCompType::WeightS4FullRangeScaleFp32);
        break;
      default:
        break;
    }
  }

  void resize(int NPad, int KPad, int Block, bool IsSym = true) {
    PackedWeightKBlock::resize(NPad, KPad, Block);
    StorageWeight4Bit::resize(NPad, KPad);
    int nk_scale = utils::updiv(KPad, Block);
    CorrectionType::resize(NPad, nk_scale, IsSym, true);
  }

 protected:
  virtual size_t getDataSerializedSize() override {
    size_t totalsize = StorageWeight4Bit::myDataSerializedSize() + CorrectionType::myDataSerializedSize();
    return totalsize;
  }
  virtual void serializeDataToBuffer(void* buf) override {
    auto wptr = reinterpret_cast<int8_t*>(buf);
    StorageWeight4Bit::mySerializeDataToBuffer(wptr);
    CorrectionType::mySerializeDataToBuffer(wptr);
  }
  virtual void deserializeDataBuffer(void* buf, int memalloc) override {
    auto rptr = reinterpret_cast<int8_t*>(buf);
    StorageWeight4Bit::myDeserializeDataBuffer(rptr, memalloc);
    CorrectionType::myDeserializeDataBuffer(rptr, memalloc);
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T, JBLAS_SIGN_INT_TYPE S4_T>
class WeightS4ScaleFp32 : public WeightS8ScaleFp32<_GemmCore_T, ISA_T> {
 public:
  using Param = typename WeightS8ScaleFp32<_GemmCore_T, ISA_T>::Param;
  using WeightBaseFloat = jblas::prologue::gemm::WeightBase<float, ISA_T>;
  using StorageWeight = StorageWeightS4ScaleFp32;
  PackedWeight* createStorage(const int N, const int K, int blocksize, bool is_sym = true) override {
    int KPad = utils::padto(K, _GemmCore_T::KTILE);
    int NPad = utils::padto(N, _GemmCore_T::NTILE);
    auto ptr = new StorageWeight(_GemmCore_T::TYPE, S4_T);
    ptr->resize(NPad, KPad, blocksize <= 0 ? K : blocksize, is_sym);
    return ptr;
  }

  virtual void quantRowBlock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                             float* scales, int8_t* zero_points, int blocksize) {
    kernel::wrapper::QuantizeSignIntRowBlock::forward<ISA_T, S4_T>(srcptr, dstptr, row, col, ld_src, ld_dst, scales,
                                                                   zero_points, blocksize);
  }

  virtual void packQWeight(const int N, const int K, const int8_t* B, const int ldb, const float* scales,
                           const int8_t* zero_points, PackedWeight* ptr) override {
    auto stor = reinterpret_cast<StorageWeight*>(ptr);
    int rawnk_scale = utils::updiv(K, stor->mBlockSize);
    int nk_scale = utils::updiv(stor->mKPad, stor->mBlockSize);
#pragma omp parallel for
    for (int i = 0; i < nk_scale; i++) {  // padding copy
      if (i < rawnk_scale) {
        std::memcpy(stor->mSPtr + i * stor->mNPad, scales + i * N, N * sizeof(scales[0]));
        if (zero_points != nullptr) {
          std::memcpy(stor->mZPtr + i * stor->mNPad, zero_points + i * N, N * sizeof(zero_points[0]));
        }
      } else {
        std::memset(stor->mSPtr + i * stor->mNPad, 0, stor->mNPad * sizeof(stor->mSPtr[0]));
        if (zero_points != nullptr) {
          std::memset(stor->mZPtr + i * stor->mNPad, 0, stor->mNPad * sizeof(zero_points[0]));
        }
      }
    }
    utils::avector<int8_t> reorded(stor->mKPad * stor->mNPad);
    WeightS8ScaleFp32<_GemmCore_T, ISA_T>::reorderWeight(N, K, B, ldb, reorded.data());
    compressWeight(stor->mNPad, stor->mKPad, reorded.data(), stor->mNPad, stor->mWPtr);
    if (stor->has_reduce()) {
      utils::avector<float> deq(K * N);
      WeightS8ScaleFp32<_GemmCore_T, ISA_T>::unpackWeight(N, K, stor, deq.data(), N);
      WeightS8ScaleFp32<_GemmCore_T, ISA_T>::reduceWeight(N, K, stor->mBlockSize, deq.data(), ldb, stor->mRPtr,
                                                          stor->mNPad);
    }
  }

  void compressWeight(const int N, const int K, const int8_t* B, const int ldb, utils::bit4x2* dstptr) {
    utils::parallel::Parallel2DRowMajor _para;
    utils::CpuBase cb;
    _para.update(K, N, _GemmCore_T::KTILE, _GemmCore_T::NTILE, cb.mNumThreads);
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      int colidx, rowidx, rowsize, colsize;
      _para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        int rowremain = utils::remainsize(rowidx, K,
                                          rowsize);  // rowremain: src valid size. rowsize: padded size
        int colremain = utils::remainsize(colidx, N, colsize);
        auto ret = doCompress(B + rowidx * ldb + colidx, dstptr + rowidx * ldb / 2 + colidx / 2, rowremain, colremain,
                              ldb, ldb);
        assert(ret == JblasSuccess);
        (void)ret;
      }
    }
  }

  virtual inline JBLAS_CODE getWeight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                      const Param& _param) override {
    auto wptr = reinterpret_cast<const StorageWeight*>(_param.packedW);
    auto KPad = wptr->mKPad;
    auto bptr = wptr->mWPtr + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      kernel::wrapper::DecompressKBlockS4S8::forward<ISA_T, S4_T>(
          (utils::int4x2*)(bptr + i * KPad / 2), *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
          _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW,
          _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW);
    }
    *dststep = k_size;
    return JblasSuccess;
  }

  virtual inline JBLAS_CODE getWeight(float** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                      const Param& _param) override {
    auto wptr = reinterpret_cast<const StorageWeight*>(_param.packedW);
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->mWPtr + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      if constexpr (_GemmCore_T::PACK_ROW == 1) {
        kernel::wrapper::DecompressKBlockS4FP<float>::forward<ISA_T, float, S4_T>(
            (utils::int4x2*)(bptr + i * KPad / 2), *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
            _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW,
            _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, wptr->mSPtr + n_offset + i,
            wptr->mZPtr != nullptr ? wptr->mZPtr + n_offset + i : nullptr, k_offset, wptr->mBlockSize, NPad);
      } else {
        kernel::wrapper::DecompressKBlockS4FPPackRow<float>::forward<ISA_T, float, S4_T>(
            (utils::int4x2*)(bptr + i * KPad / 2), *dstptr + i * k_size, k_size, _GemmCore_T::NTILE, _GemmCore_T::NTILE,
            _GemmCore_T::NTILE, wptr->mSPtr + n_offset + i,
            wptr->mZPtr != nullptr ? wptr->mZPtr + n_offset + i : nullptr, k_offset, wptr->mBlockSize, NPad,
            _GemmCore_T::PACK_ROW);
      }
    }
    *dststep = k_size;
    return JblasSuccess;
  }

  virtual inline JBLAS_CODE getWeight(utils::bf16** dstptr, int* dststep, int k_size, int n_size, int k_offset,
                                      int n_offset, const Param& _param) {
    auto wptr = reinterpret_cast<const StorageWeight*>(_param.packedW);
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->mWPtr + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      kernel::wrapper::DecompressKBlockS4FP<utils::bf16>::forward<ISA_T, float, S4_T>(
          (utils::int4x2*)(bptr + i * KPad / 2), *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
          _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW,
          _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, wptr->mSPtr + n_offset + i,
          wptr->mZPtr != nullptr ? wptr->mZPtr + n_offset + i : nullptr, k_offset / _GemmCore_T::PACK_ROW,
          wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad);
    }
    *dststep = k_size;
    return JblasSuccess;
  }

  virtual JBLAS_CODE getScale(float** dstptr, int* dststep, int n_size, int k_size, int n_offset, int k_offset,
                              const Param& _param) override {
    auto wptr = reinterpret_cast<const StorageWeight*>(_param.packedW);
    auto NPad = wptr->mNPad;
    *dstptr = wptr->mSPtr + n_offset + k_offset / wptr->mBlockSize * NPad;
    *dststep = NPad;
    return JblasSuccess;
  }

  virtual JBLAS_CODE getReduce(float** dstptr, int* dststep, int n_size, int k_size, int n_offset, int k_offset,
                               const Param& _param) override {
    auto wptr = reinterpret_cast<const StorageWeight*>(_param.packedW);
    auto NPad = wptr->mNPad;
    *dstptr = wptr->mRPtr + n_offset + k_offset / wptr->mBlockSize * NPad;
    *dststep = NPad;
    return JblasSuccess;
  }

  virtual JBLAS_CODE getZp(int8_t** dstptr, int* dststep, int n_size, int k_size, int n_offset, int k_offset,
                           const Param& _param) override {
    auto wptr = reinterpret_cast<const StorageWeight*>(_param.packedW);
    *dstptr = wptr->mZPtr == nullptr ? nullptr : wptr->mZPtr + n_offset + k_offset / wptr->mBlockSize * wptr->mStep;
    *dststep = wptr->mStep;
    return JblasSuccess;
  }

 protected:
  virtual JBLAS_CODE doCompress(const int8_t* srcptr, void* dstptr, int row, int col, int ld_src, int ld_dst) {
    return kernel::wrapper::CompressS8S4<_GemmCore_T::NTILE>::template forward<ISA_T>(
        srcptr, reinterpret_cast<utils::int4x2*>(dstptr), row, col, ld_src,
        ld_dst);  // ld_dst here not stride
  }
};

class StorageWeightS4ScaleFp32PerChannelN : public StorageWeightS4ScaleFp32 {
 public:
  using Parent = StorageWeightS4ScaleFp32;
  StorageWeightS4ScaleFp32PerChannelN(jblas::gemm::GemmCoreType _type, JBLAS_SIGN_INT_TYPE _s4_type = S4_UNDEF)
      : StorageWeightS4ScaleFp32(_type) {
    switch (_s4_type) {
      case S4_CLIP:
        mType = static_cast<int>(WeightCompType::WeightS4ClipScaleFp32PerChannelN);
        break;
      case S4_FULLRANGE:
      default:
        break;
    }
  }

  void resize(int NPad, int KPad, int K, bool IsSym = true) {
    PackedWeightKBlock::resize(NPad, KPad, K);
    StorageWeight4Bit::resize(NPad, KPad);
    Parent::CorrectionType::resize(NPad, 1, IsSym, true);  // force block number = 1
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T, JBLAS_SIGN_INT_TYPE S4_T>
class WeightS4ScaleFp32PerChannelN : public WeightS8ScaleFp32PerChannelN<_GemmCore_T, ISA_T> {
 public:
  using Parent = WeightS8ScaleFp32PerChannelN<_GemmCore_T, ISA_T>;
  using Param = typename Parent::Param;
  using StorageWeight = StorageWeightS4ScaleFp32PerChannelN;
  PackedWeight* createStorage(const int N, const int K, bool is_sym = true) override {
    int KPad = utils::padto(K, _GemmCore_T::KTILE);
    int NPad = utils::padto(N, _GemmCore_T::NTILE);
    auto ptr = new StorageWeight(_GemmCore_T::TYPE, S4_T);
    ptr->resize(NPad, KPad, K, is_sym);
    return ptr;
  }

  virtual void quantRowBlock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                             float* scales, int8_t* zero_points, int blocksize) {
    kernel::wrapper::QuantizeSignIntRowBlock::forward<ISA_T, S4_T>(srcptr, dstptr, row, col, ld_src, ld_dst, scales,
                                                                   zero_points, blocksize);
  }

  virtual void packQWeight(const int N, const int K, const int8_t* B, const int ldb, const float* scales,
                           const int8_t* zero_points, PackedWeight* ptr) override {
    auto stor = reinterpret_cast<StorageWeight*>(ptr);
    std::memcpy(stor->mSPtr, scales, N * sizeof(scales[0]));
    if (zero_points != nullptr) {
      std::memcpy(stor->mZPtr, zero_points, N * sizeof(zero_points[0]));
    }
    utils::avector<int8_t> reorded(stor->mKPad * stor->mNPad);
    WeightS8ScaleFp32<_GemmCore_T, ISA_T>::reorderWeight(N, K, B, ldb, reorded.data());
    compressWeight(stor->mNPad, stor->mKPad, reorded.data(), stor->mNPad, stor->mWPtr);
    utils::avector<float> deq(K * N);
    WeightS8ScaleFp32<_GemmCore_T, ISA_T>::unpackWeight(N, K, stor, deq.data(), N);
    WeightS8ScaleFp32<_GemmCore_T, ISA_T>::reduceWeight(N, K, K, deq.data(), ldb, stor->mRPtr, stor->mNPad);
  }

  void compressWeight(const int N, const int K, const int8_t* B, const int ldb, utils::bit4x2* dstptr) {
    utils::parallel::Parallel2DRowMajor _para;
    utils::CpuBase cb;
    _para.update(K, N, _GemmCore_T::KTILE, _GemmCore_T::NTILE, cb.mNumThreads);
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      int colidx, rowidx, rowsize, colsize;
      _para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        int rowremain = utils::remainsize(rowidx, K,
                                          rowsize);  // rowremain: src valid size. rowsize: padded size
        int colremain = utils::remainsize(colidx, N, colsize);
        auto ret = doCompress(B + rowidx * ldb + colidx, dstptr + rowidx * ldb / 2 + colidx / 2, rowremain, colremain,
                              ldb, ldb);
        assert(ret == JblasSuccess);
        (void)ret;
      }
    }
  }

  virtual inline JBLAS_CODE getWeight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                      const Param& _param) override {
    auto wptr = reinterpret_cast<const StorageWeight*>(_param.packedW);
    auto KPad = wptr->mKPad;
    auto bptr = wptr->mWPtr + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      kernel::wrapper::DecompressKBlockS4S8::forward<ISA_T, S4_T>(
          (utils::int4x2*)(bptr + i * KPad / 2), *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
          _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW,
          _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW);
    }
    *dststep = k_size;
    return JblasSuccess;
  }

  virtual inline JBLAS_CODE getWeight(float** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                      const Param& _param) override {
    auto wptr = reinterpret_cast<const StorageWeight*>(_param.packedW);
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->mWPtr + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      if constexpr (_GemmCore_T::PACK_ROW == 1) {
        kernel::wrapper::DecompressPerNS4FP<float>::forward<ISA_T, float, S4_T>(
            (utils::int4x2*)(bptr + i * KPad / 2), *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
            _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW,
            _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, wptr->mSPtr + n_offset + i,
            wptr->mZPtr != nullptr ? wptr->mZPtr + n_offset + i : nullptr, k_offset, wptr->mBlockSize, NPad);
      } else {
        kernel::wrapper::DecompressPerNS4FPPackRow<float>::forward<ISA_T, float, S4_T>(
            (utils::int4x2*)(bptr + i * KPad / 2), *dstptr + i * k_size, k_size, _GemmCore_T::NTILE, _GemmCore_T::NTILE,
            _GemmCore_T::NTILE, wptr->mSPtr + n_offset + i,
            wptr->mZPtr != nullptr ? wptr->mZPtr + n_offset + i : nullptr, k_offset, wptr->mBlockSize, NPad,
            _GemmCore_T::PACK_ROW);
      }
    }
    *dststep = k_size;
    return JblasSuccess;
  }

 protected:
  virtual JBLAS_CODE doCompress(const int8_t* srcptr, void* dstptr, int row, int col, int ld_src, int ld_dst) {
    return kernel::wrapper::CompressS8S4<_GemmCore_T::NTILE>::template forward<ISA_T>(
        srcptr, reinterpret_cast<utils::int4x2*>(dstptr), row, col, ld_src,
        ld_dst);  // ld_dst here not stride
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
using WeightS4ClipScaleFp32PerN = WeightS4ScaleFp32PerChannelN<_GemmCore_T, ISA_T, S4_CLIP>;

class StorageWeightS4ScaleBf16 : public prologue::weight_comp::PackedWeightKBlock,
                                 public StorageWeight4Bit,
                                 public StorageSimpleCorrection<utils::bf16, int8_t, float> {
 public:
  using CorrectionType = StorageSimpleCorrection<utils::bf16, int8_t, float>;
  StorageWeightS4ScaleBf16(jblas::gemm::GemmCoreType _gemm_core_type, JBLAS_SIGN_INT_TYPE _s4_type = S4_UNDEF)
      : prologue::weight_comp::PackedWeightKBlock(_gemm_core_type) {
    switch (_s4_type) {
      case S4_CLIP:
        mType = static_cast<int>(WeightCompType::WeightS4ClipScaleBf16);
        break;
      case S4_FULLRANGE:
        mType = static_cast<int>(WeightCompType::WeightS4FullRangeScaleBf16);
        break;
      default:
        break;
    }
  }

  void resize(int NPad, int KPad, int Block, bool IsSym = true) {
    PackedWeightKBlock::resize(NPad, KPad, Block);
    StorageWeight4Bit::resize(NPad, KPad);
    int nk_scale = utils::updiv(KPad, Block);
    CorrectionType::resize(NPad, nk_scale, IsSym, true);
  }

 protected:
  virtual size_t getDataSerializedSize() override {
    size_t totalsize = StorageWeight4Bit::myDataSerializedSize() + CorrectionType::myDataSerializedSize();
    return totalsize;
  }
  virtual void serializeDataToBuffer(void* buf) override {
    auto wptr = reinterpret_cast<int8_t*>(buf);
    StorageWeight4Bit::mySerializeDataToBuffer(wptr);
    CorrectionType::mySerializeDataToBuffer(wptr);
  }
  virtual void deserializeDataBuffer(void* buf, int memalloc) override {
    auto rptr = reinterpret_cast<int8_t*>(buf);
    StorageWeight4Bit::myDeserializeDataBuffer(rptr, memalloc);
    CorrectionType::myDeserializeDataBuffer(rptr, memalloc);
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T, JBLAS_SIGN_INT_TYPE S4_T>
class WeightS4ScaleBf16 : public WeightS4ScaleFp32<_GemmCore_T, ISA_T, S4_T> {
 public:
  using WeightBaseFloat = jblas::prologue::gemm::WeightBase<float, ISA_T>;
  using SType = utils::bf16;
  using StorageWeight = StorageWeightS4ScaleBf16;
  PackedWeight* createStorage(const int N, const int K, int blocksize, bool is_sym = true) override {
    int KPad = utils::padto(K, _GemmCore_T::KTILE);
    int NPad = utils::padto(N, _GemmCore_T::NTILE);
    auto ptr = new StorageWeight(_GemmCore_T::TYPE, S4_T);
    ptr->resize(NPad, KPad, blocksize <= 0 ? K : blocksize, is_sym);
    return ptr;
  }

  virtual void packQWeight(const int N, const int K, const int8_t* B, const int ldb, const float* scales,
                           const int8_t* zero_points, PackedWeight* ptr) override {
    auto stor = reinterpret_cast<StorageWeight*>(ptr);
    int rawnk_scale = utils::updiv(K, stor->mBlockSize);
    int nk_scale = utils::updiv(stor->mKPad, stor->mBlockSize);
#pragma omp parallel for
    for (int i = 0; i < nk_scale; i++) {
      if (i < rawnk_scale) {
        for (int j = 0; j < N; j++) {
          *(stor->mSPtr + i * stor->mNPad + j) = utils::cast<float, utils::bf16>(*(scales + i * N + j));
          if (zero_points != nullptr) {
            std::memcpy(stor->mZPtr + i * stor->mNPad, zero_points + i * N, N * sizeof(zero_points[0]));
          }
        }
      } else {
        std::memset(stor->mSPtr + i * stor->mNPad, 0, stor->mNPad * sizeof(stor->mSPtr[0]));
        if (zero_points != nullptr) {
          std::memset(stor->mZPtr + i * stor->mNPad, 0, stor->mNPad * sizeof(zero_points[0]));
        }
      }
    }
    utils::avector<int8_t> reorded(stor->mKPad * stor->mNPad);
    WeightS8ScaleFp32<_GemmCore_T, ISA_T>::reorderWeight(N, K, B, ldb, reorded.data());
    WeightS4ScaleFp32<_GemmCore_T, ISA_T, S4_T>::compressWeight(stor->mNPad, stor->mKPad, reorded.data(), stor->mNPad,
                                                                stor->mWPtr);
  }

  virtual inline JBLAS_CODE getWeight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                      const PackedWeight* ptr) override {
    auto wptr = reinterpret_cast<const StorageWeight*>(ptr);
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->mWPtr + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      kernel::wrapper::DecompressKBlockS4S8::forward<ISA_T, S4_T>(
          (utils::int4x2*)bptr + i * KPad / 2, *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
          _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW,
          _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW);
    }
    *dststep = k_size;
    return JblasSuccess;
  }

  virtual inline JBLAS_CODE getWeight(float** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                      const PackedWeight* ptr) override {
    auto wptr = reinterpret_cast<const StorageWeight*>(ptr);
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->mWPtr + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      kernel::wrapper::DecompressKBlockS4FP<float>::forward<ISA_T, utils::bf16, S4_T>(
          (utils::int4x2*)bptr + i * KPad / 2, *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
          _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW,
          _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, wptr->mSPtr + n_offset + i,
          wptr->mZPtr != nullptr ? wptr->mZPtr + n_offset + i : nullptr, k_offset, wptr->mBlockSize, NPad);
    }
    *dststep = k_size;
    return JblasSuccess;
  }

  virtual inline JBLAS_CODE getWeight(utils::bf16** dstptr, int* dststep, int k_size, int n_size, int k_offset,
                                      int n_offset, const PackedWeight* ptr) override {
    auto wptr = reinterpret_cast<const StorageWeight*>(ptr);
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->mWPtr + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      kernel::wrapper::DecompressKBlockS4FP<utils::bf16>::forward<ISA_T, utils::bf16, S4_T>(
          (utils::int4x2*)bptr + i * KPad / 2, *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
          _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW,
          _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, wptr->mSPtr + n_offset + i,
          wptr->mZPtr != nullptr ? wptr->mZPtr + n_offset + i : nullptr, k_offset, wptr->mBlockSize, NPad);
    }
    *dststep = k_size;
    return JblasSuccess;
  }

  virtual inline JBLAS_CODE getScale(utils::bf16** dstptr, int* dststep, int n_size, int k_size, int n_offset,
                                     int k_offset, const PackedWeight* ptr) {
    auto wptr = reinterpret_cast<const StorageWeight*>(ptr);
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    *dstptr = wptr->mSPtr + n_offset + k_offset / wptr->mBlockSize * NPad;
    *dststep = NPad;
    return JblasSuccess;
  }

  virtual inline JBLAS_CODE getZp(utils::bf16** dstptr, int* dststep, int n_size, int k_size, int n_offset,
                                  int k_offset, const PackedWeight* ptr) {
    // no asym support for any kinds of fp4
    auto wptr = reinterpret_cast<const StorageWeight*>(ptr);
    auto NPad = wptr->mNPad;
    *dstptr = nullptr;
    *dststep = NPad;
    return JblasSuccess;
  }
};

class StorageWeightF4ScaleFp32 : public StorageWeightS4ScaleFp32 {
 public:
  StorageWeightF4ScaleFp32(jblas::gemm::GemmCoreType _gemm_core_type, JBLAS_F4_TYPE _f4_type = F4_UNDEF)
      : StorageWeightS4ScaleFp32(_gemm_core_type) {
    switch (_f4_type) {
      case FP4_BNB:
        mType = static_cast<int>(WeightCompType::WeightFp4BnbScaleFp32);
        break;
      case FP4_E2M1:
        mType = static_cast<int>(WeightCompType::WeightFp4E2M1ScaleFp32);
        break;
      case NF4:
        mType = static_cast<int>(WeightCompType::WeightNf4ScaleFp32);
        break;
      default:
        break;
    }
  }

  void resize(int NPad, int KPad, int Block, bool IsSym = true) {
    PackedWeightKBlock::resize(NPad, KPad, Block);
    StorageWeight4Bit::resize(NPad, KPad);
    int nk_scale = utils::updiv(KPad, Block);
    CorrectionType::resize(NPad, nk_scale, IsSym, false);
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
using WeightS4ClipScaleFp32 = WeightS4ScaleFp32<_GemmCore_T, ISA_T, S4_CLIP>;
template <class _GemmCore_T, JBLAS_ISA ISA_T>
using WeightS4FullRangeScaleFp32 = WeightS4ScaleFp32<_GemmCore_T, ISA_T, S4_FULLRANGE>;
template <class _GemmCore_T, JBLAS_ISA ISA_T>
using WeightS4ClipScaleBf16 = WeightS4ScaleBf16<_GemmCore_T, ISA_T, S4_CLIP>;
template <class _GemmCore_T, JBLAS_ISA ISA_T>
using WeightS4FullRangeScaleBf16 = WeightS4ScaleBf16<_GemmCore_T, ISA_T, S4_FULLRANGE>;

template <class _GemmCore_T, JBLAS_ISA ISA_T, JBLAS_F4_TYPE F4_T>
class WeightF4ScaleFp32 : public WeightS4ScaleFp32<_GemmCore_T, ISA_T, S4_CLIP> {
 public:
  using Param = typename WeightS8ScaleFp32<_GemmCore_T, ISA_T>::Param;
  using StorageWeight = StorageWeightF4ScaleFp32;
  PackedWeight* createStorage(const int N, const int K, int blocksize) {
    int KPad = utils::padto(K, _GemmCore_T::KTILE);
    int NPad = utils::padto(N, _GemmCore_T::NTILE);
    auto ptr = new StorageWeight(_GemmCore_T::TYPE, F4_T);
    ptr->resize(NPad, KPad, blocksize <= 0 ? K : blocksize);
    return ptr;
  }

  virtual inline JBLAS_CODE getWeight(float** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                      const Param& _param) override {
    auto wptr = reinterpret_cast<const StorageWeight*>(_param.packedW);
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->mWPtr + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      if constexpr (_GemmCore_T::PACK_ROW == 1) {
        kernel::wrapper::DecompressKBlockF4Fp<float>::forward<ISA_T, float, F4_T>(
            reinterpret_cast<utils::f4x2*>(bptr + i * KPad / 2), *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
            _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW,
            _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, wptr->mSPtr + n_offset + i, k_offset, wptr->mBlockSize, NPad);
      } else {
        kernel::wrapper::DecompressKBlockF4FPPackRow<float>::forward<ISA_T, float, F4_T>(
            (utils::f4x2*)(bptr + i * KPad / 2), *dstptr + i * k_size, k_size, _GemmCore_T::NTILE, _GemmCore_T::NTILE,
            _GemmCore_T::NTILE, wptr->mSPtr + n_offset + i, k_offset, wptr->mBlockSize, NPad, _GemmCore_T::PACK_ROW);
      }
    }
    *dststep = k_size;
    return JblasSuccess;
  }

  virtual inline JBLAS_CODE getWeight(utils::bf16** dstptr, int* dststep, int k_size, int n_size, int k_offset,
                                      int n_offset, const Param& _param) {
    auto wptr = reinterpret_cast<const StorageWeight*>(_param.packedW);
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->mWPtr + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      kernel::wrapper::DecompressKBlockF4Fp<utils::bf16>::forward<ISA_T, float, F4_T>(
          reinterpret_cast<utils::f4x2*>(bptr + i * KPad / 2), *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
          _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW,
          _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW, wptr->mSPtr + n_offset + i, k_offset / _GemmCore_T::PACK_ROW,
          wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad);
    }
    *dststep = k_size;
    return JblasSuccess;
  }
  virtual void packQWeight(const int N, const int K, const int8_t* B, const int ldb, const float* scales,
                           PackedWeight* ptr) {
    auto stor = reinterpret_cast<StorageWeight*>(ptr);
    int rawnk_scale = utils::updiv(K, stor->mBlockSize);
    int nk_scale = utils::updiv(stor->mKPad, stor->mBlockSize);
#pragma omp parallel for
    for (int i = 0; i < nk_scale; i++) {  // padding copy
      if (i < rawnk_scale) {
        std::memcpy(stor->mSPtr + i * stor->mNPad, scales + i * N, N * sizeof(scales[0]));
      } else {
        std::memset(stor->mSPtr + i * stor->mNPad, 0, stor->mNPad * sizeof(stor->mSPtr[0]));
      }
    }
    utils::avector<int8_t> reorded(stor->mKPad * stor->mNPad);
    WeightS8ScaleFp32<_GemmCore_T, ISA_T>::reorderWeight(N, K, B, ldb, reorded.data());
    WeightS4ScaleFp32<_GemmCore_T, ISA_T, S4_CLIP>::compressWeight(stor->mNPad, stor->mKPad, reorded.data(),
                                                                   stor->mNPad, stor->mWPtr);
  }

 protected:
  virtual void quantRowBlock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                             float* scales, int8_t* zero_points, int blocksize) override {
    kernel::wrapper::QuantizeF4RowBlock::forward<ISA_T, F4_T>(srcptr, dstptr, row, col, ld_src, ld_dst, scales,
                                                              zero_points, blocksize);
  }

  virtual JBLAS_CODE doCompress(const int8_t* srcptr, void* dstptr, int row, int col, int ld_src, int ld_dst) override {
    return kernel::wrapper::CompressFp4<_GemmCore_T::NTILE>::template forward<ISA_T>(
        srcptr, reinterpret_cast<utils::f4x2*>(dstptr), row, col, ld_src,
        ld_dst);  // ld_dst here not stride
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
using WeightFp4BnbScaleFp32 = WeightF4ScaleFp32<_GemmCore_T, ISA_T, FP4_BNB>;
template <class _GemmCore_T, JBLAS_ISA ISA_T>
using WeightFp4E2M1ScaleFp32 = WeightF4ScaleFp32<_GemmCore_T, ISA_T, FP4_E2M1>;
template <class _GemmCore_T, JBLAS_ISA ISA_T>
using WeightNf4ScaleFp32 = WeightF4ScaleFp32<_GemmCore_T, ISA_T, NF4>;

class PackedWeightParser {
 public:
  static PackedWeight* deserialBuffer(void* serialized_buf, int memalloc = 0) {
    auto rptr = reinterpret_cast<int8_t*>(serialized_buf);
    rptr += PackedWeight::TypeOffset;
    int mType = utils::deserialize<int>(rptr);
    if (mType >= int(WeightCompType::Begin) && mType < int(WeightCompType::End)) {
      rptr = reinterpret_cast<int8_t*>(serialized_buf);
      auto type = static_cast<WeightCompType>(mType);
      switch (type) {
        case WeightCompType::WeightS4FullRangeScaleFp32:
        case WeightCompType::WeightS4ClipScaleFp32: {
          auto ptr = new StorageWeightS4ScaleFp32(jblas::gemm::GemmCoreType::Undef);
          ptr->deserializeBuffer(rptr, memalloc);
          return ptr;
        }
        case WeightCompType::WeightS4FullRangeScaleBf16:
        case WeightCompType::WeightS4ClipScaleBf16: {
          auto ptr = new StorageWeightS4ScaleBf16(jblas::gemm::GemmCoreType::Undef);
          ptr->deserializeBuffer(rptr, memalloc);
          return ptr;
        }
        case WeightCompType::WeightS8ScaleFp32: {
          auto ptr = new StorageWeightS8ScaleFp32(jblas::gemm::GemmCoreType::Undef);
          ptr->deserializeBuffer(rptr, memalloc);
          return ptr;
        }
        case WeightCompType::WeightFp4BnbScaleFp32:
        case WeightCompType::WeightFp4E2M1ScaleFp32:
        case WeightCompType::WeightNf4ScaleFp32: {
          auto ptr = new StorageWeightF4ScaleFp32(jblas::gemm::GemmCoreType::Undef);
          ptr->deserializeBuffer(rptr, memalloc);
          return ptr;
        }
        case WeightCompType::WeightS8ScaleFp32PerChannelN: {
          auto ptr = new StorageWeightS8ScaleFp32PerChannelN(jblas::gemm::GemmCoreType::Undef);
          ptr->deserializeBuffer(rptr, memalloc);
          return ptr;
        }
        case WeightCompType::WeightS4ClipScaleFp32PerChannelN: {
          auto ptr = new StorageWeightS4ScaleFp32PerChannelN(jblas::gemm::GemmCoreType::Undef);
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
}  // namespace gemm_kblcok
}  // namespace weight_comp
}  // namespace prologue
namespace wrapper {
namespace gemm_kblock {

template <JBLAS_ISA _RT_ISA_T, class _GemmCore_T, template <class _T, JBLAS_ISA> class _PrologueA_T,
          template <class _T, JBLAS_ISA> class _PrologueB_T, template <JBLAS_ISA> class _Epilogue_T>
class GemmSLauncherKBlockPackWeight {
 public:
  static JBLAS_ISA constexpr RT_ISA = _RT_ISA_T;
  using GemmCore = _GemmCore_T;
  using PrologueA = _PrologueA_T<GemmCore, _RT_ISA_T>;
  using PrologueB = _PrologueB_T<GemmCore, _RT_ISA_T>;
  using Epilogue = _Epilogue_T<_RT_ISA_T>;
  using AType = typename GemmCore::AType;
  using AParam = typename PrologueA::Param;
  using QuanAParam = typename PrologueA::QParam;
  using BType = typename GemmCore::BType;
  using BParam = typename PrologueB::Param;
  using BSType = typename PrologueB::SType;
  using CType = typename GemmCore::CType;
  using EpiParam = typename Epilogue::Param;
  static_assert(GemmCore::ISA <= _RT_ISA_T, "RunTime ISA should cover GEMM's ISA");
  struct Param {
    const int M, N, K;
    const AParam paramA;
    const BParam paramB;
    const EpiParam paramC;
    void* workspace;
  };
  struct ParallelConfig {
    const int rowidx, colidx;
    const int rowsize, colsize;
    const int MStep, NStep, KStep;
    const size_t StackSize;
  };
  GemmCore mGemmCore;
  PrologueA mProA;
  PrologueB mProB;
  Epilogue mEpilogue;

  template <typename... Eltops>
  void launch(const ParallelConfig& _config, const Param& _param, Eltops... ops) {
    auto blkptr = reinterpret_cast<const prologue::weight_comp::PackedWeightKBlock*>(_param.paramB.packedW);
    int rowremain = utils::remainsize(_config.rowidx, _param.M, _config.rowsize);
    int colremain = utils::remainsize(_config.colidx, _param.N, _config.colsize);
    auto StackTmp = alloca(_config.StackSize);
    auto tmpB = (BType*)(StackTmp);
    auto tmpA = (AType*)(tmpB + _config.NStep * _config.KStep);
    auto tmpC = (CType*)(tmpA + GemmCore::MTILE * _config.KStep);
    for (int itern = 0; itern < colremain; itern += _config.NStep) {
      int n_remain = utils::remainsize(itern, colremain, _config.NStep);
      for (int iterm = 0; iterm < rowremain; iterm += _config.MStep) {
        int m_remain = utils::remainsize(iterm, rowremain, _config.MStep);
        run_block(_config, _param, blkptr, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpC, ops...);
      }
    }
  }

 protected:
  template <typename... Eltops>
  void run_block(const ParallelConfig& _config, const Param& _param,
                 const prologue::weight_comp::PackedWeightKBlock* blkptr, int blk_m, int blk_n, int blk_msize,
                 int blk_nsize, AType* tmpA, BType* tmpB, CType* tmpC, Eltops... ops) {
    int n_padded = utils::padto(blk_nsize, GemmCore::NTILE);
    auto c_tile_ptr = tmpC;
    auto c_block_ptr = (float*)(c_tile_ptr + GemmCore::NTILE * GemmCore::MTILE);
    for (int iterk = 0; iterk < _param.K; iterk += _config.KStep) {
      int k_remain = utils::remainsize(iterk, _param.K, _config.KStep);
      int k_padded = utils::padto(k_remain, GemmCore::KTILE);
      auto bptr_cache = tmpB;
      int bcache_step = 0;
      mProB.getWeight(&bptr_cache, &bcache_step, k_padded, n_padded, iterk, _config.colidx + blk_n, _param.paramB);
      BSType* wscale_ptr = nullptr;
      int wscale_step = 0;
      mProB.getScale(&wscale_ptr, &wscale_step, n_padded, k_padded, (blk_n + _config.colidx), iterk, _param.paramB);

      int bcache_stride = bcache_step * sizeof(BType);
      for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
        int m_remain = utils::remainsize(i, blk_msize, GemmCore::MTILE);
        auto cptr_cache = c_block_ptr + i * _config.NStep;
        int ccache_stride = _config.NStep * sizeof(CType);

        AType* aptr_cache = nullptr;
        int acache_step = 0;
        mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_padded, (blk_m + i + _config.rowidx),
                            iterk);
        float* ascale_ptr = nullptr;
        int ascale_step = 0;
        mProA.getScale(&ascale_ptr, &ascale_step, _param.paramA, m_remain, k_padded, (blk_m + i + _config.rowidx),
                       iterk);
        AType* azp_ptr = tmpA;
        int azp_step = _config.KStep;
        mProA.getZp(&azp_ptr, &azp_step, _param.paramA, m_remain, k_padded, (blk_m + i + _config.rowidx), iterk);
        mGemmCore.forward(aptr_cache, bptr_cache, cptr_cache, azp_ptr, ascale_ptr, ascale_step, wscale_ptr, wscale_step,
                          m_remain, n_padded, k_padded, blkptr->mBlockSize, acache_step * sizeof(AType), bcache_stride,
                          ccache_stride, iterk);
      }
    }
    mEpilogue.forward(c_block_ptr, _config.NStep, (_config.rowidx + blk_m), _config.colidx + blk_n, blk_msize,
                      blk_nsize, _param.paramC, ops...);
  }
};

template <JBLAS_ISA _RT_ISA_T, class _GemmCore_T, template <class _T, JBLAS_ISA> class _PrologueA_T,
          template <class _T, JBLAS_ISA> class _PrologueB_T, template <JBLAS_ISA> class _Epilogue_T>
class GemmLauncherKBlock {
 public:
  static JBLAS_ISA constexpr RT_ISA = _RT_ISA_T;
  using GemmCore = _GemmCore_T;
  using PrologueA = _PrologueA_T<GemmCore, _RT_ISA_T>;
  using PrologueB = _PrologueB_T<GemmCore, _RT_ISA_T>;
  using Epilogue = _Epilogue_T<_RT_ISA_T>;
  using AType = typename GemmCore::AType;
  using AParam = typename PrologueA::Param;
  using QuanAParam = typename PrologueA::QParam;
  using BType = typename GemmCore::BType;
  using BParam = typename PrologueB::Param;
  using BSType = typename PrologueB::SType;
  using CType = typename GemmCore::CType;
  using EpiParam = typename Epilogue::Param;
  static_assert(GemmCore::ISA <= _RT_ISA_T, "RunTime ISA should cover GEMM's ISA");
  struct Param {
    const int M, N, K;
    const AParam paramA;
    const BParam paramB;
    const EpiParam paramC;
    void* workspace;
  };
  struct ParallelConfig {
    const int rowidx, colidx;
    const int rowsize, colsize;
    const int MStep, NStep, KStep;
    const size_t StackSize;
  };
  GemmCore mGemmCore;
  PrologueA mProA;
  PrologueB mProB;
  Epilogue mEpilogue;

  template <typename... Eltops>
  void launch(const ParallelConfig& _config, const Param& _param, Eltops... ops) {
    auto blkptr = reinterpret_cast<const prologue::weight_comp::PackedWeightKBlock*>(_param.paramB.packedW);
    int rowremain = utils::remainsize(_config.rowidx, _param.M, _config.rowsize);
    int colremain = utils::remainsize(_config.colidx, _param.N, _config.colsize);
    auto StackTmp = alloca(_config.StackSize);
    auto tmpB = (BType*)(StackTmp);
    auto tmpA = (AType*)(tmpB + _config.NStep * _config.KStep);
    auto tmpC = (CType*)(tmpA + GemmCore::MTILE * _config.KStep);
    for (int itern = 0; itern < colremain; itern += _config.NStep) {
      int n_remain = utils::remainsize(itern, colremain, _config.NStep);
      for (int iterm = 0; iterm < rowremain; iterm += _config.MStep) {
        int m_remain = utils::remainsize(iterm, rowremain, _config.MStep);
        run_block(_config, _param, blkptr, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpC, ops...);
      }
    }
  }

 protected:
  template <typename... Eltops>
  void run_block(const ParallelConfig& _config, const Param& _param,
                 const prologue::weight_comp::PackedWeightKBlock* blkptr, int blk_m, int blk_n, int blk_msize,
                 int blk_nsize, AType* tmpA, BType* tmpB, CType* tmpC, Eltops... ops) {
    int n_padded = utils::padto(blk_nsize, GemmCore::NTILE);
    auto c_tile_ptr = tmpC;
    auto c_block_ptr = (float*)(c_tile_ptr + GemmCore::NTILE * GemmCore::MTILE);
    for (int iterk = 0; iterk < _param.K; iterk += _config.KStep) {
      int k_remain = utils::remainsize(iterk, _param.K, _config.KStep);
      int k_padded = utils::padto(k_remain, GemmCore::KTILE);
      auto bptr_cache = tmpB;
      int bcache_step = 0;
      mProB.getWeight(&bptr_cache, &bcache_step, k_padded, n_padded, iterk, _config.colidx + blk_n, _param.paramB);
      BSType* wscale_ptr = nullptr;
      int wscale_step = 0;
      mProB.getScale(&wscale_ptr, &wscale_step, n_padded, k_padded, (blk_n + _config.colidx), iterk, _param.paramB);
      float* reduce_ptr = nullptr;
      mProB.getReduce(&reduce_ptr, &wscale_step, n_padded, k_padded, (blk_n + _config.colidx), iterk, _param.paramB);
      int bcache_stride = bcache_step * sizeof(BType);
      for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
        int m_remain = utils::remainsize(i, blk_msize, GemmCore::MTILE);
        auto cptr_cache = c_block_ptr + i * _config.NStep;
        int ccache_stride = _config.NStep * sizeof(CType);

        AType* aptr_cache = nullptr;
        int acache_step = 0;
        mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_padded, (blk_m + i + _config.rowidx),
                            iterk);
        float* ascale_ptr = nullptr;
        int ascale_step = 0;
        mProA.getScale(&ascale_ptr, &ascale_step, _param.paramA, m_remain, k_padded, (blk_m + i + _config.rowidx),
                       iterk);
        AType* azp_ptr = tmpA;
        int azp_step = _config.KStep;
        mProA.getZp(&azp_ptr, &azp_step, _param.paramA, m_remain, k_padded, (blk_m + i + _config.rowidx), iterk);
        mGemmCore.forward(aptr_cache, bptr_cache, cptr_cache, azp_ptr, ascale_ptr, ascale_step, wscale_ptr, reduce_ptr,
                          wscale_step, m_remain, n_padded, k_padded, blkptr->mBlockSize, acache_step * sizeof(AType),
                          bcache_stride, ccache_stride, iterk);
      }
    }
    mEpilogue.forward(c_block_ptr, _config.NStep, (_config.rowidx + blk_m), _config.colidx + blk_n, blk_msize,
                      blk_nsize, _param.paramC, ops...);
  }
};

template <class _Launcher_T, template <class _T> class _Parallel_T>
class GemmInterfaceKBlockPackWeight {
 public:
  using Arguments = typename _Launcher_T::Param;
  using Config = typename _Launcher_T::ParallelConfig;
  using ActivationType = typename _Launcher_T::PrologueA;
  using WeightType = typename _Launcher_T::PrologueB;
  using Epilogue = typename _Launcher_T::Epilogue;
  using GemmCore = typename _Launcher_T::GemmCore;
  using Parallel = _Parallel_T<GemmCore>;

  ActivationType* getActivationPtr() { return &mLauncher.mProA; }

  WeightType* getWeightPtr() { return &mLauncher.mProB; }

  template <typename... Eltops>
  JBLAS_CODE compute(const Arguments& _param, Eltops... ops) {
    auto bptr = reinterpret_cast<const prologue::weight_comp::PackedWeightKBlock*>(_param.paramB.packedW);
    auto cb = utils::CpuBase();
    auto para = Parallel();
    para.update(_param.M, _param.N, _param.K, bptr->mBlockSize, cb.mNumThreads);
    auto paraA = mLauncher.mProA.createParallel(_param.M, _param.K, bptr->mBlockSize);
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      mLauncher.mProA.launch(_param.paramA, tidx, paraA);
#pragma omp barrier
      int colidx, rowidx, rowsize, colsize;
      para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        Config _config{rowidx,          colidx,          rowsize,         colsize,
                       para.getMStep(), para.getNStep(), para.getKStep(), cb.mL2Cache};
        mLauncher.launch(_config, _param, ops...);
      }
    }
    return JblasSuccess;
  }

 protected:
  _Launcher_T mLauncher;
};

template <class _Launcher_T, template <class _T> class _Parallel_T>
class GemmInterfaceKblockParallelAB {
 public:
  using Arguments = typename _Launcher_T::Param;
  using Config = typename _Launcher_T::ParallelConfig;
  using ActivationType = typename _Launcher_T::PrologueA;
  using WeightType = typename _Launcher_T::PrologueB;
  using Epilogue = typename _Launcher_T::Epilogue;
  using GemmCore = typename _Launcher_T::GemmCore;
  using Parallel = _Parallel_T<GemmCore>;

  ActivationType* getActivationPtr() { return &mLauncher.mProA; }

  WeightType* getWeightPtr() { return &mLauncher.mProB; }

  template <bool _LaunchA, bool _LaunchB>
  JBLAS_CODE compute(const Arguments& _param) {
    auto bptr = reinterpret_cast<const prologue::weight_comp::PackedWeightKBlock*>(_param.paramB.packedW);
    auto paraA = getActivationPtr()->createParallel(_param.M, _param.K, _param.KBlock);
    auto paraB = getWeightPtr()->createParallel(_param.K, _param.N, _param.KBlock);
    auto para = Parallel();
    auto cb = utils::CpuBase();
    if (para.update(_param.M, _param.N, _param.K, bptr->mBlockSize, cb.mNumThreads)) {
      static bool dbgprint = false;
      if (dbgprint) {
        para.print();
        dbgprint = false;
      }
    }
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      if constexpr (_LaunchA) {
        getActivationPtr()->launch(_param.paramA, tidx, paraA);
      }
      if constexpr (_LaunchB) {
        getWeightPtr()->launch(_param.paramB, tidx, paraB);
      }
      if constexpr (_LaunchA || _LaunchB) {
#pragma omp barrier
        (void)(0);  // make msvc happy with c++20
      }
      int colidx, rowidx, rowsize, colsize;
      para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        Config _config{rowidx,          colidx,          rowsize,         colsize,
                       para.getMStep(), para.getNStep(), para.getKStep(), cb.mL2Cache};
        mLauncher.launch(_config, _param);
      }
    }
    return JblasSuccess;
  }

 protected:
  _Launcher_T mLauncher;
};

}  // namespace gemm_kblock
namespace gemm_default {
namespace weight_comp {
namespace avx512f {
JBLAS_ISA constexpr DefaultISA = JblasAVX512F;
using GemmKernelS4FullRangeFp32KBlock = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512F, jblas::prologue::gemm::ActivationBase,
        jblas::prologue::weight_comp::gemm_kblcok::WeightS4FullRangeScaleFp32,
        jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    DefaultParallel>;
using GemmKernelS4ClipFp32KBlock = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512F, jblas::prologue::gemm::ActivationBase,
        jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32,
        jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    DefaultParallel>;
}  // namespace avx512f
namespace avx512_vnni {
JBLAS_ISA constexpr DefaultISA = JblasAVX512_VNNI;
using GemmSKernelDynamicS4ClipFp32KBlock = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
        DefaultISA, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
        jblas::prologue::gemm::ActivationF32U8KBlockQuantize,
        jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32,
        jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;

}  // namespace avx512_vnni
namespace amx_bf16 {
JBLAS_ISA constexpr DefaultISA = JblasAMX_BF16;
using GemmKernelS4FullRangeFp32KBlock = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16,
        jblas::prologue::gemm::ActivationConverterFp32,  // activation fp32->bf16
        jblas::prologue::weight_comp::gemm_kblcok::WeightS4FullRangeScaleFp32,
        jblas::epilogue::gemm::AccumulatorWriteBackFp32>,  // output fp32->fp32
    DefaultParallel>;
using GemmKernelS4ClipFp32KBlock = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16,
        jblas::prologue::gemm::ActivationConverterFp32,  // activation fp32->bf16
        jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32,
        jblas::epilogue::gemm::AccumulatorWriteBackFp32>,  // output fp32->fp32
    DefaultParallel>;
using GemmKernelFp4KBlock = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16,  // MXNXK = 16x64x32
        jblas::prologue::gemm::ActivationConverterFp32,           // activation fp32->bf16
        jblas::prologue::weight_comp::gemm_kblcok::WeightFp4BnbScaleFp32,
        jblas::epilogue::gemm::AccumulatorWriteBackFp32>,  // output fp32->fp32
    DefaultParallel>;
}  // namespace amx_bf16
namespace amx_int8 {
JBLAS_ISA constexpr DefaultISA = JblasAMX_INT8;
using GemmSKernelDynamicS4ClipFp32KBlock = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
        DefaultISA, jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
        jblas::prologue::gemm::ActivationF32S8KBlockQuantize,
        jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32,
        jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
using GemmSKernelDynamicS4FullRangeFp32KBlock = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
        DefaultISA, jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
        jblas::prologue::gemm::ActivationF32S8KBlockQuantize,
        jblas::prologue::weight_comp::gemm_kblcok::WeightS4FullRangeScaleFp32,
        jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;

using GemmDynamicS8Fp32PerChannelN = jblas::wrapper::gemm_pack_weight::GemmInterfaceParallelAB<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_16x48_AMX_S8S8,  //
        jblas::prologue::gemm::ActivationF32S8KBlockQuantize,
        jblas::prologue::weight_comp::gemm_kblcok::WeightS8ScaleFp32PerChannelN,
        jblas::epilogue::gemm::DequantInt32ToFp32>,
    jblas::utils::parallel::Parallel2DGemm>;
}  // namespace amx_int8
}  // namespace weight_comp
}  // namespace gemm_default
}  // namespace wrapper
}  // namespace jblas
