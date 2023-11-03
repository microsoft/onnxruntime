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
#include "jit_blas_storage.h"
#include "jit_blas_device.h"
#include "jit_blas_parallel.h"
#include "kernel_wrapper.h"

namespace jblas {
namespace prologue_b {
namespace gemm {

template <typename WT, JBLAS_ISA ISA_T>
static inline void transposeWeight(const int Row, const int Col, const WT* src, const int ld_src, WT* dst,
                                   const int ld_dst, parallel::IThreading* threading) {
  jblas::parallel::Scheduler2D _para;
  _para.update({threading->num_threads(), Row, Col, 16, 16});
  threading->parallel_for([&](int tidx) {
    jblas::parallel::ThreadProblem2D thdp{tidx};
    _para.getIndex(thdp);
    if (thdp.valid) {
      kernel::wrapper::Transpose2D<float>::template forward<ISA_T>(src + thdp.loc[0] * ld_src + thdp.loc[1],
                                                                   dst + thdp.loc[0] + thdp.loc[1] * ld_dst,
                                                                   thdp.size[0], thdp.size[1], ld_src, ld_dst);
    }
  });
}

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class WeightPack {
 public:
  using WType = typename _GemmCore_T::BType;
  using StorageType = storage::gemm::StoragePackedWeight;
  struct Param {
    const WType* B;
    const int ldb;
    StorageType* packedW;
  };

  StorageType createStorage(int n, int k) {
    int KPad = utils::padto(k, _GemmCore_T::KTILE);
    int NPad = utils::padto(n, _GemmCore_T::NTILE);
    StorageType tmp(_GemmCore_T::TYPE);
    tmp.resize(NPad, KPad, n, k, utils::jblas_dtype<WType>);
    return tmp;
  }

  void packWeightTranspose(const int N, const int K, const Param& _param, parallel::IThreading* threading) {
    auto B_NT = utils::amalloc<WType>((size_t)N * K);
    transposeWeight<WType, ISA_T>(N, K, _param.B, _param.ldb, B_NT, N, threading);
    packWeight(N, K, {B_NT, N, _param.packedW}, threading);
    utils::afree(B_NT);
  }

  // from KxN int8 symmetric weight to packed N//NtilexKPadxNTile int4 weight
  void packWeight(const int N, const int K, const Param& _param, parallel::IThreading* threading) {
    parallel::Scheduler2D _para({threading->num_threads(), K, N, _GemmCore_T::KTILE, _GemmCore_T::NTILE});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp{tidx};
      _para.getIndex(thdp);
      if (thdp.valid) {
        run(_param, thdp);
      }
    });
  }

  void run(const Param& _param, parallel::ThreadProblem2D& thdp) {
    auto packedw = _param.packedW;
    auto rowpadded = utils::padto(thdp.size[0], _GemmCore_T::KTILE);
    auto colpadded = utils::padto(thdp.size[1], _GemmCore_T::NTILE);
    const auto src = _param.B + thdp.loc[0] * _param.ldb + thdp.loc[1];
    const auto dst = packedw->template get<WType>() + thdp.loc[0] * _GemmCore_T::NTILE + thdp.loc[1] * packedw->mKPad;
    using PaddingInterleaveMNWType = kernel::wrapper::PaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW>;
    auto ret = PaddingInterleaveMNWType::template forward<ISA_T>(  //
        src, dst, thdp.size[0], thdp.size[1], rowpadded, colpadded, _param.ldb, packedw->mKPad);
    assert(ret == JblasSuccess);
    (void)ret;
  }

  inline JBLAS_CODE getWeight(WType** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                              const Param param, void* tmpcache, size_t cachesize) {
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

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class WeightKBlockS8 {
 public:
  using StorageWeight = storage::gemm::StorageWeightKBlockS8;
  using BType = typename _GemmCore_T::BType;
  struct Param {
    const storage::gemm::WeightKBlockBase* packedW;
  };

  StorageWeight createStorage(int n, int k, int blocksize, JBLAS_DTYPE scat, JBLAS_DTYPE redt, bool is_asym) {
    int KPad = utils::padto(k, _GemmCore_T::KTILE);
    int NPad = utils::padto(n, _GemmCore_T::NTILE);
    StorageWeight tmp(_GemmCore_T::TYPE);
    tmp.resize(NPad, KPad, blocksize <= 0 ? KPad : blocksize, n, k, scat, redt, is_asym);
    return tmp;
  }

  virtual void packTransposeWeight(const int N, const int K, const float* B, const int ldb, void* stor,
                                   parallel::IThreading* threading) {
    auto B_NT = utils::amalloc<float>((size_t)N * K);
    transposeWeight<float, ISA_T>(N, K, B, ldb, B_NT, N, threading);
    packWeight(N, K, B_NT, N, stor, threading);
    utils::afree(B_NT);
  }

  // from packed N//NtilexKPadxNTile int8 weight to KxN f32 weight
  virtual void unpackTransposeWeight(const int N, const int K, void* stor, float* B, const int ldb,
                                     parallel::IThreading* threading) {
    auto B_NT = utils::amalloc<float>((size_t)N * K);
    unpackWeight(N, K, stor, B_NT, N, threading);
    transposeWeight<float, ISA_T>(K, N, B_NT, N, B, ldb, threading);
    utils::afree(B_NT);
  }

  // from KxN f32 weight to packed N//NtilexKPadxNTile int8 weight
  virtual void packWeight(const int N, const int K, const float* B, const int ldb, void* stor,
                          parallel::IThreading* threading) {
    auto tmpq = utils::amalloc<int8_t>((size_t)N * K);
    auto ptr = reinterpret_cast<StorageWeight*>(stor);
    int nk_scale = utils::updiv(K, ptr->mBlockSize);
    auto ssize = (size_t)N * nk_scale;
    auto Tscales = utils::amalloc<float>(ssize);
    auto Tzps = utils::amalloc<int8_t>(ptr->mIsAsym ? ssize : 0);
    quantizeWeight(N, K, B, ldb, ptr->mBlockSize, tmpq, Tscales, Tzps, ptr->mDType, threading);
    packQWeight(N, K, tmpq, ldb, Tscales, Tzps, stor, threading);
    utils::afree(tmpq);
    utils::afree(Tscales);
    utils::afree(Tzps);
  }

  virtual void unpackWeight(const int N, const int K, void* stor, float* B, const int ldb,
                            parallel::IThreading* threading) {
    parallel::Scheduler2D _para({threading->num_threads(), K, N, _GemmCore_T::KTILE, _GemmCore_T::NTILE});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp{tidx};
      _para.getIndex(thdp);
      if (thdp.valid) {
        auto rowpad = utils::padto(thdp.size[0], _GemmCore_T::KTILE);
        auto colpad = utils::padto(thdp.size[1], _GemmCore_T::NTILE);
        auto dequant = utils::amalloc<float>((size_t)rowpad * colpad);
        auto dstptr = dequant;
        int dststep = 0;
        size_t constexpr CacheSize = size_t(100) << 10;
        int8_t tmpcache[CacheSize];
        getWeight(&dstptr, &dststep, rowpad, colpad, thdp.loc[0], thdp.loc[1], {(storage::gemm::WeightKBlockBase*)stor},
                  tmpcache, CacheSize);
        kernel::wrapper::RevertPaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW>::template forward<ISA_T>(
            dstptr, B + thdp.loc[0] * ldb + thdp.loc[1], thdp.size[0], thdp.size[1], rowpad, colpad, dststep, ldb);
        utils::afree(dequant);
      }
    });
  }

  virtual void unpackWeight(const int N, const int K, void* stor, int8_t* B, const int ldb,
                            parallel::IThreading* threading) {
    parallel::Scheduler2D _para({threading->num_threads(), K, N, _GemmCore_T::KTILE, _GemmCore_T::NTILE});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp{tidx};
      _para.getIndex(thdp);
      if (thdp.valid) {
        auto rowpad = utils::padto(thdp.size[0], _GemmCore_T::KTILE);
        auto colpad = utils::padto(thdp.size[1], _GemmCore_T::NTILE);
        auto dequant = utils::amalloc<int8_t>((size_t)rowpad * colpad);
        auto dstptr = dequant;
        int dststep = 0;
        size_t constexpr CacheSize = size_t(100) << 10;
        int8_t tmpcache[CacheSize];
        getWeight(&dstptr, &dststep, rowpad, colpad, thdp.loc[0], thdp.loc[1], {(storage::gemm::WeightKBlockBase*)stor},
                  tmpcache, CacheSize);
        kernel::wrapper::RevertPaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW>::template forward<ISA_T>(
            dstptr, B + thdp.loc[0] * ldb + thdp.loc[1], thdp.size[0], thdp.size[1], rowpad, colpad, dststep, ldb);
        utils::afree(dequant);
      }
    });
  }

  virtual void setQuantCorrection(const int N, const int K, const int8_t* zero_points, const float* scales, void* ptr,
                                  parallel::IThreading* threading) {
    auto stor = reinterpret_cast<StorageWeight*>(ptr);
    int rawnk_scale = utils::updiv(K, stor->mBlockSize);
    int nk_scale = utils::updiv(stor->mKPad, stor->mBlockSize);
    parallel::Scheduler2D _para({threading->num_threads(), 1, nk_scale, 1, 1});
    if (stor->mScaT == JBLAS_DTYPE::F32) {  // fp32 to fp32 direct copy
      threading->parallel_for([&](int tidx) {
        parallel::ThreadProblem2D thdp{tidx};
        _para.getIndex(thdp);
        if (thdp.valid) {
          for (int i = thdp.loc[1]; i < thdp.loc[1] + thdp.size[1]; i++) {
            if (i < rawnk_scale) {
              if (scales != nullptr)
                std::memcpy(stor->template SPtr<float>() + i * stor->mNPad, scales + i * N, N * sizeof(scales[0]));
              if (zero_points != nullptr)
                std::memcpy(stor->template ZPtr<int8_t>() + i * stor->mNPad, zero_points + i * N,
                            N * sizeof(zero_points[0]));
            } else {
              if (scales != nullptr)
                std::memset(stor->template SPtr<float>() + i * stor->mNPad, 0, stor->mNPad * sizeof(float));
              if (zero_points != nullptr)
                std::memset(stor->template ZPtr<int8_t>() + i * stor->mNPad, 0, stor->mNPad * sizeof(zero_points[0]));
            }
          }
        }
      });
    } else if (stor->mScaT == JBLAS_DTYPE::BF16) {
      threading->parallel_for([&](int tidx) {
        parallel::ThreadProblem2D thdp{tidx};
        _para.getIndex(thdp);
        if (thdp.valid) {
          for (int i = thdp.loc[1]; i < thdp.loc[1] + thdp.size[1]; i++) {
            if (i < rawnk_scale) {
              if (scales != nullptr) {
                for (size_t j = 0; j < N; j++) {
                  stor->template SPtr<utils::bf16>()[j + i * stor->mNPad] = static_cast<utils::bf16>(scales[i * N + j]);
                }
              }
              if (zero_points != nullptr) {
                std::memcpy(stor->template ZPtr<int8_t>() + i * stor->mNPad, zero_points + i * N,
                            N * sizeof(zero_points[0]));
              }
            } else {
              if (scales != nullptr)
                std::memset(stor->template SPtr<utils::bf16>() + i * stor->mNPad, 0, stor->mNPad * sizeof(utils::bf16));
              if (zero_points != nullptr)
                std::memset(stor->template ZPtr<int8_t>() + i * stor->mNPad, 0, stor->mNPad * sizeof(zero_points[0]));
            }
          }
        }
      });
    }
  }

  virtual void setTransposeQuantCorrection(const int N, const int K, const int8_t* zero_points, const float* scales,
                                           void* ptr, parallel::IThreading* threading) {
    auto stor = reinterpret_cast<StorageWeight*>(ptr);
    int rawnk_scale = utils::updiv(K, stor->mBlockSize);
    int nk_scale = utils::updiv(stor->mKPad, stor->mBlockSize);
    parallel::Scheduler2D _para({threading->num_threads(), 1, nk_scale, 1, 1});
    if (stor->mScaT == JBLAS_DTYPE::F32) {  // fp32 to fp32 direct copy
      threading->parallel_for([&](int tidx) {
        parallel::ThreadProblem2D thdp{tidx};
        _para.getIndex(thdp);
        if (thdp.valid) {
          if (scales) {
            for (int i = thdp.loc[1]; i < thdp.loc[1] + thdp.size[1]; i++) {
              if (i < rawnk_scale) {
                for (size_t j = 0; j < N; j++) {
                  stor->template SPtr<float>()[i * stor->mNPad + j] = scales[j * rawnk_scale + i];
                }
              } else {
                std::memset(stor->template SPtr<float>() + i * stor->mNPad, 0, stor->mNPad * sizeof(float));
              }
            }
          }
          if (stor->mIsAsym && zero_points) {
            for (int i = thdp.loc[1]; i < thdp.loc[1] + thdp.size[1]; i++) {
              if (i < rawnk_scale) {
                for (size_t j = 0; j < N; j++) {
                  stor->template ZPtr<int8_t>()[i * stor->mNPad + j] = zero_points[j * rawnk_scale + i];
                }
              } else {
                std::memset(stor->template ZPtr<int8_t>() + i * stor->mNPad, 0, stor->mNPad * sizeof(zero_points[0]));
              }
            }
          }
        }
      });
    }
  }

  virtual void packQWeight(const int N, const int K, const int8_t* B, const int ldb, const float* scales,
                           const int8_t* zero_points, void* ptr, parallel::IThreading* threading) {
    setQuantCorrection(N, K, zero_points, scales, ptr, threading);
    auto stor = reinterpret_cast<StorageWeight*>(ptr);
    reorderWeight(N, K, B, ldb, stor->WPtr(), threading);
    if (stor->mHasReduce) {
      auto deq = utils::amalloc<float>((size_t)K * N);
      unpackWeight(N, K, stor, deq, N, threading);
      if (stor->mRedT == JBLAS_DTYPE::F32) {
        reduceWeight(N, K, stor->mBlockSize, deq, ldb, stor->template RPtr<float>(), stor->mCStep, threading);
      }
      utils::afree(deq);
    }
  }

  void reduceWeight(const int N, const int K, const int KBlock, const float* B, const int ldb, float* rptr,
                    const int ldr, parallel::IThreading* threading) {
    parallel::Scheduler2D _para({threading->num_threads(), K, N, KBlock, 16});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp({tidx});
      _para.getIndex(thdp);
      if (thdp.valid) {
        const auto src = B + thdp.loc[0] * ldb + thdp.loc[1];
        const auto dst = rptr + thdp.loc[1] + thdp.loc[0] / KBlock * ldr;
        using RowReduceSum = kernel::wrapper::RowReduceSum<float>;
        for (int i = 0; i < thdp.size[0]; i += KBlock) {
          int rowremain = utils::remainsize(thdp.loc[0] + i, K, KBlock);
          auto ret = RowReduceSum::template forward<ISA_T>(  //
              src + i * ldb, ldb, rowremain, thdp.size[1], dst + i / KBlock * ldr);
          assert(ret == JblasSuccess);
          (void)ret;
        }
      }
    });
  }

  void quantizeWeight(const int N, const int K, const float* B, const int ldb, int blocksize, int8_t* qB, float* scales,
                      int8_t* zero_points, JBLAS_DTYPE quant_dtype, parallel::IThreading* threading) {
    int bsize = blocksize == -1 ? K : blocksize;
    parallel::Scheduler2D _para({threading->num_threads(), K, N, bsize, 16});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp({tidx});
      _para.getIndex(thdp);
      if (thdp.valid) {
        quantRowBlock(B + thdp.loc[0] * ldb + thdp.loc[1], qB + thdp.loc[0] * N + thdp.loc[1], thdp.size[0],
                      thdp.size[1], ldb, N, scales + thdp.loc[0] / bsize * N + thdp.loc[1],
                      zero_points == nullptr ? zero_points : zero_points + thdp.loc[0] / bsize * N + thdp.loc[1], bsize,
                      quant_dtype);
      }
    });
  }

  void reorderWeight(const int N, const int K, const int8_t* B, const int ldb, int8_t* dstptr,
                     parallel::IThreading* threading) {
    int KPad = utils::padto(K, _GemmCore_T::KTILE);
    parallel::Scheduler2D _para({threading->num_threads(), K, N, _GemmCore_T::KTILE, _GemmCore_T::NTILE});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp({tidx});
      _para.getIndex(thdp);
      if (thdp.valid) {
        auto rowpadded = utils::padto(thdp.size[0], _GemmCore_T::KTILE);
        auto colpadded = utils::padto(thdp.size[1], _GemmCore_T::NTILE);
        const auto src = B + thdp.loc[0] * ldb + thdp.loc[1];
        const auto dst = dstptr + thdp.loc[0] * _GemmCore_T::NTILE + thdp.loc[1] * KPad;
        using PaddingInterleaveMNWType =
            kernel::wrapper::PaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW>;
        auto ret = PaddingInterleaveMNWType::template forward<ISA_T>(  //
            src, dst, thdp.size[0], thdp.size[1], rowpadded, colpadded, ldb, KPad);
        assert(ret == JblasSuccess);
        (void)ret;
      }
    });
  }

 public:
  virtual inline JBLAS_CODE getWeight(float** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                      const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = (StorageWeight*)(_param.packedW);
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->WPtr() + n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    auto zptr = wptr->template ZPtr<int8_t>();
    int constexpr ColSize = _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW;

    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      if (wptr->mScaT == JBLAS_DTYPE::F32) {
        auto sptr = wptr->template SPtr<float>() + n_offset + i;
        kernel::wrapper::DecompressKBlockS8F32<_GemmCore_T::PACK_ROW>::template forward<ISA_T, float>(
            bptr + i * KPad, *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
            zptr != nullptr ? zptr + n_offset + i : nullptr, k_offset / _GemmCore_T::PACK_ROW,
            wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad);
      } else if (wptr->mScaT == JBLAS_DTYPE::BF16) {
        auto sptr = wptr->template SPtr<utils::bf16>() + n_offset + i;
        kernel::wrapper::DecompressKBlockS8F32<_GemmCore_T::PACK_ROW>::template forward<ISA_T, utils::bf16>(
            bptr + i * KPad, *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
            zptr != nullptr ? zptr + n_offset + i : nullptr, k_offset / _GemmCore_T::PACK_ROW,
            wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad);
      }
    }
    *dststep = k_size;
    return JblasSuccess;
  }
  virtual inline JBLAS_CODE getWeight(utils::bf16** dstptr, int* dststep, int k_size, int n_size, int k_offset,
                                      int n_offset, const Param& _param, void* tmpcache, size_t cachesize) {
    return JblasNotSupport;
  }
  virtual inline JBLAS_CODE getWeight(utils::fp16** dstptr, int* dststep, int k_size, int n_size, int k_offset,
                                      int n_offset, const Param& _param, void* tmpcache, size_t cachesize) {
    return JblasNotSupport;
  }
  virtual inline JBLAS_CODE getWeight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                      const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = (StorageWeight*)(_param.packedW);
    auto KPad = wptr->mKPad;
    auto bptr = wptr->WPtr() + n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    kernel::wrapper::Memcpy2D::template forward<ISA_T, int8_t, int8_t>(
        bptr, *dstptr, n_size / _GemmCore_T::NTILE, _GemmCore_T::NTILE * k_size, _GemmCore_T::NTILE * KPad,
        _GemmCore_T::NTILE * k_size);
    *dststep = k_size;
    return JblasSuccess;
  }

  virtual inline JBLAS_CODE getKBlockWeight(float** dstptr, int* dststep, int k_size, int n_size, int k_offset,
                                            int n_offset, const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = (StorageWeight*)(_param.packedW);
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->WPtr() + n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    int constexpr ColSize = _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      kernel::wrapper::DecompressKBlockS8S8Fp::template forward<ISA_T>(
          bptr + i * KPad, *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize);
    }
    *dststep = k_size;
    return JblasSuccess;
  }

  virtual inline JBLAS_CODE getKBlockWeight(utils::bf16** dstptr, int* dststep, int k_size, int n_size, int k_offset,
                                            int n_offset, const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = (StorageWeight*)(_param.packedW);
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->WPtr() + n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    int constexpr ColSize = _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      kernel::wrapper::DecompressKBlockS8S8Fp::template forward<ISA_T>(
          bptr + i * KPad, *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize);
    }
    *dststep = k_size;
    return JblasSuccess;
  }

  virtual inline JBLAS_CODE getKBlockWeight(utils::fp16** dstptr, int* dststep, int k_size, int n_size, int k_offset,
                                            int n_offset, const Param& _param, void* tmpcache, size_t cachesize) {
    return JblasNotSupport;
  }

  virtual inline JBLAS_CODE getKBlockWeight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset,
                                            int n_offset, const Param& _param, void* tmpcache, size_t cachesize) {
    return getWeight(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
  }

 protected:
  virtual void quantRowBlock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                             float* scales, int8_t* zero_points, int blocksize, JBLAS_DTYPE quant_dtype) {
    if (quant_dtype == JBLAS_DTYPE::S8) {
      kernel::wrapper::QuantizeSignIntRowBlock::forward<ISA_T, JBLAS_DTYPE::S8>(srcptr, dstptr, row, col, ld_src,
                                                                                ld_dst, scales, zero_points, blocksize);
    } else {
      assert(0);
    }
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class WeightKBlockS4 : public WeightKBlockS8<_GemmCore_T, ISA_T> {
 public:
  using Param = typename WeightKBlockS8<_GemmCore_T, ISA_T>::Param;
  using StorageWeight = storage::gemm::StorageWeightKBlockS4;
  StorageWeight createStorage(const int N, const int K, int blocksize, JBLAS_DTYPE weiT, JBLAS_DTYPE scaT,
                              JBLAS_DTYPE redT, bool is_asym = false) {
    int KPad = utils::padto(K, _GemmCore_T::KTILE);
    int NPad = utils::padto(N, _GemmCore_T::NTILE);
    StorageWeight tmp(_GemmCore_T::TYPE);
    tmp.resize(NPad, KPad, blocksize <= 0 ? KPad : blocksize, N, K, weiT, scaT, redT, is_asym);
    return tmp;
  }

  virtual void packQWeight(const int N, const int K, const int8_t* B, const int ldb, const float* scales,
                           const int8_t* zero_points, void* ptr, parallel::IThreading* threading) override {
    WeightKBlockS8<_GemmCore_T, ISA_T>::setQuantCorrection(N, K, zero_points, scales, ptr, threading);
    auto stor = reinterpret_cast<StorageWeight*>(ptr);
    auto tmp = utils::amalloc<float>((size_t)stor->mKPad * stor->mNPad);
    auto reorded = (int8_t*)tmp;
    WeightKBlockS8<_GemmCore_T, ISA_T>::reorderWeight(N, K, B, ldb, reorded, threading);
    compressWeight(stor->mNPad, stor->mKPad, reorded, stor->mNPad, stor->WPtr(), threading);
    if (stor->mHasReduce) {
      auto deq = tmp;
      WeightKBlockS8<_GemmCore_T, ISA_T>::unpackWeight(N, K, stor, deq, N, threading);
      if (stor->mRedT == JBLAS_DTYPE::F32) {
        WeightKBlockS8<_GemmCore_T, ISA_T>::reduceWeight(N, K, stor->mBlockSize, deq, N, stor->RPtr<float>(),
                                                         stor->mNPad, threading);
      }
    }
    utils::afree(tmp);
  }

  virtual void packNbitsWeight(const int N, const int K, bool isasym, const uint8_t* B, const int ldb,
                               const float* scales, const uint8_t* zero_points, void* ptr,
                               parallel::IThreading* threading) {
    auto stor = reinterpret_cast<StorageWeight*>(ptr);
    auto tmp = utils::amalloc<float>((size_t)stor->mKPad * stor->mNPad);
    auto blks = utils::updiv(K, stor->mBlockSize);
    auto tmpscales = (float*)tmp;
    auto tmpzeropoints = (int8_t*)(tmpscales + N * blks);
    if (scales) {
      for (size_t i = 0; i < N * blks; i += 2) {
        tmpscales[i] = scales[i] / 16;
        tmpscales[i + 1] = scales[i + 1] / 16;
      }
    }
    if (zero_points) {
      for (size_t i = 0; i < N * blks; i += 2) {
        auto tmpzp = *(zero_points + i / 2);
        tmpzeropoints[i] = ((tmpzp & 0xf) - 8) << 4;
        tmpzeropoints[i + 1] = (((tmpzp & 0xf0) >> 4) - 8) << 4;
      }
    }

    WeightKBlockS8<_GemmCore_T, ISA_T>::setTransposeQuantCorrection(N, K, zero_points ? tmpzeropoints : nullptr,
                                                                    scales ? tmpscales : nullptr, ptr, threading);
    if (B) {
      auto s8ptr = (int8_t*)tmp;
      auto transposeunpackfunc_u4s4 = [&]() {
        parallel::Scheduler2D para({threading->num_threads(), N, K, 1, 2});
        threading->parallel_for([&](int tid) {
          parallel::ThreadProblem2D thdp{tid};
          para.getIndex(thdp);
          if (thdp.valid) {
            for (size_t i = thdp.loc[0]; i < thdp.loc[0] + thdp.size[0]; i++) {
              for (size_t j = thdp.loc[1]; j < thdp.loc[1] + thdp.size[1]; j += 2) {
                auto src = *(B + i * ldb / 2 + j / 2);
                s8ptr[(j + 0) * N + i] = ((src & 0xf) - 8) << 4;
                s8ptr[(j + 1) * N + i] = (((src & 0xf0) >> 4) - 8) << 4;
              }
            }
          }
        });
      };
      transposeunpackfunc_u4s4();
      auto reorded = s8ptr + (size_t)K * N;
      WeightKBlockS8<_GemmCore_T, ISA_T>::reorderWeight(N, K, s8ptr, N, reorded, threading);
      compressWeight(stor->mNPad, stor->mKPad, reorded, stor->mNPad, stor->WPtr(), threading);
      if (stor->mHasReduce) {
        auto deq = (float*)tmp;
        WeightKBlockS8<_GemmCore_T, ISA_T>::unpackWeight(N, K, stor, deq, N, threading);
        if (stor->mRedT == JBLAS_DTYPE::F32) {
          WeightKBlockS8<_GemmCore_T, ISA_T>::reduceWeight(N, K, stor->mBlockSize, deq, N, stor->template RPtr<float>(),
                                                           stor->mCStep, threading);
        }
      }
    }
    utils::afree(tmp);
  }

  void compressWeight(const int N, const int K, const int8_t* B, const int ldb, utils::bit4x2* dstptr,
                      parallel::IThreading* threading) {
    parallel::Scheduler2D _para({threading->num_threads(), K, N, _GemmCore_T::KTILE, _GemmCore_T::NTILE});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp({tidx});
      _para.getIndex(thdp);
      if (thdp.valid) {
        auto ret = doCompress(B + thdp.loc[0] * ldb + thdp.loc[1], dstptr + thdp.loc[0] * ldb / 2 + thdp.loc[1] / 2,
                              thdp.size[0], thdp.size[1], ldb, ldb);
        assert(ret == JblasSuccess);
        (void)ret;
      }
    });
  }

 public:
  inline JBLAS_CODE getWeight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                              const Param& _param, void* tmpcache, size_t cachesize) override {
    auto wptr = (StorageWeight*)(_param.packedW);
    auto KPad = wptr->mKPad;
    auto bptr = wptr->WPtr() + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
    int constexpr ColSize = _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      if (wptr->mDType == JBLAS_DTYPE::S4_CLIP) {
        kernel::wrapper::DecompressKBlockS4S8::template forward<ISA_T, JBLAS_DTYPE::S4_CLIP>(
            (utils::int4x2*)(bptr + i * KPad / 2), *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize,
            ColSize, ColSize);
      } else if (wptr->mDType == JBLAS_DTYPE::S4_FULLRANGE) {
        kernel::wrapper::DecompressKBlockS4S8::template forward<ISA_T, JBLAS_DTYPE::S4_FULLRANGE>(
            (utils::int4x2*)(bptr + i * KPad / 2), *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize,
            ColSize, ColSize);
      }
    }
    *dststep = k_size;
    return JblasSuccess;
  }

  inline JBLAS_CODE getKBlockWeight(float** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                    const Param& _param, void* tmpcache, size_t cachesize) override {
    return getFpKBlockWeight(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
  }

  inline JBLAS_CODE getKBlockWeight(utils::bf16** dstptr, int* dststep, int k_size, int n_size, int k_offset,
                                    int n_offset, const Param& _param, void* tmpcache, size_t cachesize) override {
    return getFpKBlockWeight(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
  }

  inline JBLAS_CODE getKBlockWeight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                    const Param& _param, void* tmpcache, size_t cachesize) override {
    return getWeight(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
  }

  inline JBLAS_CODE getWeight(float** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                              const Param& _param, void* tmpcache, size_t cachesize) override {
    return getFpWeight(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
  }

  inline JBLAS_CODE getWeight(utils::bf16** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                              const Param& _param, void* tmpcache, size_t cachesize) override {
    return getFpWeight(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
  }

 protected:
  virtual JBLAS_CODE doCompress(const int8_t* srcptr, void* dstptr, int row, int col, int ld_src, int ld_dst) {
    return kernel::wrapper::CompressS8S4<_GemmCore_T::NTILE>::template forward<ISA_T>(
        srcptr, reinterpret_cast<utils::int4x2*>(dstptr), row, col, ld_src,
        ld_dst);  // ld_dst here not stride
  }

  virtual void quantRowBlock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                             float* scales, int8_t* zero_points, int blocksize, JBLAS_DTYPE quant_dtype) {
    if (quant_dtype == JBLAS_DTYPE::S4_FULLRANGE) {
      kernel::wrapper::QuantizeSignIntRowBlock::forward<ISA_T, JBLAS_DTYPE::S4_FULLRANGE>(
          srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points, blocksize);
    } else if (quant_dtype == JBLAS_DTYPE::S4_CLIP) {
      kernel::wrapper::QuantizeSignIntRowBlock::forward<ISA_T, JBLAS_DTYPE::S4_CLIP>(
          srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points, blocksize);
    }
  }

  template <typename T>
  inline JBLAS_CODE getFpKBlockWeight(T** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                      const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = (StorageWeight*)(_param.packedW);
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->WPtr() + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
    int constexpr ColSize = _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      if (wptr->mScaT == JBLAS_DTYPE::F32) {
        auto sptr = wptr->template SPtr<float>() + n_offset + i;
        if (wptr->mDType == JBLAS_DTYPE::S4_CLIP) {
          kernel::wrapper::DecompressKBlockS4S8Fp<T>::template forward<ISA_T, JBLAS_DTYPE::S4_CLIP>(
              (utils::int4x2*)(bptr + i * KPad / 2), *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize,
              ColSize, ColSize, tmpcache, cachesize);
        } else if (wptr->mDType == JBLAS_DTYPE::S4_FULLRANGE) {
          kernel::wrapper::DecompressKBlockS4S8Fp<T>::template forward<ISA_T, JBLAS_DTYPE::S4_FULLRANGE>(
              (utils::int4x2*)(bptr + i * KPad / 2), *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize,
              ColSize, ColSize, tmpcache, cachesize);
        }
      } else if (wptr->mScaT == JBLAS_DTYPE::BF16) {
        auto sptr = wptr->template SPtr<utils::bf16>() + n_offset + i;
        if (wptr->mDType == JBLAS_DTYPE::S4_CLIP) {
          kernel::wrapper::DecompressKBlockS4S8Fp<T>::template forward<ISA_T, JBLAS_DTYPE::S4_CLIP>(
              (utils::int4x2*)(bptr + i * KPad / 2), *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize,
              ColSize, ColSize, tmpcache, cachesize);
        } else if (wptr->mDType == JBLAS_DTYPE::S4_FULLRANGE) {
          kernel::wrapper::DecompressKBlockS4S8Fp<T>::template forward<ISA_T, JBLAS_DTYPE::S4_FULLRANGE>(
              (utils::int4x2*)(bptr + i * KPad / 2), *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize,
              ColSize, ColSize, tmpcache, cachesize);
        }
      }
    }
    *dststep = k_size;
    return JblasSuccess;
  }

  template <typename _T>
  inline JBLAS_CODE getFpWeight(_T** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = (StorageWeight*)(_param.packedW);
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->WPtr() + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
    int constexpr ColSize = _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      auto zptr = wptr->template ZPtr<int8_t>();
      if (wptr->mScaT == JBLAS_DTYPE::F32) {
        auto sptr = wptr->template SPtr<float>() + n_offset + i;
        if (wptr->mDType == JBLAS_DTYPE::S4_CLIP) {
          kernel::wrapper::DecompressKBlockS4Fp<_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, float,
                                                                                             JBLAS_DTYPE::S4_CLIP>(
              (utils::int4x2*)(bptr + i * KPad / 2), *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize,
              ColSize, ColSize, sptr, zptr != nullptr ? zptr + n_offset + i : nullptr, k_offset / _GemmCore_T::PACK_ROW,
              wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else if (wptr->mDType == JBLAS_DTYPE::S4_FULLRANGE) {
          kernel::wrapper::DecompressKBlockS4Fp<_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, float,
                                                                                             JBLAS_DTYPE::S4_FULLRANGE>(
              (utils::int4x2*)(bptr + i * KPad / 2), *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize,
              ColSize, ColSize, sptr, zptr != nullptr ? zptr + n_offset + i : nullptr, k_offset / _GemmCore_T::PACK_ROW,
              wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        }
      } else if (wptr->mScaT == JBLAS_DTYPE::BF16) {
        auto sptr = wptr->template SPtr<utils::bf16>() + n_offset + i;
        if (wptr->mDType == JBLAS_DTYPE::S4_CLIP) {
          kernel::wrapper::DecompressKBlockS4Fp<_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, utils::bf16,
                                                                                             JBLAS_DTYPE::S4_CLIP>(
              (utils::int4x2*)(bptr + i * KPad / 2), *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize,
              ColSize, ColSize, sptr, zptr != nullptr ? zptr + n_offset + i : nullptr, k_offset / _GemmCore_T::PACK_ROW,
              wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else if (wptr->mDType == JBLAS_DTYPE::S4_FULLRANGE) {
          kernel::wrapper::DecompressKBlockS4Fp<_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, utils::bf16,
                                                                                             JBLAS_DTYPE::S4_FULLRANGE>(
              (utils::int4x2*)(bptr + i * KPad / 2), *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize,
              ColSize, ColSize, sptr, zptr != nullptr ? zptr + n_offset + i : nullptr, k_offset / _GemmCore_T::PACK_ROW,
              wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        }
      }
    }
    *dststep = k_size;
    return JblasSuccess;
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class WeightKBlockF4 : public WeightKBlockS4<_GemmCore_T, ISA_T> {
 public:
  using Param = typename WeightKBlockS8<_GemmCore_T, ISA_T>::Param;
  using StorageWeight = storage::gemm::StorageWeightKBlockF4;
  StorageWeight createStorage(const int N, const int K, int blocksize, JBLAS_DTYPE f4T, JBLAS_DTYPE scaT) {
    int KPad = utils::padto(K, _GemmCore_T::KTILE);
    int NPad = utils::padto(N, _GemmCore_T::NTILE);
    StorageWeight tmp(_GemmCore_T::TYPE);
    tmp.resize(NPad, KPad, blocksize <= 0 ? KPad : blocksize, N, K, f4T, scaT);
    return tmp;
  }

  virtual void packQWeight(const int N, const int K, const int8_t* B, const int ldb, const float* scales, void* ptr,
                           parallel::IThreading* threading) {
    WeightKBlockS8<_GemmCore_T, ISA_T>::setQuantCorrection(N, K, NULL, scales, ptr, threading);
    auto stor = reinterpret_cast<StorageWeight*>(ptr);
    auto reorded = utils::amalloc<int8_t>((size_t)stor->mKPad * stor->mNPad);
    WeightKBlockS8<_GemmCore_T, ISA_T>::reorderWeight(N, K, B, ldb, reorded, threading);
    WeightKBlockS4<_GemmCore_T, ISA_T>::compressWeight(stor->mNPad, stor->mKPad, reorded, stor->mNPad, stor->WPtr(),
                                                       threading);
    utils::afree(reorded);
  }

  inline JBLAS_CODE getWeight(float** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                              const Param& _param, void* tmpcache, size_t cachesize) override {
    return getFpWeight(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
  }

  inline JBLAS_CODE getWeight(utils::bf16** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                              const Param& _param, void* tmpcache, size_t cachesize) override {
    return getFpWeight(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
  }

  inline JBLAS_CODE getKBlockWeight(float** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                    const Param& _param, void* tmpcache, size_t cachesize) override {
    return getFpKBlockWeight(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
  }

  inline JBLAS_CODE getKBlockWeight(utils::bf16** dstptr, int* dststep, int k_size, int n_size, int k_offset,
                                    int n_offset, const Param& _param, void* tmpcache, size_t cachesize) override {
    return getFpKBlockWeight(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
  }

 protected:
  virtual void quantRowBlock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                             float* scales, int8_t* zero_points, int blocksize, JBLAS_DTYPE quant_dtype) override {
    if (quant_dtype == JBLAS_DTYPE::F4_BNB) {
      kernel::wrapper::QuantizeF4RowBlock::forward<ISA_T, JBLAS_DTYPE::F4_BNB>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                               scales, zero_points, blocksize);
    } else if (quant_dtype == JBLAS_DTYPE::F4_E2M1) {
      kernel::wrapper::QuantizeF4RowBlock::forward<ISA_T, JBLAS_DTYPE::F4_E2M1>(srcptr, dstptr, row, col, ld_src,
                                                                                ld_dst, scales, zero_points, blocksize);
    } else if (quant_dtype == JBLAS_DTYPE::F4_NF4) {
      kernel::wrapper::QuantizeF4RowBlock::forward<ISA_T, JBLAS_DTYPE::F4_NF4>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                               scales, zero_points, blocksize);
    }
  }

  virtual JBLAS_CODE doCompress(const int8_t* srcptr, void* dstptr, int row, int col, int ld_src, int ld_dst) override {
    return kernel::wrapper::CompressFp4<_GemmCore_T::NTILE>::template forward<ISA_T>(
        srcptr, reinterpret_cast<utils::f4x2*>(dstptr), row, col, ld_src,
        ld_dst);  // ld_dst here not stride
  }

  template <typename T>
  inline JBLAS_CODE getFpWeight(T** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = (StorageWeight*)(_param.packedW);
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->WPtr() + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
    int constexpr ColSize = _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      auto f4ptr = reinterpret_cast<utils::f4x2*>(bptr + i * KPad / 2);
      auto fp32ptr = *dstptr + i * k_size;
      if (wptr->mScaT == JBLAS_DTYPE::F32) {
        auto sptr = wptr->SPtr<float>() + n_offset + i;
        if (wptr->mDType == JBLAS_DTYPE::F4_NF4) {
          kernel::wrapper::DecompressKBlockF4Fp<T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, float,
                                                                                            JBLAS_DTYPE::F4_NF4>(
              f4ptr, fp32ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              k_offset / _GemmCore_T::PACK_ROW, wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else if (wptr->mDType == JBLAS_DTYPE::F4_E2M1) {
          kernel::wrapper::DecompressKBlockF4Fp<T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, float,
                                                                                            JBLAS_DTYPE::F4_E2M1>(
              f4ptr, fp32ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              k_offset / _GemmCore_T::PACK_ROW, wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else if (wptr->mDType == JBLAS_DTYPE::F4_BNB) {
          kernel::wrapper::DecompressKBlockF4Fp<T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, float,
                                                                                            JBLAS_DTYPE::F4_BNB>(
              f4ptr, fp32ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              k_offset / _GemmCore_T::PACK_ROW, wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        }
      } else if (wptr->mScaT == JBLAS_DTYPE::BF16) {
        auto sptr = wptr->SPtr<utils::bf16>() + n_offset + i;
        if (wptr->mDType == JBLAS_DTYPE::F4_NF4) {
          kernel::wrapper::DecompressKBlockF4Fp<T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, utils::bf16,
                                                                                            JBLAS_DTYPE::F4_NF4>(
              f4ptr, fp32ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              k_offset / _GemmCore_T::PACK_ROW, wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else if (wptr->mDType == JBLAS_DTYPE::F4_E2M1) {
          kernel::wrapper::DecompressKBlockF4Fp<T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, utils::bf16,
                                                                                            JBLAS_DTYPE::F4_E2M1>(
              f4ptr, fp32ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              k_offset / _GemmCore_T::PACK_ROW, wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else if (wptr->mDType == JBLAS_DTYPE::F4_BNB) {
          kernel::wrapper::DecompressKBlockF4Fp<T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, utils::bf16,
                                                                                            JBLAS_DTYPE::F4_BNB>(
              f4ptr, fp32ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              k_offset / _GemmCore_T::PACK_ROW, wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        }
      }
    }
    *dststep = k_size;
    return JblasSuccess;
  }

  template <typename T>
  inline JBLAS_CODE getFpKBlockWeight(T** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                      const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = (StorageWeight*)(_param.packedW);
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->WPtr() + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
    int constexpr ColSize = _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      auto f4ptr = reinterpret_cast<utils::f4x2*>(bptr + i * KPad / 2);
      auto fp32ptr = *dstptr + i * k_size;
      if (wptr->mDType == JBLAS_DTYPE::F4_NF4) {
        kernel::wrapper::DecompressKBlockF4FpNoscale<T>::template forward<ISA_T, JBLAS_DTYPE::F4_NF4>(
            f4ptr, fp32ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, tmpcache, cachesize);
      } else if (wptr->mDType == JBLAS_DTYPE::F4_E2M1) {
        kernel::wrapper::DecompressKBlockF4FpNoscale<T>::template forward<ISA_T, JBLAS_DTYPE::F4_E2M1>(
            f4ptr, fp32ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, tmpcache, cachesize);
      } else if (wptr->mDType == JBLAS_DTYPE::F4_BNB) {
        kernel::wrapper::DecompressKBlockF4FpNoscale<T>::template forward<ISA_T, JBLAS_DTYPE::F4_BNB>(
            f4ptr, fp32ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, tmpcache, cachesize);
      }
    }
    *dststep = k_size;
    return JblasSuccess;
  }
};
}  // namespace gemm
}  // namespace prologue_b
}  // namespace jblas
