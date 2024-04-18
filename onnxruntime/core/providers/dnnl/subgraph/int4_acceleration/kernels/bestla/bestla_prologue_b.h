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
#include <cassert>
#include "bestla_utils.h"
#include "bestla_storage.h"
#include "bestla_device.h"
#include "bestla_parallel.h"
#include "kernel_wrapper.h"

namespace bestla {
namespace prologue_b {
namespace gemm {

template <typename WT, BTLA_ISA ISA_T>
static inline void transposeWeight(const int Row, const int Col, const WT* src, const int ld_src, WT* dst,
                                   const int ld_dst, parallel::IThreading* threading) {
  bestla::parallel::Scheduler2D _para;
  _para.update({threading->num_threads(), Row, Col, 16, 16});
  threading->parallel_for([&](int tidx) {
    bestla::parallel::ThreadProblem2D thdp{tidx};
    _para.getIndex(thdp);
    if (thdp.valid) {
      kernel::wrapper::Transpose2D<WT>::template forward<ISA_T>(src + thdp.loc[0] * ld_src + thdp.loc[1],
                                                                dst + thdp.loc[0] + thdp.loc[1] * ld_dst, thdp.size[0],
                                                                thdp.size[1], ld_src, ld_dst);
    }
  });
}
template <typename WType>
struct ParamWeightPack {
  const WType* B;
  const int ldb;
  storage::gemm::StoragePackedWeight* packedW;
};

template <class _GemmCore_T, BTLA_ISA ISA_T>
class WeightPack {
 public:
  using WType = typename _GemmCore_T::BType;
  using StorageType = storage::gemm::StoragePackedWeight;
  using Param = ParamWeightPack<WType>;

  StorageType createStorage(int n, int k) {
    int KPad = utils::padto(k, _GemmCore_T::KTILE);
    int NPad = utils::padto(n, _GemmCore_T::NTILE);
    StorageType tmp(_GemmCore_T::ID);
    tmp.resize(NPad, KPad, n, k, utils::bestla_dtype<WType>);
    return tmp;
  }

  void packWeightTranspose(const int N, const int K, const Param& _param, parallel::IThreading* threading) {
    auto B_NT = utils::amalloc<WType>(static_cast<size_t>(N) * K);
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
    const auto dst = packedw->template WPtr<WType>() + thdp.loc[0] * _GemmCore_T::NTILE + thdp.loc[1] * packedw->mKPad;
    using PaddingInterleaveMNWType = kernel::wrapper::PaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW>;
    auto ret = PaddingInterleaveMNWType::template forward<ISA_T>(  //
        src, dst, thdp.size[0], thdp.size[1], rowpadded, colpadded, _param.ldb, packedw->mKPad);
    assert(ret == BTLA_CODE::Success);
    (void)ret;
  }

  inline BTLA_CODE getWeight(WType** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                             const Param param, void* tmpcache, size_t cachesize) {
    auto wptr = param.packedW;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->template WPtr<WType>() + n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    kernel::wrapper::Memcpy2D::template forward<BTLA_ISA::NoSIMD, WType, WType>(
        bptr, *dstptr, n_size / _GemmCore_T::NTILE, _GemmCore_T::NTILE * k_size, _GemmCore_T::NTILE * KPad,
        _GemmCore_T::NTILE * k_size);
    *dststep = k_size;
    return BTLA_CODE::Success;
  }
};

struct ParamWeightKBlockNInteger {
  storage::gemm::StorageWeightKBlockNInteger* packedW;
};

template <class _GemmCore_T, BTLA_ISA ISA_T>
class WeightKBlockNInteger {
 public:
  using StorageWeight = storage::gemm::StorageWeightKBlockNInteger;
  using BType = typename _GemmCore_T::BType;
  using Param = ParamWeightKBlockNInteger;

  static StorageWeight createStorage(int n, int k, int blocksize, BTLA_DTYPE qtype, BTLA_DTYPE scat, BTLA_DTYPE redt,
                                     bool is_asym) {
    int KPad = utils::padto(k, _GemmCore_T::KTILE);
    int NPad = utils::padto(n, _GemmCore_T::NTILE);
    StorageWeight tmp(_GemmCore_T::ID);
    tmp.resize(NPad, KPad, blocksize <= 0 ? KPad : blocksize, n, k, qtype, scat, redt, is_asym);
    return tmp;
  }

  static void enableShuffle(StorageWeight* stor) { stor->enable_shuffle(); }

  void packTransposeWeight(const int N, const int K, const float* B, const int ldb, StorageWeight* stor,
                           parallel::IThreading* threading) {
    auto B_NT = utils::amalloc<float>(static_cast<size_t>(N) * K);
    transposeWeight<float, ISA_T>(N, K, B, ldb, B_NT, N, threading);
    packWeight(N, K, B_NT, N, stor, threading);
    utils::afree(B_NT);
  }

  // from packed N//NtilexKPadxNTile int8 weight to KxN f32 weight
  void unpackTransposeWeight(const int N, const int K, StorageWeight* stor, float* B, const int ldb,
                             parallel::IThreading* threading) {
    auto B_NT = utils::amalloc<float>(static_cast<size_t>(N) * K);
    unpackWeight(N, K, stor, B_NT, N, threading);
    transposeWeight<float, ISA_T>(K, N, B_NT, N, B, ldb, threading);
    utils::afree(B_NT);
  }

  // from KxN f32 weight to packed N//NtilexKPadxNTile int8 weight
  void packWeight(const int N, const int K, const float* B, const int ldb, StorageWeight* ptr,
                  parallel::IThreading* threading) {
    auto tmpq = utils::amalloc<int8_t>(static_cast<size_t>(N) * K);
    int nk_scale = utils::updiv(K, ptr->mBlockSize);
    auto ssize = static_cast<size_t>(N) * nk_scale;
    auto Tscales = utils::amalloc<float>(ssize);
    auto Tzps = utils::amalloc<int8_t>(ptr->IsAsym() ? ssize : 0);
    quantizeWeight(N, K, B, ldb, tmpq, Tscales, Tzps, ptr, threading);
    packQWeight(N, K, tmpq, N, Tscales, Tzps, ptr, threading);
    utils::afree(tmpq);
    utils::afree(Tscales);
    utils::afree(Tzps);
  }

  void unpackWeight(const int N, const int K, StorageWeight* stor, float* B, const int ldb,
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
        getWeight(&dstptr, &dststep, rowpad, colpad, thdp.loc[0], thdp.loc[1], {stor}, tmpcache, CacheSize);
        kernel::wrapper::RevertPaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW>::template forward<ISA_T>(
            dstptr, B + thdp.loc[0] * ldb + thdp.loc[1], thdp.size[0], thdp.size[1], rowpad, colpad, dststep, ldb);
        utils::afree(dequant);
      }
    });
  }

  static void unpackWeight(const int N, const int K, StorageWeight* stor, int8_t* B, const int ldb,
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
        getWeight(&dstptr, &dststep, rowpad, colpad, thdp.loc[0], thdp.loc[1], {stor}, tmpcache, CacheSize);
        kernel::wrapper::RevertPaddingInterleaveMN<_GemmCore_T::NTILE, _GemmCore_T::PACK_ROW>::template forward<ISA_T>(
            dstptr, B + thdp.loc[0] * ldb + thdp.loc[1], thdp.size[0], thdp.size[1], rowpad, colpad, dststep, ldb);
        utils::afree(dequant);
      }
    });
  }

  static void setQuantCorrection(const int N, const int K, const int8_t* zero_points, const float* scales,
                                 StorageWeight* stor, parallel::IThreading* threading) {
    int rawnk_scale = utils::updiv(K, stor->mBlockSize);
    int nk_scale = utils::updiv(stor->mKPad, stor->mBlockSize);
    parallel::Scheduler2D _para({threading->num_threads(), 1, nk_scale, 1, 1});
    if (stor->SDtype() == BTLA_DTYPE::F32) {  // fp32 to fp32 direct copy
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
    } else if (stor->SDtype() == BTLA_DTYPE::BF16) {
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
    } else if (stor->SDtype() == BTLA_DTYPE::F8_E8M0) {
      threading->parallel_for([&](int tidx) {
        parallel::ThreadProblem2D thdp{tidx};
        _para.getIndex(thdp);
        if (thdp.valid) {
          for (int i = thdp.loc[1]; i < thdp.loc[1] + thdp.size[1]; i++) {
            if (i < rawnk_scale) {
              if (scales != nullptr) {
                for (size_t j = 0; j < N; j++) {
                  stor->template SPtr<utils::f8>()[j + i * stor->mNPad] = static_cast<int8_t>(scales[i * N + j]);
                }
              }
            } else {
              if (scales != nullptr)
                std::memset(stor->template SPtr<utils::f8>() + i * stor->mNPad, 0, stor->mNPad * sizeof(utils::f8));
            }
          }
        }
      });
    } else {
      assert(0);
    }
  }

  static void setShuffleIndices(const int* groupindices, StorageWeight* stor, parallel::IThreading* threading) {
    auto groupsize = utils::updiv(stor->mK, stor->mBlockSize);
    parallel::Scheduler2D _para({threading->num_threads(), 1, groupsize, 1, 1});
    auto countptr = utils::amalloc<int>(groupsize);
    std::memset(countptr, 0, groupsize * sizeof(int));
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp{tidx};
      _para.getIndex(thdp);
      if (thdp.valid) {
        auto siptr = stor->ShfIndice();
        for (size_t i = 0; i < stor->mK; i++) {
          if (groupindices[i] >= thdp.loc[1] && groupindices[i] < thdp.loc[1] + thdp.size[1]) {
            siptr[groupindices[i] * stor->mBlockSize + countptr[groupindices[i]]] = i;
            countptr[groupindices[i]]++;
          }
        }
      }
    });
    utils::afree(countptr);
  }

  static void setTransposeQuantCorrection(const int N, const int K, const int8_t* zero_points, const float* scales,
                                          StorageWeight* stor, parallel::IThreading* threading) {
    int rawnk_scale = utils::updiv(K, stor->mBlockSize);
    int nk_scale = utils::updiv(stor->mKPad, stor->mBlockSize);
    parallel::Scheduler2D _para({threading->num_threads(), 1, nk_scale, 1, 1});
    if (stor->SDtype() == BTLA_DTYPE::F32) {  // fp32 to fp32 direct copy
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
        }
      });
    } else if (stor->SDtype() == BTLA_DTYPE::BF16) {
      threading->parallel_for([&](int tidx) {
        parallel::ThreadProblem2D thdp{tidx};
        _para.getIndex(thdp);
        if (thdp.valid) {
          if (scales) {
            for (int i = thdp.loc[1]; i < thdp.loc[1] + thdp.size[1]; i++) {
              if (i < rawnk_scale) {
                for (size_t j = 0; j < N; j++) {
                  stor->template SPtr<utils::bf16>()[i * stor->mNPad + j] = utils::bf16(scales[j * rawnk_scale + i]);
                }
              } else {
                std::memset(stor->template SPtr<utils::bf16>() + i * stor->mNPad, 0, stor->mNPad * sizeof(utils::bf16));
              }
            }
          }
        }
      });
    } else if (stor->SDtype() == BTLA_DTYPE::F8_E8M0) {
      threading->parallel_for([&](int tidx) {
        parallel::ThreadProblem2D thdp{tidx};
        _para.getIndex(thdp);
        if (thdp.valid) {
          if (scales) {
            for (int i = thdp.loc[1]; i < thdp.loc[1] + thdp.size[1]; i++) {
              if (i < rawnk_scale) {
                for (size_t j = 0; j < N; j++) {
                  stor->template SPtr<utils::f8>()[i * stor->mNPad + j] = scales[j * rawnk_scale + i];
                }
              } else {
                std::memset(stor->template SPtr<utils::f8>() + i * stor->mNPad, 0, stor->mNPad * sizeof(utils::f8));
              }
            }
          }
        }
      });
    } else {
      assert(0);
    }
    if (stor->IsAsym() && zero_points)
      threading->parallel_for([&](int tidx) {
        parallel::ThreadProblem2D thdp{tidx};
        _para.getIndex(thdp);
        if (thdp.valid) {
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
      });
  }

  void packQWeight(const int N, const int K, const int8_t* B, const int ldb, const float* scales,
                   const int8_t* zero_points, StorageWeight* stor, parallel::IThreading* threading) {
    setQuantCorrection(N, K, zero_points, scales, stor, threading);
    if (stor->mDType == BTLA_DTYPE::S8 || stor->mDType == BTLA_DTYPE::F8_E4M3 || stor->mDType == BTLA_DTYPE::F8_E5M2) {
      reorderWeight(N, K, B, ldb, stor->WPtr<int8_t>(), threading);
    } else {
      auto reorded = utils::amalloc<int8_t>((size_t)stor->mKPad * stor->mNPad);
      reorderWeight(N, K, B, ldb, reorded, threading);
      compressWeight(stor->mNPad, stor->mKPad, reorded, stor->mNPad, stor->WPtr<int8_t>(), stor->mDType, threading);
      utils::afree(reorded);
    }
    reduceWeight(stor, threading);
  }

  virtual void packNbitsWeightQ4(const int N, const int K, bool isasym, const uint8_t* B, const int ldb,
                                 const float* scales, const uint8_t* zero_points, void* ptr,
                                 parallel::IThreading* threading) {
    auto stor = reinterpret_cast<StorageWeight*>(ptr);
    auto tmp = utils::amalloc<float>(static_cast<size_t>(stor->mKPad) * stor->mNPad);
    auto blks = utils::updiv(K, stor->mBlockSize);
    auto blks_padding2 = utils::padto(blks, 2);
    auto tmpscales = tmp;
    auto tmpzeropoints = reinterpret_cast<int8_t*>(tmpscales + N * blks);
    if (scales) {
      for (size_t i = 0; i < N * blks; i += 2) {
        tmpscales[i] = scales[i] / 16;
        tmpscales[i + 1] = scales[i + 1] / 16;
      }
    }
    if (zero_points) {
      for (size_t i = 0; i < N; i += 1) {
        for (size_t ib = 0; ib < blks; ib += 2) {
          auto tmpzp = *(zero_points + i * blks_padding2 / 2 + ib / 2);
          tmpzeropoints[i * blks + ib] = ((tmpzp & 0xf) - 8) << 4;
          if (ib + 1 < blks) {
            tmpzeropoints[i * blks + ib + 1] = (((tmpzp & 0xf0) >> 4) - 8) << 4;
          }
        }
      }
    }

    setTransposeQuantCorrection(N, K, zero_points ? tmpzeropoints : nullptr, scales ? tmpscales : nullptr, stor,
                                threading);
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
      auto reorded = s8ptr + static_cast<size_t>(K) * N;
      reorderWeight(N, K, s8ptr, N, reorded, threading);
      compressWeight(stor->mNPad, stor->mKPad, reorded, stor->mNPad, stor->WPtr<int8_t>(), stor->mDType, threading);
    }
    utils::afree(tmp);
  }

  void reduceWeight(StorageWeight* stor, parallel::IThreading* threading) {
    if (stor->HasReduce()) {
      auto deq = utils::amalloc<float>((size_t)stor->mK * stor->mN);
      unpackWeight(stor->mN, stor->mK, stor, deq, stor->mN, threading);
      if (stor->RDtype() == BTLA_DTYPE::F32) {
        reduce(stor->mN, stor->mK, stor->mBlockSize, deq, stor->mN, stor->template RPtr<float>(), stor->CStep(),
               threading);
      } else if (stor->RDtype() == BTLA_DTYPE::BF16) {
        reduce(stor->mN, stor->mK, stor->mBlockSize, deq, stor->mN, stor->template RPtr<utils::bf16>(), stor->CStep(),
               threading);
      } else {
        assert(0);
      }
      utils::afree(deq);
    }
  }

  void quantizeWeight(const int N, const int K, const float* B, const int ldb, int8_t* qB, float* scales,
                      int8_t* zero_points, void* stor, parallel::IThreading* threading) {
    auto ptr = reinterpret_cast<StorageWeight*>(stor);
    int bsize = ptr->mBlockSize == -1 ? K : ptr->mBlockSize;
    parallel::Scheduler2D _para({threading->num_threads(), K, N, bsize, 16});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp({tidx});
      _para.getIndex(thdp);
      if (thdp.valid) {
        quantRowBlock(B + thdp.loc[0] * ldb + thdp.loc[1], qB + thdp.loc[0] * N + thdp.loc[1], thdp.size[0],
                      thdp.size[1], ldb, N, scales + thdp.loc[0] / bsize * N + thdp.loc[1],
                      zero_points == nullptr ? zero_points : zero_points + thdp.loc[0] / bsize * N + thdp.loc[1], ptr);
      }
    });
  }

  static void reorderWeight(const int N, const int K, const int8_t* B, const int ldb, int8_t* dstptr,
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
        assert(ret == BTLA_CODE::Success);
        (void)ret;
      }
    });
  }

  static void compressWeight(const int N, const int K, const int8_t* B, const int ldb, int8_t* dstptr, BTLA_DTYPE qtype,
                             parallel::IThreading* threading) {
    parallel::Scheduler2D _para({threading->num_threads(), K, N, _GemmCore_T::KTILE, _GemmCore_T::NTILE});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp({tidx});
      _para.getIndex(thdp);
      if (thdp.valid) {
        auto ret = doCompress(B + thdp.loc[0] * ldb + thdp.loc[1], dstptr + thdp.loc[0] * ldb / 2 + thdp.loc[1] / 2,
                              thdp.size[0], thdp.size[1], ldb, ldb, qtype);
        assert(ret == BTLA_CODE::Success);
        (void)ret;
      }
    });
  }

  template <typename RED_T>
  static void reduce(const int N, const int K, const int KBlock, const float* B, const int ldb, RED_T* rptr,
                     const int ldr, parallel::IThreading* threading) {
    parallel::Scheduler2D _para({threading->num_threads(), K, N, KBlock, 16});
    threading->parallel_for([&](int tidx) {
      parallel::ThreadProblem2D thdp({tidx});
      _para.getIndex(thdp);
      if (thdp.valid) {
        const auto src = B + thdp.loc[0] * ldb + thdp.loc[1];
        const auto dst = rptr + thdp.loc[1] + thdp.loc[0] / KBlock * ldr;
        using RowReduceSum = kernel::wrapper::RowReduceSum<RED_T>;
        for (int i = 0; i < thdp.size[0]; i += KBlock) {
          int rowremain = utils::remainsize(thdp.loc[0] + i, K, KBlock);
          auto ret = RowReduceSum::template forward<ISA_T>(  //
              src + i * ldb, ldb, rowremain, thdp.size[1], dst + i / KBlock * ldr);
          assert(ret == BTLA_CODE::Success);
          (void)ret;
        }
      }
    });
  }

 public:
  virtual inline BTLA_CODE getWeight(float** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                     const Param& _param, void* tmpcache, size_t cachesize) {
    return getFpWeight(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
  }

  virtual inline BTLA_CODE getWeight(utils::bf16** dstptr, int* dststep, int k_size, int n_size, int k_offset,
                                     int n_offset, const Param& _param, void* tmpcache, size_t cachesize) {
    return getFpWeight(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
  }

  static inline BTLA_CODE getWeight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                    const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.packedW;
    if (wptr->mDType == BTLA_DTYPE::S8) {
      return getQ8Weight(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
    } else if (wptr->mDType == BTLA_DTYPE::S4_CLIP || wptr->mDType == BTLA_DTYPE::S4_FULLRANGE) {
      return getQ4Weight(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
    } else {
      assert(0);
    }
    return BTLA_CODE::NotSupport;
  }

  static inline BTLA_CODE getKBlockWeight(float** dstptr, int* dststep, int k_size, int n_size, int k_offset,
                                          int n_offset, const Param& _param, void* tmpcache, size_t cachesize) {
    return getFpKBlockWeight(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
  }

  static inline BTLA_CODE getKBlockWeight(utils::bf16** dstptr, int* dststep, int k_size, int n_size, int k_offset,
                                          int n_offset, const Param& _param, void* tmpcache, size_t cachesize) {
    return getFpKBlockWeight(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
  }

  static inline BTLA_CODE getKBlockWeight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset,
                                          int n_offset, const Param& _param, void* tmpcache, size_t cachesize) {
    return getWeight(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
  }

  static inline BTLA_CODE getScale(float** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                   const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.packedW;
    if (wptr->SDtype() == BTLA_DTYPE::F32) {
      auto aptr = wptr->template SPtr<float>();
      kernel::wrapper::Memcpy2D::template forward<BTLA_ISA::NoSIMD>(
          aptr + k_offset / wptr->mBlockSize * wptr->CStep() + n_offset, *dstptr,
          utils::updiv(k_size, wptr->mBlockSize), n_size, wptr->CStep(), n_size);
      *dststep = n_size;
    }
    if (wptr->SDtype() == BTLA_DTYPE::BF16) {
      auto aptr = wptr->template SPtr<utils::bf16>();
      kernel::wrapper::Memcpy2DBf16CvtFp32::forward<ISA_T>(
          aptr + k_offset / wptr->mBlockSize * wptr->CStep() + n_offset, *dstptr,
          utils::updiv(k_size, wptr->mBlockSize), n_size, wptr->CStep() * 2, n_size * 4, false);
      *dststep = n_size;
    }
    return BTLA_CODE::Success;
  }

  static inline BTLA_CODE getReduce(float** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                    const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.packedW;
    if (wptr->RDtype() == BTLA_DTYPE::F32) {
      auto aptr = wptr->template RPtr<float>();
      kernel::wrapper::Memcpy2D::template forward<BTLA_ISA::NoSIMD>(
          aptr + k_offset / wptr->mBlockSize * wptr->CStep() + n_offset, *dstptr,
          utils::updiv(k_size, wptr->mBlockSize), n_size, wptr->CStep(), n_size);
      *dststep = n_size;
    }
    if (wptr->RDtype() == BTLA_DTYPE::BF16) {
      auto aptr = wptr->template RPtr<utils::bf16>();
      kernel::wrapper::Memcpy2DBf16CvtFp32::forward<ISA_T>(
          aptr + k_offset / wptr->mBlockSize * wptr->CStep() + n_offset, *dstptr,
          utils::updiv(k_size, wptr->mBlockSize), n_size, wptr->CStep() * 2, n_size * 4, false);
      *dststep = n_size;
    }
    return BTLA_CODE::Success;
  }

 protected:
  template <typename T>
  static inline BTLA_CODE getFpKBlockWeight(T** dstptr, int* dststep, int k_size, int n_size, int k_offset,
                                            int n_offset, const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.packedW;
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    int constexpr ColSize = _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      if (wptr->SDtype() == BTLA_DTYPE::F32) {
        auto sptr = wptr->template SPtr<float>() + n_offset + i;
        if (wptr->mDType == BTLA_DTYPE::S4_CLIP) {
          kernel::wrapper::DecompressKBlockS4S8Fp<T>::template forward<ISA_T, BTLA_DTYPE::S4_CLIP>(
              wptr->template WPtr<utils::int4x2>() + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2 +
                  i * KPad / 2,
              *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, tmpcache, cachesize);
        } else if (wptr->mDType == BTLA_DTYPE::S4_FULLRANGE) {
          kernel::wrapper::DecompressKBlockS4S8Fp<T>::template forward<ISA_T, BTLA_DTYPE::S4_FULLRANGE>(
              wptr->template WPtr<utils::int4x2>() + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2 +
                  i * KPad / 2,
              *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, tmpcache, cachesize);
        } else if (wptr->mDType == BTLA_DTYPE::S8) {
          kernel::wrapper::DecompressKBlockS8S8Fp<T>::template forward<ISA_T>(
              wptr->template WPtr<int8_t>() + n_offset * KPad + k_offset * _GemmCore_T::NTILE + i * KPad,
              *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, tmpcache, cachesize);
        }
      } else if (wptr->SDtype() == BTLA_DTYPE::BF16) {
        auto sptr = wptr->template SPtr<utils::bf16>() + n_offset + i;
        if (wptr->mDType == BTLA_DTYPE::S4_CLIP) {
          kernel::wrapper::DecompressKBlockS4S8Fp<T>::template forward<ISA_T, BTLA_DTYPE::S4_CLIP>(
              wptr->template WPtr<utils::int4x2>() + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2 +
                  i * KPad / 2,
              *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, tmpcache, cachesize);
        } else if (wptr->mDType == BTLA_DTYPE::S4_FULLRANGE) {
          kernel::wrapper::DecompressKBlockS4S8Fp<T>::template forward<ISA_T, BTLA_DTYPE::S4_FULLRANGE>(
              wptr->template WPtr<utils::int4x2>() + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2 +
                  i * KPad / 2,
              *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, tmpcache, cachesize);
        } else if (wptr->mDType == BTLA_DTYPE::S8) {
          kernel::wrapper::DecompressKBlockS8S8Fp<T>::template forward<ISA_T>(
              wptr->template WPtr<int8_t>() + n_offset * KPad + k_offset * _GemmCore_T::NTILE + i * KPad,
              *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, tmpcache, cachesize);
        }
      }
    }
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

  template <typename _T>
  static inline BTLA_CODE getFpWeight(_T** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                      const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.packedW;
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    int constexpr ColSize = _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      auto zptr = wptr->template ZPtr<int8_t>();
      if (wptr->SDtype() == BTLA_DTYPE::F32) {
        auto sptr = wptr->template SPtr<float>() + n_offset + i;
        if (wptr->mDType == BTLA_DTYPE::S4_CLIP) {
          kernel::wrapper::DecompressKBlockS4Fp<_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, float,
                                                                                             BTLA_DTYPE::S4_CLIP>(
              wptr->template WPtr<utils::int4x2>() + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2 +
                  i * KPad / 2,
              *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              zptr != nullptr ? zptr + n_offset + i : nullptr, k_offset / _GemmCore_T::PACK_ROW,
              wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else if (wptr->mDType == BTLA_DTYPE::S4_FULLRANGE) {
          kernel::wrapper::DecompressKBlockS4Fp<_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, float,
                                                                                             BTLA_DTYPE::S4_FULLRANGE>(
              wptr->template WPtr<utils::int4x2>() + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2 +
                  i * KPad / 2,
              *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              zptr != nullptr ? zptr + n_offset + i : nullptr, k_offset / _GemmCore_T::PACK_ROW,
              wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else if (wptr->mDType == BTLA_DTYPE::S8) {
          kernel::wrapper::DecompressKBlockS8Fp<_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, float>(
              wptr->template WPtr<int8_t>() + n_offset * KPad + k_offset * _GemmCore_T::NTILE + i * KPad,
              *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              zptr != nullptr ? zptr + n_offset + i : nullptr, k_offset / _GemmCore_T::PACK_ROW,
              wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else {
          assert(0);
        }
      } else if (wptr->SDtype() == BTLA_DTYPE::BF16) {
        auto sptr = wptr->template SPtr<utils::bf16>() + n_offset + i;
        if (wptr->mDType == BTLA_DTYPE::S4_CLIP) {
          kernel::wrapper::DecompressKBlockS4Fp<_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, utils::bf16,
                                                                                             BTLA_DTYPE::S4_CLIP>(
              wptr->template WPtr<utils::int4x2>() + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2 +
                  i * KPad / 2,
              *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              zptr != nullptr ? zptr + n_offset + i : nullptr, k_offset / _GemmCore_T::PACK_ROW,
              wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else if (wptr->mDType == BTLA_DTYPE::S4_FULLRANGE) {
          kernel::wrapper::DecompressKBlockS4Fp<_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, utils::bf16,
                                                                                             BTLA_DTYPE::S4_FULLRANGE>(
              wptr->template WPtr<utils::int4x2>() + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2 +
                  i * KPad / 2,
              *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              zptr != nullptr ? zptr + n_offset + i : nullptr, k_offset / _GemmCore_T::PACK_ROW,
              wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else if (wptr->mDType == BTLA_DTYPE::S8) {
          kernel::wrapper::DecompressKBlockS8Fp<_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, utils::bf16>(
              wptr->template WPtr<int8_t>() + n_offset * KPad + k_offset * _GemmCore_T::NTILE + i * KPad,
              *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              zptr != nullptr ? zptr + n_offset + i : nullptr, k_offset / _GemmCore_T::PACK_ROW,
              wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else {
          assert(0);
        }
      }
    }
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

  static inline BTLA_CODE getQ8Weight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                      const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.packedW;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->template WPtr<int8_t>() + n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    kernel::wrapper::Memcpy2D::template forward<BTLA_ISA::NoSIMD, int8_t, int8_t>(
        bptr, *dstptr, n_size / _GemmCore_T::NTILE, _GemmCore_T::NTILE * k_size, _GemmCore_T::NTILE * KPad,
        _GemmCore_T::NTILE * k_size);
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

  static inline BTLA_CODE getQ4Weight(int8_t** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                      const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = _param.packedW;
    auto KPad = wptr->mKPad;
    auto bptr = wptr->template WPtr<utils::int4x2>() + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
    int constexpr ColSize = _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      if (wptr->mDType == BTLA_DTYPE::S4_CLIP) {
        kernel::wrapper::DecompressKBlockS4S8::template forward<ISA_T, BTLA_DTYPE::S4_CLIP>(
            bptr + i * KPad / 2, *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize);
      } else if (wptr->mDType == BTLA_DTYPE::S4_FULLRANGE) {
        kernel::wrapper::DecompressKBlockS4S8::template forward<ISA_T, BTLA_DTYPE::S4_FULLRANGE>(
            bptr + i * KPad / 2, *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize);
      }
    }
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

  virtual inline void quantRowBlock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                                    float* scales, int8_t* zero_points, void* stor) {
    auto ptr = reinterpret_cast<StorageWeight*>(stor);
    auto quant_dtype = ptr->mDType;
    if (quant_dtype == BTLA_DTYPE::S8) {
      kernel::wrapper::QuantizeSignIntRowBlock::forward<ISA_T, BTLA_DTYPE::S8>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                               scales, zero_points, ptr->mBlockSize);
    } else if (quant_dtype == BTLA_DTYPE::S4_FULLRANGE) {
      kernel::wrapper::QuantizeSignIntRowBlock::forward<ISA_T, BTLA_DTYPE::S4_FULLRANGE>(
          srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points, ptr->mBlockSize);
    } else if (quant_dtype == BTLA_DTYPE::S4_CLIP) {
      kernel::wrapper::QuantizeSignIntRowBlock::forward<ISA_T, BTLA_DTYPE::S4_CLIP>(
          srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points, ptr->mBlockSize);
    }
  }

  static inline BTLA_CODE doCompress(const int8_t* srcptr, void* dstptr, int row, int col, int ld_src, int ld_dst,
                                     BTLA_DTYPE quant_dtype) {
    if (quant_dtype == BTLA_DTYPE::S4_CLIP || quant_dtype == BTLA_DTYPE::S4_FULLRANGE) {
      return kernel::wrapper::CompressS8S4<_GemmCore_T::NTILE>::template forward<ISA_T>(
          srcptr, reinterpret_cast<utils::int4x2*>(dstptr), row, col, ld_src, ld_dst);
    } else if (quant_dtype == BTLA_DTYPE::F4_BNB || quant_dtype == BTLA_DTYPE::F4_NF4 ||
               quant_dtype == BTLA_DTYPE::F4_E2M1) {
      return kernel::wrapper::CompressFp4<_GemmCore_T::NTILE>::template forward<ISA_T>(
          srcptr, reinterpret_cast<utils::f4x2*>(dstptr), row, col, ld_src,
          ld_dst);  // ld_dst here not stride
    } else {
      assert(0);
      return BTLA_CODE::NotSupport;
    }
  }
};

struct ParamWeightKBlockNFloat {
  storage::gemm::StorageWeightKBlockNFloat* packedW;
};

template <class _GemmCore_T, BTLA_ISA ISA_T>
class WeightKBlockNFloat : public WeightKBlockNInteger<_GemmCore_T, ISA_T> {
 public:
  using Param = ParamWeightKBlockNInteger;  // NFloat storage Param same with NInteger storage.
  using StorageWeight = storage::gemm::StorageWeightKBlockNFloat;

  StorageWeight createStorage(const int N, const int K, int blocksize, BTLA_DTYPE fT, BTLA_DTYPE scaT) {
    int KPad = utils::padto(K, _GemmCore_T::KTILE);
    int NPad = utils::padto(N, _GemmCore_T::NTILE);
    StorageWeight tmp(_GemmCore_T::ID);
    tmp.resize(NPad, KPad, blocksize <= 0 ? KPad : blocksize, N, K, fT, scaT);
    return tmp;
  }

  inline BTLA_CODE getWeight(float** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                             const Param& _param, void* tmpcache, size_t cachesize) override {
    return getFpWeight(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
  }

  inline BTLA_CODE getWeight(utils::bf16** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                             const Param& _param, void* tmpcache, size_t cachesize) override {
    return getFpWeight(dstptr, dststep, k_size, n_size, k_offset, n_offset, _param, tmpcache, cachesize);
  }

  template <typename _DST_T>
  inline BTLA_CODE getFpWeight(_DST_T** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                               const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = reinterpret_cast<StorageWeight*>(_param.packedW);
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    char* bptr;
    if (wptr->mDType == BTLA_DTYPE::F8_E5M2 || wptr->mDType == BTLA_DTYPE::F8_E4M3) {
      bptr = wptr->template WPtr<char>() + n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    } else {
      bptr = wptr->template WPtr<char>() + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
    }
    int constexpr ColSize = _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      if (wptr->SDtype() == BTLA_DTYPE::F8_E8M0) {
        assert(wptr->mDType == BTLA_DTYPE::F8_E4M3 || wptr->mDType == BTLA_DTYPE::F8_E5M2);
        auto sptr = wptr->template SPtr<utils::f8>() + n_offset + i;
        kernel::wrapper::DecompressKBlockF8FP<_GemmCore_T::PACK_ROW>::template forward<ISA_T>(
            reinterpret_cast<utils::f8*>(bptr) + i * KPad, *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
            ColSize, ColSize, ColSize, sptr, k_offset / _GemmCore_T::PACK_ROW, wptr->mBlockSize / _GemmCore_T::PACK_ROW,
            NPad, wptr->mDType);
      } else if (wptr->SDtype() == BTLA_DTYPE::F32) {
        auto sptr = wptr->template SPtr<float>() + n_offset + i;
        auto f4ptr = reinterpret_cast<utils::f4x2*>(bptr + i * KPad / 2);
        auto fp_ptr = *dstptr + i * k_size;
        if (wptr->mDType == BTLA_DTYPE::F8_E4M3 || wptr->mDType == BTLA_DTYPE::F8_E5M2) {
          kernel::wrapper::DecompressKBlockF8FP<_GemmCore_T::PACK_ROW>::template forward<ISA_T>(
              reinterpret_cast<utils::f8*>(bptr) + i * KPad, *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
              ColSize, ColSize, ColSize, sptr, k_offset / _GemmCore_T::PACK_ROW,
              wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, wptr->mDType);
        } else if (wptr->mDType == BTLA_DTYPE::F4_NF4) {
          kernel::wrapper::DecompressKBlockF4Fp<_DST_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, float,
                                                                                                 BTLA_DTYPE::F4_NF4>(
              f4ptr, fp_ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              k_offset / _GemmCore_T::PACK_ROW, wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else if (wptr->mDType == BTLA_DTYPE::F4_E2M1) {
          kernel::wrapper::DecompressKBlockF4Fp<_DST_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, float,
                                                                                                 BTLA_DTYPE::F4_E2M1>(
              f4ptr, fp_ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              k_offset / _GemmCore_T::PACK_ROW, wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else if (wptr->mDType == BTLA_DTYPE::F4_BNB) {
          kernel::wrapper::DecompressKBlockF4Fp<_DST_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, float,
                                                                                                 BTLA_DTYPE::F4_BNB>(
              f4ptr, fp_ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              k_offset / _GemmCore_T::PACK_ROW, wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else {
          assert(0);
        }
      } else if (wptr->SDtype() == BTLA_DTYPE::BF16) {
        auto sptr = wptr->template SPtr<utils::bf16>() + n_offset + i;
        auto f4ptr = reinterpret_cast<utils::f4x2*>(bptr + i * KPad / 2);
        auto fp_ptr = *dstptr + i * k_size;
        if (wptr->mDType == BTLA_DTYPE::F4_NF4) {
          kernel::wrapper::DecompressKBlockF4Fp<_DST_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, utils::bf16,
                                                                                                 BTLA_DTYPE::F4_NF4>(
              f4ptr, fp_ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              k_offset / _GemmCore_T::PACK_ROW, wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else if (wptr->mDType == BTLA_DTYPE::F4_E2M1) {
          kernel::wrapper::DecompressKBlockF4Fp<_DST_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, utils::bf16,
                                                                                                 BTLA_DTYPE::F4_E2M1>(
              f4ptr, fp_ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              k_offset / _GemmCore_T::PACK_ROW, wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else if (wptr->mDType == BTLA_DTYPE::F4_BNB) {
          kernel::wrapper::DecompressKBlockF4Fp<_DST_T, _GemmCore_T::PACK_ROW>::template forward<ISA_T, utils::bf16,
                                                                                                 BTLA_DTYPE::F4_BNB>(
              f4ptr, fp_ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, sptr,
              k_offset / _GemmCore_T::PACK_ROW, wptr->mBlockSize / _GemmCore_T::PACK_ROW, NPad, tmpcache, cachesize);
        } else {
          assert(0);
        }
      } else {
        assert(0);
      }
    }
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

  template <typename T>
  static inline BTLA_CODE getKBlockWeight(T** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                                          const Param& _param, void* tmpcache, size_t cachesize) {
    auto wptr = reinterpret_cast<StorageWeight*>(_param.packedW);
    auto NPad = wptr->mNPad;
    auto KPad = wptr->mKPad;
    char* bptr;
    if (wptr->mDType == BTLA_DTYPE::F8_E4M3 || wptr->mDType == BTLA_DTYPE::F8_E5M2) {
      bptr = wptr->template WPtr<char>() + n_offset * KPad + k_offset * _GemmCore_T::NTILE;
    } else {
      bptr = wptr->template WPtr<char>() + n_offset * KPad / 2 + k_offset * _GemmCore_T::NTILE / 2;
    }
    int constexpr ColSize = _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW;
    for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
      if (wptr->mDType == BTLA_DTYPE::F8_E4M3 || wptr->mDType == BTLA_DTYPE::F8_E5M2) {
        kernel::wrapper::DecompressKBlockF8FpNoScale<T>::template forward<ISA_T>(
            reinterpret_cast<utils::f8*>(bptr) + i * KPad, *dstptr + i * k_size, k_size / _GemmCore_T::PACK_ROW,
            ColSize, ColSize, ColSize, tmpcache, cachesize, wptr->mDType);
      } else {
        auto f4ptr = reinterpret_cast<utils::f4x2*>(bptr + i * KPad / 2);
        auto fp_ptr = *dstptr + i * k_size;
        if (wptr->mDType == BTLA_DTYPE::F4_NF4) {
          kernel::wrapper::DecompressKBlockF4FpNoscale<T>::template forward<ISA_T, BTLA_DTYPE::F4_NF4>(
              f4ptr, fp_ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, tmpcache, cachesize);
        } else if (wptr->mDType == BTLA_DTYPE::F4_E2M1) {
          kernel::wrapper::DecompressKBlockF4FpNoscale<T>::template forward<ISA_T, BTLA_DTYPE::F4_E2M1>(
              f4ptr, fp_ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, tmpcache, cachesize);
        } else if (wptr->mDType == BTLA_DTYPE::F4_BNB) {
          kernel::wrapper::DecompressKBlockF4FpNoscale<T>::template forward<ISA_T, BTLA_DTYPE::F4_BNB>(
              f4ptr, fp_ptr, k_size / _GemmCore_T::PACK_ROW, ColSize, ColSize, ColSize, tmpcache, cachesize);
        } else {
          assert(0);
        }
      }
    }
    *dststep = k_size;
    return BTLA_CODE::Success;
  }

 protected:
  void quantRowBlock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst, float* scales,
                     int8_t* zero_points, void* stor) override {
    auto ptr = reinterpret_cast<StorageWeight*>(stor);
    auto quant_dtype = ptr->mDType;
    if (quant_dtype == BTLA_DTYPE::F8_E4M3) {
      kernel::wrapper::QuantizeF8RowBlock::forward<ISA_T, BTLA_DTYPE::F8_E4M3>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                               scales, ptr->mBlockSize, ptr->SDtype());
    } else if (quant_dtype == BTLA_DTYPE::F8_E5M2) {
      kernel::wrapper::QuantizeF8RowBlock::forward<ISA_T, BTLA_DTYPE::F8_E5M2>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                               scales, ptr->mBlockSize, ptr->SDtype());
    } else if (quant_dtype == BTLA_DTYPE::F4_BNB) {
      kernel::wrapper::QuantizeF4RowBlock::forward<ISA_T, BTLA_DTYPE::F4_BNB>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                              scales, zero_points, ptr->mBlockSize);
    } else if (quant_dtype == BTLA_DTYPE::F4_E2M1) {
      kernel::wrapper::QuantizeF4RowBlock::forward<ISA_T, BTLA_DTYPE::F4_E2M1>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                               scales, zero_points, ptr->mBlockSize);
    } else if (quant_dtype == BTLA_DTYPE::F4_NF4) {
      kernel::wrapper::QuantizeF4RowBlock::forward<ISA_T, BTLA_DTYPE::F4_NF4>(srcptr, dstptr, row, col, ld_src, ld_dst,
                                                                              scales, zero_points, ptr->mBlockSize);
    } else {
      assert(0);
    }
  }
};
}  // namespace gemm
}  // namespace prologue_b
}  // namespace bestla
