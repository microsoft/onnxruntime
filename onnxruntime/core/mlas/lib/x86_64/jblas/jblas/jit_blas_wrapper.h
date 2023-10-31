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
#include <thread>

#include "jit_blas_epilogue.h"
#include "jit_blas_gemm.h"
#include "jit_blas_prologue_a.h"
#include "jit_blas_prologue_b.h"
#include "jit_blas_utils.h"
#include "kernel_avx512f.h"
#include "kernel_jit.h"
#include "kernel_ref.h"

namespace jblas {
namespace wrapper {
namespace gemm {

template <JBLAS_ISA _RT_ISA_T, class _GemmCore_T, template <class _T, JBLAS_ISA> class _PrologueA_T,
          template <class _T, JBLAS_ISA> class _PrologueB_T, template <JBLAS_ISA> class _Epilogue_T>
class LauncherBase {
 public:
  using GemmCore = _GemmCore_T;
  using PrologueA = _PrologueA_T<GemmCore, _RT_ISA_T>;
  using PrologueB = _PrologueB_T<GemmCore, _RT_ISA_T>;
  using Epilogue = _Epilogue_T<_RT_ISA_T>;
  using AType = typename GemmCore::AType;
  using AParam = typename PrologueA::Param;
  using BType = typename GemmCore::BType;
  using BParam = typename PrologueB::Param;
  using CType = typename GemmCore::CType;
  using EpiParam = typename Epilogue::Param;
  static_assert(GemmCore::ISA <= _RT_ISA_T, "RunTime ISA should cover GEMM's ISA");
  struct Param {
    const int M, N, K;
    const AParam paramA;
    const BParam paramB;
    const EpiParam paramC;
  };
  _GemmCore_T mGemmCore;
  PrologueA mProA;
  PrologueB mProB;
  Epilogue mEpilogue;

  void run(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
    auto StackTmp = alloca(_config.l2cachesize);
    auto tmpB = (BType*)(StackTmp);
    auto tmpA = (AType*)(tmpB + (size_t)_config.block[1] * _config.block[2]);
    auto tmpC = (CType*)(tmpA + (size_t)GemmCore::MTILE * _config.block[2]);
    auto tmpCache = (void*)(tmpC + (size_t)_config.block[0] * _config.block[1]);
    for (int itern = 0; itern < _config.size[1]; itern += _config.block[1]) {
      int n_remain = utils::remainsize(itern, _config.size[1], _config.block[1]);
      for (int iterm = 0; iterm < _config.size[0]; iterm += _config.block[0]) {
        int m_remain = utils::remainsize(iterm, _config.size[0], _config.block[0]);
        run_block(_param, _config, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpC, tmpCache);
      }
    }
  }

 protected:
  void run_block(const Param& _param, const parallel::gemm::ThreadProblemBase& _config, int blk_m, int blk_n,
                 int blk_msize, int blk_nsize, AType* tmpA, BType* tmpB, CType* tmpC, void* tmpcache) {
    int n_padded = utils::padto(blk_nsize, GemmCore::NTILE);
    for (int iterk = 0; iterk < _param.K; iterk += _config.block[2]) {
      int k_remain = utils::remainsize(iterk, _param.K, _config.block[2]);
      int k_padded = utils::padto(k_remain, GemmCore::KTILE);
      int k_paddedle = utils::padto_le(k_remain, GemmCore::KTILE);
      auto bptr_cache = tmpB;
      int bcache_step = 0;
      mProB.getWeight(&bptr_cache, &bcache_step, k_padded, n_padded, iterk, _config.loc[1] + blk_n, _param.paramB,
                      tmpcache, _config.tmpcachesize);
      int bcache_stride = bcache_step * sizeof(BType);
      for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
        int m_remain = utils::remainsize(i, blk_msize, GemmCore::MTILE);
        auto cptr_cache = tmpC + i * _config.block[1];
        int ccache_stride = _config.block[1] * sizeof(CType);
        if (k_paddedle) {
          AType* aptr_cache = tmpA;
          int acache_step = 0;
          mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_paddedle,
                              (blk_m + i + _config.loc[0]), iterk, tmpcache, _config.tmpcachesize);
          mGemmCore.forward(aptr_cache, bptr_cache, cptr_cache, m_remain, n_padded, k_paddedle,
                            acache_step * sizeof(AType), bcache_stride, ccache_stride, iterk, tmpcache,
                            _config.tmpcachesize);
        }
        int k_tail = k_remain - k_paddedle;
        if (k_tail) {
          AType* aptr_cache = tmpA;
          int acache_step = 0;
          mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_tail, (blk_m + i + _config.loc[0]),
                              iterk + k_paddedle, tmpcache, _config.tmpcachesize);
          mGemmCore.forward(aptr_cache, bptr_cache + k_paddedle * GemmCore::NTILE, cptr_cache, m_remain, n_padded,
                            GemmCore::KTILE, acache_step * sizeof(AType), bcache_stride, ccache_stride,
                            iterk + k_paddedle, tmpcache, _config.tmpcachesize);
        }
      }
    }
    mEpilogue.forward(tmpC, _config.block[1], (_config.loc[0] + blk_m), _config.loc[1] + blk_n, blk_msize, blk_nsize,
                      _param.paramC, tmpcache, _config.tmpcachesize);
  }
};

template <JBLAS_ISA _RT_ISA_T, class _GemmCore_T, template <class _T, JBLAS_ISA> class _PrologueA_T,
          template <class _T, JBLAS_ISA> class _PrologueB_T, template <JBLAS_ISA> class _BlockEpilogue_T,
          template <JBLAS_ISA> class _Epilogue_T>
class LauncherKBlock {
 public:
  using GemmCore = _GemmCore_T;
  using PrologueA = _PrologueA_T<GemmCore, _RT_ISA_T>;
  using PrologueB = _PrologueB_T<GemmCore, _RT_ISA_T>;
  using Epilogue = _Epilogue_T<_RT_ISA_T>;
  using BlockEpilogue = _BlockEpilogue_T<_RT_ISA_T>;
  using AType = typename GemmCore::AType;
  using AParam = typename PrologueA::Param;
  using BType = typename GemmCore::BType;
  using BParam = typename PrologueB::Param;
  using CType = typename GemmCore::CType;
  using BEpiParam = typename BlockEpilogue::Param;
  using EpiParam = typename Epilogue::Param;
  using AccType = float;
  static_assert(GemmCore::ISA <= _RT_ISA_T, "RunTime ISA should cover GEMM's ISA");
  struct Param {
    const int M, N, K, KBlock;
    const AParam paramA;
    const BParam paramB;
    const BEpiParam paramBlk;
    const EpiParam paramC;
  };
  _GemmCore_T mGemmCore;
  PrologueA mProA;
  PrologueB mProB;
  BlockEpilogue mBlockEpi;
  Epilogue mEpilogue;

  void run(const Param& _param, const parallel::gemm::ThreadProblemBase& _config) {
    auto StackTmp = alloca(_config.l2cachesize);
    auto tmpB = (BType*)(StackTmp);
    auto tmpA = (AType*)(tmpB + (size_t)_config.block[1] * _config.block[2]);
    auto tmpC = (AccType*)(tmpA + (size_t)GemmCore::MTILE * _config.block[2]);
    auto tmpBlk = (CType*)(tmpC + (size_t)_config.block[0] * _config.block[1]);
    auto tmpCache = (void*)(tmpBlk + (size_t)_config.block[0] * _config.block[1]);
    for (int itern = 0; itern < _config.size[1]; itern += _config.block[1]) {
      int n_remain = utils::remainsize(itern, _config.size[1], _config.block[1]);
      for (int iterm = 0; iterm < _config.size[0]; iterm += _config.block[0]) {
        int m_remain = utils::remainsize(iterm, _config.size[0], _config.block[0]);
        std::memset(tmpC, 0, _config.block[0] * _config.block[1] * sizeof(AccType));
        if (_param.KBlock <= _config.block[2]) {
          run_block(_param, _config, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpBlk, tmpC, tmpCache);
        } else {
          run_block_large(_param, _config, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpBlk, tmpC, tmpCache);
        }
      }
    }
  }

 protected:
  void run_block(const Param& _param, const parallel::gemm::ThreadProblemBase& _config, int blk_m, int blk_n,
                 int blk_msize, int blk_nsize, AType* tmpA, BType* tmpB, CType* tmpBlk, AccType* tmpC, void* tmpcache) {
    int n_padded = utils::padto(blk_nsize, GemmCore::NTILE);
    assert(_config.block[2] == _param.KBlock);
    for (int iterk = 0; iterk < _param.K; iterk += _config.block[2]) {
      int k_remain = utils::remainsize(iterk, _param.K, _config.block[2]);
      int k_padded = utils::padto(k_remain, GemmCore::KTILE);
      int k_paddedle = utils::padto_le(k_remain, GemmCore::KTILE);
      auto bptr_cache = tmpB;
      int bcache_step = 0;
      mProB.getKBlockWeight(&bptr_cache, &bcache_step, k_padded, n_padded, iterk, _config.loc[1] + blk_n, _param.paramB,
                            tmpcache, _config.tmpcachesize);
      int bcache_stride = bcache_step * sizeof(BType);
      for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
        int m_remain = utils::remainsize(i, blk_msize, GemmCore::MTILE);
        auto cptr_cache = tmpBlk + i * _config.block[1];
        int ccache_stride = _config.block[1] * sizeof(CType);
        if (k_paddedle) {
          AType* aptr_cache = tmpA;
          int acache_step = 0;
          mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_paddedle,
                              (blk_m + i + _config.loc[0]), iterk, tmpcache, _config.tmpcachesize);
          mGemmCore.forward(aptr_cache, bptr_cache, cptr_cache, m_remain, n_padded, k_paddedle,
                            acache_step * sizeof(AType), bcache_stride, ccache_stride, 0, tmpcache,
                            _config.tmpcachesize);
        }
        int k_tail = k_remain - k_paddedle;
        if (k_tail) {
          AType* aptr_cache = tmpA;
          int acache_step = 0;
          mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_tail, (blk_m + i + _config.loc[0]),
                              iterk + k_paddedle, tmpcache, _config.tmpcachesize);
          mGemmCore.forward(aptr_cache, bptr_cache + k_paddedle * GemmCore::NTILE, cptr_cache, m_remain, n_padded,
                            k_tail, acache_step * sizeof(AType), bcache_stride, ccache_stride, 0 + k_paddedle, tmpcache,
                            _config.tmpcachesize);
        }
      }
      mBlockEpi.forward(tmpBlk, tmpC, _config.block[1], (_config.loc[0] + blk_m), _config.loc[1] + blk_n,
                        iterk / _param.KBlock, blk_msize, blk_nsize, _param.paramBlk, tmpcache, _config.tmpcachesize);
    }
    auto cachewithblk = _config.tmpcachesize + (size_t)_config.block[0] * _config.block[1] * sizeof(CType);
    mEpilogue.forward(tmpC, _config.block[1], (_config.loc[0] + blk_m), _config.loc[1] + blk_n, blk_msize, blk_nsize,
                      _param.paramC, tmpBlk, cachewithblk);
  }

  void run_block_large(const Param& _param, const parallel::gemm::ThreadProblemBase& _config, int blk_m, int blk_n,
                       int blk_msize, int blk_nsize, AType* tmpA, BType* tmpB, CType* tmpBlk, AccType* tmpC,
                       void* tmpcache) {
    int n_padded = utils::padto(blk_nsize, GemmCore::NTILE);
    for (int iterk = 0; iterk < _param.K; iterk += _param.KBlock) {
      for (int iblkk = 0; iblkk < _param.KBlock; iblkk += _config.block[2]) {
        int k_remain = utils::remainsize(iterk + iblkk, _param.K, _config.block[2]);
        int k_padded = utils::padto(k_remain, GemmCore::KTILE);
        int k_paddedle = utils::padto_le(k_remain, GemmCore::KTILE);
        auto bptr_cache = tmpB;
        int bcache_step = 0;
        mProB.getKBlockWeight(&bptr_cache, &bcache_step, k_padded, n_padded, iterk + iblkk, _config.loc[1] + blk_n,
                              _param.paramB, tmpcache, _config.tmpcachesize);
        int bcache_stride = bcache_step * sizeof(BType);
        for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
          int m_remain = utils::remainsize(i, blk_msize, GemmCore::MTILE);
          auto cptr_cache = tmpBlk + i * _config.block[1];
          int ccache_stride = _config.block[1] * sizeof(CType);
          if (k_paddedle) {
            AType* aptr_cache = tmpA;
            int acache_step = 0;
            mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_paddedle,
                                (blk_m + i + _config.loc[0]), iterk + iblkk, tmpcache, _config.tmpcachesize);
            mGemmCore.forward(aptr_cache, bptr_cache, cptr_cache, m_remain, n_padded, k_paddedle,
                              acache_step * sizeof(AType), bcache_stride, ccache_stride, iblkk, tmpcache,
                              _config.tmpcachesize);
          }
          int k_tail = k_remain - k_paddedle;
          if (k_tail) {
            AType* aptr_cache = tmpA;
            int acache_step = 0;
            mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_tail,
                                (blk_m + i + _config.loc[0]), iterk + k_paddedle + iblkk, tmpcache,
                                _config.tmpcachesize);
            mGemmCore.forward(aptr_cache, bptr_cache + k_paddedle * GemmCore::NTILE, cptr_cache, m_remain, n_padded,
                              k_tail, acache_step * sizeof(AType), bcache_stride, ccache_stride, iblkk + k_paddedle,
                              tmpcache, _config.tmpcachesize);
          }
        }
      }
      mBlockEpi.forward(tmpBlk, tmpC, _config.block[1], (_config.loc[0] + blk_m), _config.loc[1] + blk_n,
                        iterk / _param.KBlock, blk_msize, blk_nsize, _param.paramBlk, tmpcache, _config.tmpcachesize);
    }
    auto cachewithblk = _config.tmpcachesize + (size_t)_config.block[0] * _config.block[1] * sizeof(CType);
    mEpilogue.forward(tmpC, _config.block[1], (_config.loc[0] + blk_m), _config.loc[1] + blk_n, blk_msize, blk_nsize,
                      _param.paramC, tmpBlk, cachewithblk);
  }
};
}  // namespace gemm
}  // namespace wrapper
}  // namespace jblas
