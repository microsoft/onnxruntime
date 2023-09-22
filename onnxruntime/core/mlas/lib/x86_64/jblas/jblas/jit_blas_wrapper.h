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
#include "jit_blas_prologue.h"
#include "jit_blas_utils.h"
#include "kernel_avx512f.h"
#include "kernel_jit.h"
#include "kernel_ref.h"

namespace jblas {
namespace wrapper {
namespace gemm_pack_weight {

template <JBLAS_ISA _RT_ISA_T, class _GemmCore_T, template <class _T, JBLAS_ISA> class _PrologueA_T,
          template <class _T, JBLAS_ISA> class _PrologueB_T, template <JBLAS_ISA> class _Epilogue_T>
class GemmLauncherPackWeight {
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
    void* workspace;
  };
  struct ParallelConfig {
    const int rowidx, colidx;
    const int rowsize, colsize;
    const int MStep, NStep, KStep;
    const size_t StackSize;
  };
  _GemmCore_T mGemmCore;
  PrologueA mProA;
  PrologueB mProB;
  Epilogue mEpilogue;
  GemmLauncherPackWeight() {}

  void launch(const ParallelConfig& _config, const Param& _param) {
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
        run_block(_config, _param, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpC);
      }
    }
  }

 protected:
  void run_block(const ParallelConfig& _config, const Param& _param, int blk_m, int blk_n, int blk_msize, int blk_nsize,
                 AType* tmpA, BType* tmpB, CType* tmpC) {
    int n_padded = utils::padto(blk_nsize, GemmCore::NTILE);
    for (int iterk = 0; iterk < _param.K; iterk += _config.KStep) {
      int k_remain = utils::remainsize(iterk, _param.K, _config.KStep);
      int k_padded = utils::padto(k_remain, GemmCore::KTILE);
      int k_paddedle = utils::padto_le(k_remain, GemmCore::KTILE);
      auto bptr_cache = tmpB;
      int bcache_step = 0;
      mProB.getWeight(&bptr_cache, &bcache_step, k_padded, n_padded, iterk, _config.colidx + blk_n, _param.paramB);
      int bcache_stride = bcache_step * sizeof(BType);
      for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
        int m_remain = utils::remainsize(i, blk_msize, GemmCore::MTILE);
        auto cptr_cache = tmpC + i * _config.NStep;
        int ccache_stride = _config.NStep * sizeof(CType);

        if (k_paddedle) {
          AType* aptr_cache = tmpA;
          int acache_step = 0;
          mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_paddedle,
                              (blk_m + i + _config.rowidx), iterk);
          mGemmCore.forward(aptr_cache, bptr_cache, cptr_cache, m_remain, n_padded, k_paddedle,
                            acache_step * sizeof(AType), bcache_stride, ccache_stride, iterk);
        }
        int k_tail = k_remain - k_paddedle;
        if (k_tail) {
          AType* aptr_cache = tmpA;
          int acache_step = 0;
          mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_tail, (blk_m + i + _config.rowidx),
                              iterk + k_paddedle);
          mGemmCore.forward(aptr_cache, bptr_cache + k_paddedle * GemmCore::NTILE, cptr_cache, m_remain, n_padded,
                            GemmCore::KTILE, acache_step * sizeof(AType), bcache_stride, ccache_stride,
                            iterk + k_paddedle);
        }
      }
    }
    mEpilogue.forward(tmpC, _config.NStep, (_config.rowidx + blk_m), _config.colidx + blk_n, blk_msize, blk_nsize,
                      _param.paramC);
  }
};

template <class _Launcher_T, template <class _T> class _Parallel_T>
class GemmInterfacePackWeight {
 public:
  using Arguments = typename _Launcher_T::Param;
  using Config = typename _Launcher_T::ParallelConfig;
  using ActivationType = typename _Launcher_T::PrologueA;
  using WeightType = typename _Launcher_T::PrologueB;
  using Epilogue = typename _Launcher_T::Epilogue;
  using GemmCore = typename _Launcher_T::GemmCore;
  using Parallel = _Parallel_T<GemmCore>;

  GemmInterfacePackWeight() {}

  WeightType* getWeightPtr() { return &mLauncher.mProB; }

  JBLAS_CODE compute(const Arguments& _param) {
    auto cb = utils::CpuBase();
    auto para = Parallel();
    para.update(_param.M, _param.N, _param.K, cb.mNumThreads);
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
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

template <class _Launcher_T, template <class _T> class _Parallel_T>
class GemmInterfaceParallelAB {
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
    auto cb = utils::CpuBase();
    auto para = Parallel();
    para.update(_param.M, _param.N, _param.K, cb.mNumThreads);
    auto paraA = getActivationPtr()->createParallel(_param.M, _param.K);
    auto paraB = getWeightPtr()->createParallel(_param.K, _param.N);
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
}  // namespace gemm_pack_weight
namespace gemm_default {
template <class T>
using DefaultParallel = jblas::utils::parallel::Parallel2DGemm<T>;
namespace avx2 {
JBLAS_ISA constexpr DefaultISA = JblasAVX2;
using GemmKernel = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<  //
        DefaultISA,                                            //
        jblas::gemm::GemmCore_Row_NN_4x24_AVX2,             //
        jblas::prologue::gemm::ActivationBase,                 //
        jblas::prologue::gemm::WeightPack,                     //
        jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    DefaultParallel>;
}
namespace avx_vnni {
JBLAS_ISA constexpr DefaultISA = JblasAVX_VNNI;
using GemmKernel48 = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<  //
        DefaultISA,                                            //
        jblas::gemm::GemmCore_Row_NN_2x48_AVX_VNNI,         //
        jblas::prologue::gemm::ActivationBase,                 //
        jblas::prologue::gemm::WeightPack,                     //
        jblas::epilogue::gemm::AlphaBetaProcessS32U8>,
    DefaultParallel>;
}
namespace avx512f {
JBLAS_ISA constexpr DefaultISA = JblasAVX512F;
using GemmKernel = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<  //
        DefaultISA,                                            //
        jblas::gemm::GemmCore_Row_NN_8x48_AVX512F,             //
        jblas::prologue::gemm::ActivationBase,                 //
        jblas::prologue::gemm::WeightPack,                     //
        jblas::epilogue::gemm::AlphaBetaProcessFp32>,
    DefaultParallel>;
}  // namespace avx512f
namespace avx512_vnni {
JBLAS_ISA constexpr DefaultISA = JblasAVX512_VNNI;
using GemmKernel = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<  //
        DefaultISA,                                            //
        jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI,         //
        jblas::prologue::gemm::ActivationBase,                 //
        jblas::prologue::gemm::WeightPack,                     //
        jblas::epilogue::gemm::AlphaBetaProcessS32U8>,
    DefaultParallel>;

using GemmKernelDynamicU8 = jblas::wrapper::gemm_pack_weight::GemmInterfaceParallelAB<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<  //
        DefaultISA,                                            //
        jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI,         //
        jblas::prologue::gemm::ActivationFp32AsymU8Quantize,   //
        jblas::prologue::gemm::WeightPack,                     //
        jblas::epilogue::gemm::ZpDequantInt32ToFp32>,
    DefaultParallel>;
}  // namespace avx512_vnni

namespace amx_bf16 {
JBLAS_ISA constexpr DefaultISA = JblasAMX_BF16;
using GemmKernelPackedWeightNN = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<  //
        DefaultISA,                                            //
        jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16,           //
        jblas::prologue::gemm::ActivationBase,                 //
        jblas::prologue::gemm::WeightPack,                     //
        jblas::epilogue::gemm::AccumulatorWriteBackFp32Bf16>,
    DefaultParallel>;
using GemmKernelPackedWeightNN_48 = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<  //
        DefaultISA,                                            //
        jblas::gemm::GemmCore_Row_NN_16x48_AMX_BF16,           //
        jblas::prologue::gemm::ActivationBase,                 //
        jblas::prologue::gemm::WeightPack,                     //
        jblas::epilogue::gemm::AccumulatorWriteBackFp32Bf16>,
    DefaultParallel>;
}  // namespace amx_bf16

namespace amx_int8 {
JBLAS_ISA constexpr DefaultISA = JblasAMX_INT8;

using GemmKernel48 = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<  //
        DefaultISA,                                            //
        jblas::gemm::GemmCore_Row_NN_16x48_AMX_U8S8,           //
        jblas::prologue::gemm::ActivationBase,                 //
        jblas::prologue::gemm::WeightPack,                     //
        jblas::epilogue::gemm::AlphaBetaProcessS32U8>,
    DefaultParallel>;

using GemmKernel = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<  //
        DefaultISA,                                            //
        jblas::gemm::GemmCore_Row_NN_16x64_AMX_U8S8,           //
        jblas::prologue::gemm::ActivationBase,                 //
        jblas::prologue::gemm::WeightPack,                     //
        jblas::epilogue::gemm::AlphaBetaProcessS32U8>,
    DefaultParallel>;

using GemmKernelSSFp32 = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<  //
        DefaultISA,                                            //
        jblas::gemm::GemmCore_Row_NN_16x48_AMX_S8S8,           //
        jblas::prologue::gemm::ActivationBase,                 //
        jblas::prologue::gemm::WeightPack,                     //
        jblas::epilogue::gemm::DequantInt32ToFp32>,
    DefaultParallel>;

using GemmKernelDynamicQuant = jblas::wrapper::gemm_pack_weight::GemmInterfaceParallelAB<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<  //
        DefaultISA,                                            //
        jblas::gemm::GemmCore_Row_NN_16x48_AMX_S8S8,           //
        jblas::prologue::gemm::ActivationFp32SymS8Quantize,    //
        jblas::prologue::gemm::WeightPack,                     //
        jblas::epilogue::gemm::DequantInt32ToFp32>,
    DefaultParallel>;
}  // namespace amx_int8

namespace avx512_fp16 {
JBLAS_ISA constexpr DefaultISA = JblasAVX512_FP16;
using GemmKernel = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<  //
        DefaultISA,                                            //
        jblas::gemm::GemmCore_Row_NN_8x64_AVX512_FP16,         //
        jblas::prologue::gemm::ActivationBase,                 //
        jblas::prologue::gemm::WeightPack,                     //
        jblas::epilogue::gemm::AccumulatorWriteBackFp16>,
    DefaultParallel>;
using GemmKernel_96 = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<  //
        DefaultISA,                                            //
        jblas::gemm::GemmCore_Row_NN_8x96_AVX512_FP16,         //
        jblas::prologue::gemm::ActivationBase,                 //
        jblas::prologue::gemm::WeightPack,                     //
        jblas::epilogue::gemm::AccumulatorWriteBackFp16>,
    DefaultParallel>;
}  // namespace avx512_fp16
}  // namespace gemm_default
}  // namespace wrapper
}  // namespace jblas
