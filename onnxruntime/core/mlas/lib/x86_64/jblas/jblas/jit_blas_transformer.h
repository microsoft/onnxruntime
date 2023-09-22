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
#include "jit_blas_weight_compression.h"
#include "jit_blas_wrapper.h"
#include "kernel_wrapper.h"

namespace jblas {
namespace wrapper {
namespace transformer {
// compared with BatchGemm, QKV has the same activation matrix.
// This is the reason why making it a new template.
template <class _Launcher_T, template <class _T> class _Parallel_T>
class QKVGemmInterfacePackWeight {
 public:
  struct Arguments {
    const int M, N, K, Batch;
    const typename _Launcher_T::AParam paramA;
    const typename _Launcher_T::BParam* paramsB;
    const typename _Launcher_T::EpiParam* paramsC;
    void* workspace;
  };
  using Config = typename _Launcher_T::ParallelConfig;
  using ActivationType = typename _Launcher_T::PrologueA;
  using WeightType = typename _Launcher_T::PrologueB;
  using GemmCore = typename _Launcher_T::GemmCore;
  using LArguments = typename _Launcher_T::Param;
  using CParam = typename _Launcher_T::EpiParam;
  using Parallel = _Parallel_T<GemmCore>;
  Parallel createParallel(int M = 0, int N = 0, int K = 0, int Batch = 1, int KBlock = 0) {
    Parallel _paral;
    auto cb = utils::CpuBase();
    _paral.update(M, N, K, KBlock, cb.mNumThreads);
    return _paral;
  }
  ActivationType* getActivationPtr() { return &mLauncher.mProA; }
  WeightType* getWeightPtr() { return &mLauncher.mProB; }

  JBLAS_CODE compute(const Arguments& _param, Parallel _paral = Parallel()) {
    auto bptr = dynamic_cast<const prologue::weight_comp::PackedWeightKBlock*>(_param.paramsB[0].packedW);
    if (bptr == nullptr) {
      return JblasInvalidParam;
    }

    auto cb = utils::CpuBase();
    if (_paral.update(_param.M, _param.N, _param.K, cb.mNumThreads)) {
      static bool dbgprint = false;
      if (dbgprint) {
        _paral.print();
        dbgprint = false;
      }
    }
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      launchT(_param, tidx, _paral, cb.mL2Cache);
    }
    return JblasSuccess;
  }

 protected:
  void launchT(const Arguments& _param, int tidx, Parallel& _paral, size_t l2cache) {
    int colidx, rowidx, rowsize, colsize;
    _paral.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
    if (rowsize > 0 && colsize > 0) {
      Config _config{rowidx, colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(),
                     l2cache};
      for (size_t i = 0; i < _param.Batch; i++) {
        mLauncher.launch(_config, {_param.M, _param.N, _param.K, _param.paramA, _param.paramsB[i], _param.paramsC[i],
                                   _param.workspace});
      }
    }
  }

  _Launcher_T mLauncher;
};

template <class _Launcher_T, template <class _T> class _Parallel_T>
class QKVGemmInterfacePackWeightParallelAB {
 public:
  struct Arguments {
    const int M, N, K, Batch;
    const typename _Launcher_T::AParam paramA;
    const typename _Launcher_T::BParam* paramsB;
    const typename _Launcher_T::EpiParam* paramsC;
    void* workspace;
  };
  using Config = typename _Launcher_T::ParallelConfig;
  using ActivationType = typename _Launcher_T::PrologueA;
  using WeightType = typename _Launcher_T::PrologueB;
  using GemmCore = typename _Launcher_T::GemmCore;
  using LArguments = typename _Launcher_T::Param;
  using CParam = typename _Launcher_T::EpiParam;
  using Parallel = _Parallel_T<GemmCore>;
  Parallel createParallel(int M = 0, int N = 0, int K = 0, int Batch = 1, int KBlock = 0) {
    Parallel _paral;
    auto cb = utils::CpuBase();
    _paral.update(M, N, K, KBlock, cb.mNumThreads);
    return _paral;
  }
  ActivationType* getActivationPtr() { return &mLauncher.mProA; }
  WeightType* getWeightPtr() { return &mLauncher.mProB; }

  template <bool _LaunchA, bool _LaunchB>
  JBLAS_CODE compute(const Arguments& _param) {
    auto bptr = dynamic_cast<const prologue::weight_comp::PackedWeightKBlock*>(_param.paramsB[0].packedW);
    if (bptr == nullptr) {
      return JblasInvalidParam;
    }
    auto cb = utils::CpuBase();
    Parallel _paral = Parallel();
    if (_paral.update(_param.M, _param.N, _param.K, cb.mNumThreads)) {
      static bool dbgprint = false;
      if (dbgprint) {
        _paral.print();
        dbgprint = false;
      }
    }
    auto paraA = getActivationPtr()->createParallel(_param.M, _param.K);
    auto paraB = getWeightPtr()->createParallel(_param.M, _param.K);
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      if constexpr (_LaunchA) {
        getActivationPtr()->launch(_param.paramA, tidx, paraA);
      }
      if constexpr (_LaunchB) {
        for (size_t i = 0; i < _param.Batch; i++) {
          getWeightPtr()->launch(_param.paramsB[i], tidx, paraB);
        }
      }
      if constexpr (_LaunchA || _LaunchB) {
#pragma omp barrier
      }
      launchT(_param, tidx, _paral, cb.mL2Cache);
    }
    return JblasSuccess;
  }

 protected:
  void launchT(const Arguments& _param, int tidx, Parallel& _paral, size_t l2cache) {
    int colidx, rowidx, rowsize, colsize;
    _paral.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
    if (rowsize > 0 && colsize > 0) {
      Config _config{rowidx, colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(),
                     l2cache};
      for (size_t i = 0; i < _param.Batch; i++) {
        mLauncher.launch(_config, {_param.M, _param.N, _param.K, _param.paramA, _param.paramsB[i], _param.paramsC[i],
                                   _param.workspace});
      }
    }
  }

  _Launcher_T mLauncher;
};

// compared with BatchGemm, QKV has the same activation matrix.
// This is the reason why making it a new template.
template <class _Launcher_T, template <class _T> class _Parallel_T>
class QKVGemmInterfaceKBlockPackWeight {
 public:
  struct Arguments {
    const int M, N, K, Batch;
    const typename _Launcher_T::AParam paramA;
    const typename _Launcher_T::BParam* paramsB;
    const typename _Launcher_T::EpiParam* paramsC;
    void* workspace;
  };
  using Config = typename _Launcher_T::ParallelConfig;
  using ActivationType = typename _Launcher_T::PrologueA;
  using WeightType = typename _Launcher_T::PrologueB;
  using GemmCore = typename _Launcher_T::GemmCore;
  using LArguments = typename _Launcher_T::Param;
  using CParam = typename _Launcher_T::EpiParam;
  using QuanParam = typename _Launcher_T::QuanAParam;
  using Parallel = _Parallel_T<GemmCore>;
  Parallel createParallel(int M = 0, int N = 0, int K = 0, int Batch = 1, int KBlock = 0) {
    Parallel _paral;
    auto cb = utils::CpuBase();
    _paral.update(M, N, K, KBlock, cb.mNumThreads);
    return _paral;
  }
  ActivationType* getActivationPtr() { return &mLauncher.mProA; }
  WeightType* getWeightPtr() { return &mLauncher.mProB; }

  JBLAS_CODE compute(const Arguments& _param, Parallel _paral = Parallel()) {
    auto bptr = dynamic_cast<const prologue::weight_comp::PackedWeightKBlock*>(_param.paramsB[0].packedW);
    if (bptr == nullptr) {
      return JblasInvalidParam;
    }
    auto cb = utils::CpuBase();
    if (_paral.update(_param.M, _param.N, _param.K, bptr->mBlockSize, cb.mNumThreads)) {
      static bool dbgprint = false;
      if (dbgprint) {
        _paral.print();
        dbgprint = false;
      }
    }
    auto paraA = mLauncher.mProA.createParallel(_param.M, _param.K, bptr->mBlockSize);
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      mLauncher.mProA.launch(_param.paramA, tidx, paraA);
#pragma omp barrier
      launchT(_param, tidx, _paral, cb.mL2Cache);
    }
    return JblasSuccess;
  }

 protected:
  void launchT(const Arguments& _param, int tidx, Parallel& _paral, size_t l2cache) {
    int colidx, rowidx, rowsize, colsize;
    _paral.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
    if (rowsize > 0 && colsize > 0) {
      Config _config{rowidx, colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(),
                     l2cache};
      for (size_t i = 0; i < _param.Batch; i++) {
        mLauncher.launch(_config, {_param.M, _param.N, _param.K, _param.paramA, _param.paramsB[i], _param.paramsC[i],
                                   _param.workspace});
      }
    }
  }

  _Launcher_T mLauncher;
};

}  // namespace transformer
namespace transformer_default {
namespace weight_comp {
namespace avx512_vnni {
static JBLAS_ISA constexpr DefaultISA = JblasAVX512_VNNI;
using QKVGemmDynamicS4Fp32KBlock = jblas::wrapper::transformer::QKVGemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
        DefaultISA, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
        jblas::prologue::gemm::ActivationF32U8KBlockQuantize,
        jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32,
        jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;

}  // namespace avx512_vnni
namespace amx_int8 {
static JBLAS_ISA constexpr DefaultISA = JblasAMX_INT8;
using QKVGemmDynamicS4Fp32KBlock = jblas::wrapper::transformer::QKVGemmInterfaceKBlockPackWeight<
    jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
        DefaultISA, jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
        jblas::prologue::gemm::ActivationF32S8KBlockQuantize,
        jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32,
        jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
}  // namespace amx_int8
namespace avx512f {
static JBLAS_ISA constexpr DefaultISA = JblasAVX512F;
using QKVGemmS4Fp32Kblock = jblas::wrapper::transformer::QKVGemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512F,
        jblas::prologue::gemm::ActivationBase,  // activation fp32->bf16
        jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32,
        jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemm>;
}  // namespace avx512f
namespace amx_bf16 {
static JBLAS_ISA constexpr DefaultISA = JblasAMX_BF16;
using QKVGemmS4Fp32Kblock = jblas::wrapper::transformer::QKVGemmInterfacePackWeight<
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
        DefaultISA, jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16,
        jblas::prologue::gemm::ActivationConverterFp32,  // activation fp32->bf16
        jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32,
        jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
    jblas::utils::parallel::Parallel2DGemm>;
}  // namespace amx_bf16
}  // namespace weight_comp
}  // namespace transformer_default
}  // namespace wrapper
}  // namespace jblas
