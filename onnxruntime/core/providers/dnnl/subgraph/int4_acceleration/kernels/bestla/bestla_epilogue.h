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
#include <tuple>

#include "bestla.h"
#include "bestla_jit.h"
#include "bestla_utils.h"
#include "kernel_wrapper.h"

namespace bestla {
namespace epilogue {
namespace gemm {

template <typename DT>
struct ParamAccumulatorWriteBack {
  DT* C;
  int ldc;
  void* elt_const_v;
};

template <BTLA_ISA ISA_T, typename _SRC_T, typename _DST_T>
class AccumulatorWriteBack {
 public:
  using SType = _SRC_T;
  using DType = _DST_T;
  using Param = ParamAccumulatorWriteBack<DType>;

  BTLA_CODE forward(const _SRC_T* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                    const int N, const Param& _param, void* tmpcache, size_t cachesize) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    return kernel::wrapper::Memcpy2D::template forward<ISA_T, SType, DType>(cacheptr, cptr, M, N, cachestep, _param.ldc,
                                                                            _param.elt_const_v);
  }
};

template <BTLA_ISA ISA_T, typename _SRC_T, typename _DST_T, BTLA_ELTWISEOP _OP>
class CustomAccumulatorWriteBackWithEltop {
 public:
  using Param = ParamAccumulatorWriteBack<_DST_T>;
  BTLA_CODE forward(const _SRC_T* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                    const int N, const Param& _param, void* tmpcache, size_t cachesize) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    if constexpr (std::is_same<_SRC_T, float>::value && std::is_same<_DST_T, float>::value) {
      return kernel::wrapper::Memcpy2D::template forward1<ISA_T, float, float, _OP>(cacheptr, cptr, M, N, cachestep,
                                                                                    _param.ldc, _param.elt_const_v);
    } else {
      assert(false);
    }
  }
};
template <BTLA_ISA ISA_T>
using AccumulatorWriteBackFp32 = AccumulatorWriteBack<ISA_T, float, float>;
template <BTLA_ISA ISA_T>
using AccumulatorWriteBackInt32 = AccumulatorWriteBack<ISA_T, int, int>;
template <BTLA_ISA ISA_T>
using AccumulatorWriteBackBf16 = AccumulatorWriteBack<ISA_T, utils::bf16, utils::bf16>;
template <BTLA_ISA ISA_T>
using AccumulatorWriteBackFp16 = AccumulatorWriteBack<ISA_T, utils::fp16, utils::fp16>;
template <BTLA_ISA ISA_T>
using AccumulatorWriteBackBf16Fp32 = AccumulatorWriteBack<ISA_T, utils::bf16, float>;
template <BTLA_ISA ISA_T>
using AccumulatorWriteBackFp16Fp32 = AccumulatorWriteBack<ISA_T, utils::fp16, float>;
template <BTLA_ISA ISA_T>
using AccumulatorWriteBackFp32Bf16 = AccumulatorWriteBack<ISA_T, float, utils::bf16>;

template <BTLA_ISA ISA_T>
using AccumulatorWriteBackWithGeluFp32 = CustomAccumulatorWriteBackWithEltop<ISA_T, float, float, BTLA_ELTWISEOP::GELU>;

template <BTLA_ISA ISA_T>
using AccumulatorWriteBackWithSwishFp32 =
    CustomAccumulatorWriteBackWithEltop<ISA_T, float, float, BTLA_ELTWISEOP::SWISH>;

template <typename DT>
struct ParamAlphaBetaProcess {
  DT *C, *D;
  int ldc, ldd;
  float alpha, beta;
};
template <BTLA_ISA ISA_T>
class AlphaBetaProcessFp32 {
 public:
  using Param = ParamAlphaBetaProcess<float>;

  BTLA_CODE forward(const float* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                    const int N, const Param& _param, void* tmpcache, size_t cachesize) {
    auto DOffset = M_offset * _param.ldd + N_offset;
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    auto dptr = _param.D + DOffset;
    return kernel::wrapper::AlphaBetaF32F32::template forward<ISA_T>(_param.alpha, cacheptr, cachestep, _param.beta,
                                                                     dptr, _param.ldd, cptr, _param.ldc, M, N);
  }
};

struct ParamCompFp32BlockEpilogue {
  void* scales;
  BTLA_DTYPE scaledtype;
  int ldsb;
  int8_t* zps = nullptr;
  float* reduce = nullptr;
  int ldra;
};
template <BTLA_ISA ISA_T>
class CompFp32BlockEpilogue {
 public:
  using Param = ParamCompFp32BlockEpilogue;
  BTLA_CODE forward(const float* srcptr, float* dstptr, const int cachestep, const int M_offset, const int N_offset,
                    const int K_offset, const int M, const int N, const Param& _param, void* tmpcache,
                    size_t cachesize) {
    auto ret = BTLA_CODE::NotSupport;
    if (_param.scaledtype == BTLA_DTYPE::F32) {
      ret = kernel::wrapper::CompFp32BlockScale::template forward<ISA_T>(
          reinterpret_cast<float*>(_param.scales) + K_offset * _param.ldsb + N_offset, srcptr, cachestep, dstptr,
          cachestep, M, N);
      assert(ret == BTLA_CODE::Success);
      if (_param.zps != nullptr) {
        ret = kernel::wrapper::RemoveZeroPointBias::forward_wei<ISA_T>(
            dstptr, cachestep, M, N, _param.zps + K_offset * _param.ldsb + N_offset,
            reinterpret_cast<float*>(_param.scales) + K_offset * _param.ldsb + N_offset, _param.ldra,
            _param.reduce + M_offset * _param.ldra + K_offset);
      }
      assert(ret == BTLA_CODE::Success);
      return ret;
    } else if (_param.scaledtype == BTLA_DTYPE::BF16) {
      ret = kernel::wrapper::CompFp32BlockScale::template forward<ISA_T>(
          reinterpret_cast<utils::bf16*>(_param.scales) + K_offset * _param.ldsb + N_offset, srcptr, cachestep, dstptr,
          cachestep, M, N);
      if (_param.zps != nullptr) {
        assert(0);
      }
      assert(ret == BTLA_CODE::Success);
      return ret;
    } else if (_param.scaledtype == BTLA_DTYPE::F8_E8M0) {
      ret = kernel::wrapper::CompFp32BlockScale::template forward<ISA_T>(
          reinterpret_cast<utils::f8*>(_param.scales) + K_offset * _param.ldsb + N_offset, srcptr, cachestep, dstptr,
          cachestep, M, N);
      if (_param.zps != nullptr) {
        assert(0);
      }
    }
    return BTLA_CODE::NotSupport;
  }
};

struct ParamDequantInt32ToFp32 {
  float* C;
  int ldc;
  int ldsa;
  float* scalesA;
  float* scalesB;
};
template <BTLA_ISA ISA_T>
class DequantInt32ToFp32 {
 public:
  using Param = ParamDequantInt32ToFp32;
  BTLA_CODE forward(const int32_t* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                    const int N, const Param& _param, void* tmpcache, size_t cachesize) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    return kernel::wrapper::DequanS32Fp32::template forward<ISA_T>(cacheptr, cachestep, cptr, _param.ldc, M, N,
                                                                   _param.scalesA + M_offset * _param.ldsa, _param.ldsa,
                                                                   _param.scalesB + N_offset);
  }
};

struct ParamCompInt8BlockEpilogue {
  void* scalesB;
  BTLA_DTYPE scaleBdtype;
  int ldsb;
  float* scalesA;
  int ldsa;
  // optional if A asym
  uint8_t* zpA = nullptr;
  void* reduceB = nullptr;
  BTLA_DTYPE reduceBdtype = BTLA_DTYPE::F32;
  // optional if B asym
  int8_t* zpB = nullptr;
  float* reduceA = nullptr;
  int K = 1;
};
template <BTLA_ISA ISA_T>
class CompInt8BlockEpilogue {
 public:
  using Param = ParamCompInt8BlockEpilogue;
  BTLA_CODE forward(const int32_t* srcptr, float* dstptr, const int cachestep, const int M_offset, const int N_offset,
                    const int K_offset, const int M, const int N, const Param& _param, void* tmpcache,
                    size_t cachesize) {
    BTLA_CODE ret = BTLA_CODE::NotSupport;
    float* scab = nullptr;
    size_t ScaleBTmpSize = N * sizeof(float);
    size_t ReduceBTmpSize = N * sizeof(float);
    assert(cachesize >= (ScaleBTmpSize + ReduceBTmpSize));
    if (_param.scaleBdtype == BTLA_DTYPE::BF16) {
      auto scache = reinterpret_cast<float*>(tmpcache);
      ret = kernel::wrapper::Memcpy2DBf16CvtFp32::template forward<ISA_T>(
          reinterpret_cast<utils::bf16*>(_param.scalesB) + N_offset + K_offset * _param.ldsb, scache, 1, N, N, N,
          false);
      assert(ret == BTLA_CODE::Success);
      scab = scache;
    } else if (_param.scaleBdtype == BTLA_DTYPE::F32) {
      scab = reinterpret_cast<float*>(_param.scalesB) + N_offset + K_offset * _param.ldsb;
    }
    float* redb = nullptr;
    if (_param.reduceB) {
      if (_param.reduceBdtype == BTLA_DTYPE::BF16) {
        auto rcache = reinterpret_cast<float*>(reinterpret_cast<char*>(tmpcache) + ScaleBTmpSize);
        ret = kernel::wrapper::Memcpy2DBf16CvtFp32::template forward<ISA_T>(
            reinterpret_cast<utils::bf16*>(_param.reduceB) + N_offset + K_offset * _param.ldsb, rcache, 1, N, N, N,
            false);
        assert(ret == BTLA_CODE::Success);
        redb = rcache;
      } else if (_param.reduceBdtype == BTLA_DTYPE::F32) {
        redb = reinterpret_cast<float*>(_param.reduceB) + N_offset + K_offset * _param.ldsb;
      }
    }
    ret = kernel::wrapper::DequanS32Fp32::template forward<ISA_T>(
        srcptr, cachestep, reinterpret_cast<float*>(const_cast<int32_t*>(srcptr)), cachestep, M, N,
        _param.scalesA + M_offset * _param.ldsa + K_offset, _param.ldsa, scab);
    assert(ret == BTLA_CODE::Success);
    ret = kernel::wrapper::AccumulateFp32::template forward<ISA_T>(reinterpret_cast<const float*>(srcptr), cachestep,
                                                                   dstptr, cachestep, M, N);
    assert(ret == BTLA_CODE::Success);

    if (_param.zpA == nullptr) {
      if (_param.zpB == nullptr) {
        return ret;
      } else {
        ret = kernel::wrapper::RemoveZeroPointBias::template forward_wei<ISA_T>(
            dstptr, cachestep, M, N, _param.zpB + N_offset + K_offset * _param.ldsb, scab, _param.ldsa,
            _param.reduceA + M_offset * _param.ldsa + K_offset);
      }
    } else {
      if (_param.zpB == nullptr) {
        ret = kernel::wrapper::RemoveZeroPointBias::template forward_act<ISA_T>(
            dstptr, cachestep, M, N, _param.zpA + M_offset * _param.ldsa + K_offset,
            _param.scalesA + M_offset * _param.ldsa + K_offset, _param.ldsa, redb);
      } else {
        ret = kernel::wrapper::RemoveZeroPointBias::template forward_both<ISA_T>(
            dstptr, cachestep, M, N, _param.zpA + M_offset * _param.ldsa + K_offset,
            _param.zpB + N_offset + K_offset * _param.ldsb, _param.scalesA + M_offset * _param.ldsa + K_offset, scab,
            _param.ldsa, _param.K, _param.reduceA + M_offset * _param.ldsa + K_offset, redb);
      }
    }
    return ret;
  }
};

struct ParamZpDequantInt32ToFp32 {
  // necessary
  float* C;
  int ldc;
  int ldsa;
  float* scalesA;
  float* scalesB;
  // optional if A asym
  uint8_t* zpA = nullptr;
  float* reduceB = nullptr;
  // optional if B asym
  int8_t* zpB = nullptr;
  float* reduceA = nullptr;
  int K = 1;
};
template <BTLA_ISA ISA_T>
class ZpDequantInt32ToFp32 {
 public:
  using Param = ParamZpDequantInt32ToFp32;
  BTLA_CODE forward(const int32_t* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                    const int N, const Param& _param, void* tmpcache, size_t cachesize) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    auto ret = kernel::wrapper::DequanS32Fp32::template forward<ISA_T>(cacheptr, cachestep, cptr, _param.ldc, M, N,
                                                                       _param.scalesA + M_offset * _param.ldsa,
                                                                       _param.ldsa, _param.scalesB + N_offset);
    if (ret != BTLA_CODE::Success) {
      return ret;
    }
    if (_param.zpA == nullptr && _param.zpB == nullptr) {
      return ret;
    } else if (_param.zpA != nullptr && _param.zpB == nullptr) {
      ret = kernel::wrapper::RemoveZeroPointBias::template forward_act<ISA_T>(
          cptr, _param.ldc, M, N, _param.zpA + M_offset * _param.ldsa, _param.scalesA + M_offset * _param.ldsa,
          _param.ldsa, _param.reduceB + N_offset);
    } else if (_param.zpA == nullptr && _param.zpB != nullptr) {
      ret = kernel::wrapper::RemoveZeroPointBias::template forward_wei<ISA_T>(
          cptr, _param.ldc, M, N, _param.zpB + N_offset, _param.scalesB + N_offset, _param.ldsa,
          _param.reduceA + M_offset * _param.ldsa);
    } else {
      ret = kernel::wrapper::RemoveZeroPointBias::template forward_both<ISA_T>(
          cptr, _param.ldc, M, N, _param.zpA + M_offset * _param.ldsa, _param.zpB + N_offset,
          _param.scalesA + M_offset * _param.ldsa, _param.scalesB + N_offset, _param.ldsa, _param.K,
          _param.reduceA + M_offset * _param.ldsa, _param.reduceB + N_offset);
    }
    return ret;
  }
};

struct ParamAlphaBetaProcessS32U8 {
  uint8_t* C;
  int ldc;
  float alpha;
  float scaleAcc, scaleC;
  int zpC;
};
template <BTLA_ISA ISA_T>
class AlphaBetaProcessS32U8 {
 public:
  using Param = ParamAlphaBetaProcessS32U8;
  BTLA_CODE forward(const int32_t* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                    const int N, const Param& _param, void* tmpcache, size_t cachesize) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    return kernel::wrapper::QuanOutS32U32::template forward<ISA_T>(_param.alpha, cacheptr, cachestep, cptr, _param.ldc,
                                                                   M, N, _param.scaleAcc, _param.scaleC, _param.zpC);
  }
};

}  // namespace gemm
}  // namespace epilogue
}  // namespace bestla
