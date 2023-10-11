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

#include "jit_base.hpp"
#include "jit_blas.h"
#include "jit_blas_utils.h"
#include "kernel_wrapper.h"

namespace jblas {
namespace epilogue {
namespace gemm {

template <JBLAS_ISA ISA_T, typename _SRC_T, typename _DST_T>
class AccumulatorWriteBack {
 public:
  using SType = _SRC_T;
  using DType = _DST_T;
  struct Param {
    DType* C;
    int ldc;
    void* elt_const_v;
  };

  template <typename... Eltops>
  JBLAS_CODE forward(const _SRC_T* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param, Eltops... ops) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    bool constexpr Valid = !std::is_same<DType, utils::bf16>::value || std::is_same<SType, float>::value;
    static_assert(Valid, "fp32 to bf16 conversion only.");
    if constexpr (std::is_same<DType, utils::bf16>::value) {
      return kernel::wrapper::Memcpy2DFp32CvtBf16::template forward<ISA_T>(
          (void*)cacheptr, (void*)cptr, M, N, cachestep * sizeof(SType), _param.ldc * sizeof(DType), false);
    } else if constexpr (std::is_same<std::tuple<SType, DType>, std::tuple<utils::fp16, float>>::value) {
      return kernel::wrapper::Memcpy2DFp16CvtFp32::template forward<ISA_T>(
          (void*)cacheptr, (void*)cptr, M, N, cachestep * sizeof(SType), _param.ldc * sizeof(DType), false);
    } else if constexpr (sizeof(SType) == sizeof(DType)) {
      return kernel::wrapper::Memcpy2D::template forward<ISA_T, SType, DType>(cacheptr, cptr, M, N, cachestep,
                                                                              _param.ldc, _param.elt_const_v, ops...);
    } else {
      assert(false);
    }
  }
};

template <JBLAS_ISA ISA_T, typename _SRC_T, typename _DST_T, JBLAS_ELTWISEOP _OP>
class CustomAccumulatorWriteBackWithEltop {
 public:
  struct Param {
    _DST_T* C;
    int ldc;
    void* elt_const_v;
  };
  JBLAS_CODE forward(const _SRC_T* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
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
template <JBLAS_ISA ISA_T>
using AccumulatorWriteBackFp32 = AccumulatorWriteBack<ISA_T, float, float>;
template <JBLAS_ISA ISA_T>
using AccumulatorWriteBackBf16 = AccumulatorWriteBack<ISA_T, utils::bf16, utils::bf16>;
template <JBLAS_ISA ISA_T>
using AccumulatorWriteBackFp16 = AccumulatorWriteBack<ISA_T, utils::fp16, utils::fp16>;
template <JBLAS_ISA ISA_T>
using AccumulatorWriteBackFp16Fp32 = AccumulatorWriteBack<ISA_T, utils::fp16, float>;
template <JBLAS_ISA ISA_T>
using AccumulatorWriteBackFp32Bf16 = AccumulatorWriteBack<ISA_T, float, utils::bf16>;

template <JBLAS_ISA ISA_T>
using AccumulatorWriteBackWithGeluFp32 = CustomAccumulatorWriteBackWithEltop<ISA_T, float, float, GELU>;

template <JBLAS_ISA ISA_T>
using AccumulatorWriteBackWithSwishFp32 = CustomAccumulatorWriteBackWithEltop<ISA_T, float, float, SWISH>;

template <JBLAS_ISA ISA_T>
class AlphaBetaProcessFp32 {
 public:
  struct Param {
    float *C, *D;
    int ldc, ldd;
    float alpha, beta;
  };

  JBLAS_CODE forward(const float* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    auto DOffset = M_offset * _param.ldd + N_offset;
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    auto dptr = _param.D + DOffset;
    return kernel::wrapper::AlphaBetaF32F32::template forward<ISA_T>(_param.alpha, cacheptr, cachestep, _param.beta,
                                                                     dptr, _param.ldd, cptr, _param.ldc, M, N);
  }
};

template <JBLAS_ISA ISA_T>
class DequantInt32ToFp32 {
 public:
  struct Param {
    float* C;
    int ldc;
    float* scalesA;
    int ldsa;
    float* scalesB;
  };
  JBLAS_CODE forward(const int32_t* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    return kernel::wrapper::DequanS32Fp32::template forward<ISA_T>(cacheptr, cachestep, cptr, _param.ldc, M, N,
                                                                   _param.scalesA + M_offset * _param.ldsa, _param.ldsa,
                                                                   _param.scalesB + N_offset);
  }
};

template <JBLAS_ISA ISA_T>
class ZpDequantInt32ToFp32 {
 public:
  struct Param {
    float* C;
    int ldc;
    uint8_t* zpA;
    float* scalesA;
    int ldsa;
    float* reduceB;
    float* scalesB;
  };
  JBLAS_CODE forward(const int32_t* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    auto ret = kernel::wrapper::DequanS32Fp32::template forward<ISA_T>(cacheptr, cachestep, cptr, _param.ldc, M, N,
                                                                       _param.scalesA + M_offset * _param.ldsa,
                                                                       _param.ldsa, _param.scalesB + N_offset);
    if (ret != JblasSuccess) {
      return ret;
    }
    ret = kernel::wrapper::RemoveZeroPointBias::template forward<ISA_T>(
        cptr, _param.ldc, M, N, _param.zpA + M_offset * _param.ldsa, _param.scalesA + M_offset * _param.ldsa,
        _param.ldsa, _param.reduceB + N_offset);
    return ret;
  }
};

template <JBLAS_ISA ISA_T>
class AlphaBetaProcessS32U8 {
 public:
  struct Param {
    uint8_t* C;
    int ldc;
    float alpha;
    float scaleAcc, scaleC;
    int zpC;
  };

  JBLAS_CODE forward(const int32_t* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    return kernel::wrapper::QuanOutS32U32::template forward<ISA_T>(_param.alpha, cacheptr, cachestep, cptr, _param.ldc,
                                                                   M, N, _param.scaleAcc, _param.scaleC, _param.zpC);
  }
};

}  // namespace gemm
}  // namespace epilogue
}  // namespace jblas
