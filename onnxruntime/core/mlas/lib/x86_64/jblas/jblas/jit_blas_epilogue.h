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

#include "jit_base.h"
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
                     const int N, const Param& _param, void* tmpcache, size_t cachesize, Eltops... ops) {
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
                     const int N, const Param& _param, void* tmpcache, size_t cachesize) {
    auto DOffset = M_offset * _param.ldd + N_offset;
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    auto dptr = _param.D + DOffset;
    return kernel::wrapper::AlphaBetaF32F32::template forward<ISA_T>(_param.alpha, cacheptr, cachestep, _param.beta,
                                                                     dptr, _param.ldd, cptr, _param.ldc, M, N);
  }
};

template <JBLAS_ISA ISA_T>
class CompFp32BlockEpilogue {
 public:
  struct Param {
    void* scales;
    JBLAS_DTYPE scaledtype;
    int ldsb;
    int8_t* zps = nullptr;
    float* reduce = nullptr;
    int ldra;
  };
  JBLAS_CODE forward(const float* srcptr, float* dstptr, const int cachestep, const int M_offset, const int N_offset,
                     const int K_offset, const int M, const int N, const Param& _param, void* tmpcache,
                     size_t cachesize) {
    auto ret = JblasNotSupport;
    if (_param.scaledtype == JBLAS_DTYPE::F32) {
      ret = kernel::wrapper::CompFp32BlockScale::template forward<ISA_T>(
          (float*)_param.scales + K_offset * _param.ldsb + N_offset, srcptr, cachestep, dstptr, cachestep, M, N);
      assert(ret == JblasSuccess);
      if (_param.zps != nullptr) {
        ret = kernel::wrapper::RemoveZeroPointBias::forward<ISA_T>(
            dstptr, cachestep, M, N, _param.zps + K_offset * _param.ldsb + N_offset,
            (float*)_param.scales + K_offset * _param.ldsb + N_offset, _param.ldra,
            _param.reduce + M_offset * _param.ldra + K_offset);
      }
      assert(ret == JblasSuccess);
      return ret;
    } else if (_param.scaledtype == JBLAS_DTYPE::BF16) {
      ret = kernel::wrapper::CompFp32BlockScale::template forward<ISA_T>(
          (utils::bf16*)_param.scales + K_offset * _param.ldsb + N_offset, srcptr, cachestep, dstptr, cachestep, M, N);
      if (_param.zps != nullptr) {
        assert(0);
      }
      assert(ret == JblasSuccess);
      return ret;
    }
    return JblasNotSupport;
  }
};

template <JBLAS_ISA ISA_T>
class DequantInt32ToFp32 {
 public:
  struct Param {
    float* C;
    int ldc;
    int ldsa;
    float* scalesA;
    float* scalesB;
  };
  JBLAS_CODE forward(const int32_t* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param, void* tmpcache, size_t cachesize) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    return kernel::wrapper::DequanS32Fp32::template forward<ISA_T>(cacheptr, cachestep, cptr, _param.ldc, M, N,
                                                                   _param.scalesA + M_offset * _param.ldsa, _param.ldsa,
                                                                   _param.scalesB + N_offset);
  }
};

template <JBLAS_ISA ISA_T>
class CompInt8BlockEpilogue {
 public:
  struct Param {
    void* scalesB;
    JBLAS_DTYPE scaleBdtype;
    int ldsb;
    float* scalesA;
    int ldsa;
    // optional if A asym
    uint8_t* zpA = nullptr;
    float* reduceB = nullptr;
    // optional if B asym
    int8_t* zpB = nullptr;
    float* reduceA = nullptr;
    int K = 1;
  };
  JBLAS_CODE forward(const int32_t* srcptr, float* dstptr, const int cachestep, const int M_offset, const int N_offset,
                     const int K_offset, const int M, const int N, const Param& _param, void* tmpcache,
                     size_t cachesize) {
    JBLAS_CODE ret = JblasNotSupport;
    if (_param.scaleBdtype == JBLAS_DTYPE::F32) {
      ret = kernel::wrapper::DequanS32Fp32::template forward<ISA_T>(
          srcptr, cachestep, (float*)srcptr, cachestep, M, N, _param.scalesA + M_offset * _param.ldsa + K_offset,
          _param.ldsa, (float*)_param.scalesB + N_offset + K_offset * _param.ldsb);
      assert(ret == JblasSuccess);
    } else if (_param.scaleBdtype == JBLAS_DTYPE::BF16) {
      ret = kernel::wrapper::DequanS32Fp32::template forward<ISA_T>(
          srcptr, cachestep, (float*)srcptr, cachestep, M, N, _param.scalesA + M_offset * _param.ldsa + K_offset,
          _param.ldsa, (utils::bf16*)_param.scalesB + N_offset + K_offset * _param.ldsb);
      assert(ret == JblasSuccess);
    }
    ret = kernel::wrapper::AccumulateFp32::template forward<ISA_T>((float*)srcptr, cachestep, dstptr, cachestep, M, N);
    if (ret != JblasSuccess) {
      assert(0);
      return ret;
    }

    if (_param.zpA == nullptr && _param.zpB == nullptr) {
      return ret;
    } else if (_param.zpA != nullptr && _param.zpB == nullptr) {
      ret = kernel::wrapper::RemoveZeroPointBias::template forward<ISA_T>(
          dstptr, cachestep, M, N, _param.zpA + M_offset * _param.ldsa + K_offset,
          _param.scalesA + M_offset * _param.ldsa + K_offset, _param.ldsa,
          _param.reduceB + N_offset + K_offset * _param.ldsb);

    } else if (_param.zpA == nullptr && _param.zpB != nullptr) {
      if (_param.scaleBdtype == JBLAS_DTYPE::F32) {
        ret = kernel::wrapper::RemoveZeroPointBias::template forward<ISA_T>(
            dstptr, cachestep, M, N, _param.zpB + N_offset + K_offset * _param.ldsb,
            (float*)_param.scalesB + N_offset + K_offset * _param.ldsb, _param.ldsa,
            _param.reduceA + M_offset * _param.ldsa + K_offset);
      }

    } else {
      if (_param.scaleBdtype == JBLAS_DTYPE::F32) {
        ret = kernel::wrapper::RemoveZeroPointBias::template forward<ISA_T>(
            dstptr, cachestep, M, N, _param.zpA + M_offset * _param.ldsa + K_offset,
            _param.zpB + N_offset + K_offset * _param.ldsb, _param.scalesA + M_offset * _param.ldsa + K_offset,
            (float*)_param.scalesB + N_offset + K_offset * _param.ldsb, _param.ldsa, _param.K,
            _param.reduceA + M_offset * _param.ldsa + K_offset, _param.reduceB + N_offset + K_offset * _param.ldsb);
      }
    }
    return ret;
  }
};

template <JBLAS_ISA ISA_T>
class ZpDequantInt32ToFp32 {
 public:
  struct Param {
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
  JBLAS_CODE forward(const int32_t* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param, void* tmpcache, size_t cachesize) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    auto ret = kernel::wrapper::DequanS32Fp32::template forward<ISA_T>(cacheptr, cachestep, cptr, _param.ldc, M, N,
                                                                       _param.scalesA + M_offset * _param.ldsa,
                                                                       _param.ldsa, _param.scalesB + N_offset);
    if (ret != JblasSuccess) {
      return ret;
    }
    if (_param.zpA == nullptr && _param.zpB == nullptr) {
      return ret;
    } else if (_param.zpA != nullptr && _param.zpB == nullptr) {
      ret = kernel::wrapper::RemoveZeroPointBias::template forward<ISA_T>(
          cptr, _param.ldc, M, N, _param.zpA + M_offset * _param.ldsa, _param.scalesA + M_offset * _param.ldsa,
          _param.ldsa, _param.reduceB + N_offset);
    } else if (_param.zpA == nullptr && _param.zpB != nullptr) {
      ret = kernel::wrapper::RemoveZeroPointBias::template forward<ISA_T>(cptr, _param.ldc, M, N, _param.zpB + N_offset,
                                                                          _param.scalesB + N_offset, _param.ldsa,
                                                                          _param.reduceA + M_offset * _param.ldsa);
    } else {
      ret = kernel::wrapper::RemoveZeroPointBias::template forward<ISA_T>(
          cptr, _param.ldc, M, N, _param.zpA + M_offset * _param.ldsa, _param.zpB + N_offset,
          _param.scalesA + M_offset * _param.ldsa, _param.scalesB + N_offset, _param.ldsa, _param.K,
          _param.reduceA + M_offset * _param.ldsa, _param.reduceB + N_offset);
    }
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
                     const int N, const Param& _param, void* tmpcache, size_t cachesize) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    return kernel::wrapper::QuanOutS32U32::template forward<ISA_T>(_param.alpha, cacheptr, cachestep, cptr, _param.ldc,
                                                                   M, N, _param.scaleAcc, _param.scaleC, _param.zpC);
  }
};

}  // namespace gemm
}  // namespace epilogue
}  // namespace jblas
