// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/common/status.h"
#include "core/framework/float16.h"
#include "core/providers/cuda/cuda_pch.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/providers/cuda/shared_inc/fast_divmod.h"
#include "gsl/gsl"

// Can't include "core/util/math.h" in a provider, so this is the part we need for cuda:
namespace onnxruntime {
namespace math {
uint16_t floatToHalf(float f);
}

namespace cuda {

#define CUDA_RETURN_IF_ERROR(expr)               \
  ORT_RETURN_IF_ERROR(CUDA_CALL(expr)            \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDA error executing ", #expr))

#define CUBLAS_RETURN_IF_ERROR(expr)             \
  ORT_RETURN_IF_ERROR(CUBLAS_CALL(expr)          \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUBLAS error executing ", #expr))

#define CUSPARSE_RETURN_IF_ERROR(expr)           \
  ORT_RETURN_IF_ERROR(CUSPARSE_CALL(expr)        \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUSPARSE error executing ", #expr))

#define CURAND_RETURN_IF_ERROR(expr)             \
  ORT_RETURN_IF_ERROR(CURAND_CALL(expr)          \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CURAND error executing ", #expr))

#define CUDNN_RETURN_IF_ERROR(expr)              \
  ORT_RETURN_IF_ERROR(CUDNN_CALL(expr)           \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDNN error executing ", #expr))

#define CUDNN2_RETURN_IF_ERROR(expr, m)          \
  ORT_RETURN_IF_ERROR(CUDNN_CALL2(expr, m)       \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDNN2 error executing ", #expr))

#define CUFFT_RETURN_IF_ERROR(expr)              \
  ORT_RETURN_IF_ERROR(CUFFT_CALL(expr)           \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUFFT error executing ", #expr))

// Type mapping for MLFloat16 to half
template <typename T>
class ToCudaType {
 public:
  typedef T MappedType;
  static MappedType FromFloat(float f) {
    return static_cast<T>(f);
  }
};

template <>
class ToCudaType<MLFloat16> {
 public:
  typedef half MappedType;
  static MappedType FromFloat(float f) {
    uint16_t h = math::floatToHalf(f);
    return *reinterpret_cast<MappedType*>(&h);
  }
};

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
template <>
class ToCudaType<BFloat16> {
 public:
  typedef nv_bfloat16 MappedType;
  static MappedType FromFloat(float f) {
    uint16_t h = BFloat16(f).val;
    return *reinterpret_cast<MappedType*>(&h);
  }
};
#endif

inline bool CalculateFdmStrides(gsl::span<fast_divmod> p, const std::vector<int64_t>& dims) {
  int stride = 1;
  if (dims.empty() || p.size() < dims.size())
    return false;
  auto rank = p.size();
  for (size_t i = 0; i < rank; i++) {
    p[rank - 1 - i] = fast_divmod(stride);
    if (i < dims.size() - 1) {
      stride *= static_cast<int>(dims[dims.size() - 1 - i]);
    }
  }
  return true;
}

class CublasMathModeSetter {
 public:
  CublasMathModeSetter(const cudaDeviceProp& prop, cublasHandle_t handle, cublasMath_t mode) : prop_(prop), handle_(handle) {
    cublasGetMathMode(handle, &mode_);
#if defined(CUDA_VERSION) && CUDA_VERSION < 11000
    if (prop.major >= 7 && mode == CUBLAS_TENSOR_OP_MATH) {
      cublasSetMathMode(handle, mode);
    }
#endif
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    if (prop.major >= 8 && mode == CUBLAS_TF32_TENSOR_OP_MATH) {
      cublasSetMathMode(handle, mode);
    }
#endif
  }

  ~CublasMathModeSetter() {
    cublasSetMathMode(handle_, mode_);
  }

 private:
  const cudaDeviceProp& prop_;
  cublasHandle_t handle_;
  cublasMath_t mode_;
};

}  // namespace cuda
}  // namespace onnxruntime
