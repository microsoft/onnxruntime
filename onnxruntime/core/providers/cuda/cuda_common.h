// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/providers/cuda/cuda_pch.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/providers/cuda/shared_inc/fast_divmod.h"
#include "core/util/math.h"

namespace onnxruntime {
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

#ifdef USE_CUSPARSELT
#define CUSPARSELT_RETURN_IF_ERROR(expr)           \
  ORT_RETURN_IF_ERROR(CUSPARSELT_CALL(expr)        \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUSPARSE error executing ", #expr))
#endif


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

// Default float
template<typename T>
struct ToCudaTypeEnum;

template<>
struct ToCudaTypeEnum<double> {
  static constexpr cudaDataType type = CUDA_R_64F;
#ifdef USE_CUSPARSELT
  static constexpr cusparseComputeType at_least_precision = CUSPARSE_COMPUTE_16F;
#endif
};

template <>
struct ToCudaTypeEnum<float> {
  static constexpr cudaDataType type = CUDA_R_32F;
#ifdef USE_CUSPARSELT
  static constexpr cusparseComputeType at_least_precision = CUSPARSE_COMPUTE_16F;
#endif
};

template <>
struct ToCudaTypeEnum<MLFloat16> {
  static constexpr cudaDataType type = CUDA_R_16U;
#ifdef USE_CUSPARSELT
  static constexpr cusparseComputeType at_least_precision = CUSPARSE_COMPUTE_16F;
#endif
};

template <>
struct ToCudaTypeEnum<BFloat16> {
  static constexpr cudaDataType type = CUDA_R_16BF;
#ifdef USE_CUSPARSELT
  static constexpr cusparseComputeType at_least_precision = CUSPARSE_COMPUTE_16F;
#endif
};

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
#if defined(CUDA_VERSION) && CUDA_VERSION < 11000    
    enable_ = (mode == CUBLAS_TENSOR_OP_MATH ? prop.major >= 7 : true );
#else
    enable_ = (mode == CUBLAS_TF32_TENSOR_OP_MATH ? prop.major >= 8 : true);
#endif
    
    if (enable_) {
      cublasGetMathMode(handle, &mode_);
      enable_ = (mode_ != mode);
      if (enable_) {
        cublasSetMathMode(handle, mode);
      }
    }
  }

  ~CublasMathModeSetter() {
    if (enable_) {
      cublasSetMathMode(handle_, mode_);
    }
  }

 private:
  const cudaDeviceProp& prop_;
  cublasHandle_t handle_;
  cublasMath_t mode_;
  bool enable_;
};

// Cublas Gemm options for half data type
class HalfGemmOptions {
 public:
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  cublasMath_t GetMathMode() const {
    if (pedantic_) {
      return CUBLAS_PEDANTIC_MATH;
    }
    return disallow_reduced_precision_reduction_ && !compute_16f_ ? CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION : CUBLAS_DEFAULT_MATH;
  }

  cublasComputeType_t GetComputeType() const {
    if (compute_16f_) {
      return pedantic_ ? CUBLAS_COMPUTE_16F_PEDANTIC : CUBLAS_COMPUTE_16F;
    } else {
      return pedantic_ ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
    }
  }
#else
  cublasMath_t GetMathMode() const {
    // CublasMathModeSetter will check whether device has tensor cores later.
    return CUBLAS_TENSOR_OP_MATH;
  }

  cudaDataType GetComputeType() const {
    return compute_16f_ ? CUDA_R_16F : CUDA_R_32F;
  }
#endif

  static const HalfGemmOptions* GetInstance();

  bool IsCompute16F() const { return compute_16f_; }

  void Initialize(int value);

 private:
  // Default is FP32. Aggregate in FP16 might be faster but the cost is loss in precision.
  bool compute_16f_{false};

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  // Avoid intermediate overflows in accumulation. When compute type is FP32, it will not use FP16 in reduction.
  bool disallow_reduced_precision_reduction_{false};

  // For numerical robustness studies only. It is much slower.
  bool pedantic_{false};
#endif

  bool initialized_{false};

  static HalfGemmOptions instance;
};

}  // namespace cuda
}  // namespace onnxruntime
