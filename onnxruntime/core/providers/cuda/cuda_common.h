// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// The following three lines were copied from ABSL
// cutlass needs them, because cutlass uses "and"/"or" keywords
#ifdef __cplusplus
#include <ciso646>
#endif

#include "core/providers/shared_library/provider_api.h"
#include "core/common/status.h"
#include "core/framework/float8.h"
#include "core/framework/float16.h"
#include "core/providers/cuda/cuda_pch.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/providers/cuda/shared_inc/fast_divmod.h"
#include "core/common/gsl.h"

namespace onnxruntime {
namespace cuda {

#define CUDA_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(CUDA_CALL(expr))
#define CUBLAS_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(CUBLAS_CALL(expr))
#define CUSPARSE_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(CUSPARSE_CALL(expr))
#define CURAND_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(CURAND_CALL(expr))
#define CUDNN_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(CUDNN_CALL(expr))
#define CUDNN2_RETURN_IF_ERROR(expr, m) ORT_RETURN_IF_ERROR(CUDNN_CALL2(expr, m))
#define CUFFT_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(CUFFT_CALL(expr))

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

template <>
class ToCudaType<BFloat16> {
 public:
  typedef BFloat16 MappedType;
  static MappedType FromFloat(float f) {
    return MappedType(f);
  }
};

#if !defined(DISABLE_FLOAT8_TYPES)

template <>
class ToCudaType<Float8E4M3FN> {
 public:
  typedef Float8E4M3FN MappedType;
  static MappedType FromFloat(float f) {
    return MappedType(f);
  }
};

template <>
class ToCudaType<Float8E5M2> {
 public:
  typedef Float8E5M2 MappedType;
  static MappedType FromFloat(float f) {
    return MappedType(f);
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
  CublasMathModeSetter(const cudaDeviceProp& prop, cublasHandle_t handle, cublasMath_t mode) : handle_(handle) {
    enable_ = (mode == CUBLAS_TF32_TENSOR_OP_MATH ? prop.major >= 8 : true);

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
  cublasHandle_t handle_;
  cublasMath_t mode_ = CUBLAS_DEFAULT_MATH;
  bool enable_;
};

// Cublas Gemm options for half data type
class HalfGemmOptions {
 public:
#if defined(USE_CUDA)
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

  void Initialize(int value) {
    compute_16f_ = (value & 0x01) > 0;
#if defined(USE_CUDA)
    disallow_reduced_precision_reduction_ = (value & 0x02) > 0;
    pedantic_ = (value & 0x04) > 0;
    LOGS_DEFAULT(INFO) << "ORT_CUDA_GEMM_OPTIONS: compute_16f=" << instance.compute_16f_
                       << " disallow_reduced_precision_reduction=" << instance.disallow_reduced_precision_reduction_
                       << " pedantic=" << instance.pedantic_;
#else
    LOGS_DEFAULT(INFO) << "ORT_CUDA_GEMM_OPTIONS: compute_16f=" << instance.compute_16f_;
#endif
    initialized_ = true;
  }

 private:
  // Default is FP32. Aggregate in FP16 might be faster but the cost is loss in precision.
  bool compute_16f_{false};

#if defined(USE_CUDA)
  // Avoid intermediate overflows in accumulation. When compute type is FP32, it will not use FP16 in reduction.
  bool disallow_reduced_precision_reduction_{false};

  // For numerical robustness studies only. It is much slower.
  bool pedantic_{false};
#endif

  bool initialized_{false};

  static HalfGemmOptions instance;
};

const char* cublasGetErrorEnum(cublasStatus_t error);

const char* CudaDataTypeToString(cudaDataType_t dt);

const char* CublasComputeTypeToString(cublasComputeType_t ct);

cudaDataType_t ToCudaDataType(int32_t element_type);

}  // namespace cuda
}  // namespace onnxruntime
