// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef AAA

#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {
namespace test {

void CudaGemmOptions_TestDefaultOptions() {
  HalfGemmOptions gemm_options;
  ORT_ENFORCE(!gemm_options.IsCompute16F(), "Actual: ", gemm_options.IsCompute16F());
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  ORT_ENFORCE(gemm_options.GetMathMode() == CUBLAS_DEFAULT_MATH, "Actual:", gemm_options.GetMathMode());
  ORT_ENFORCE(gemm_options.GetComputeType() == CUBLAS_COMPUTE_32F, "Actual:", gemm_options.GetComputeType());
#else
  ORT_ENFORCE(gemm_options.GetMathMode() == CUBLAS_TENSOR_OP_MATH, "Actual:", gemm_options.GetMathMode());
  ORT_ENFORCE(gemm_options.GetComputeType() == CUDA_R_32F, "Actual:", gemm_options.GetComputeType());
#endif
}

void CudaGemmOptions_TestCompute16F() {
  HalfGemmOptions gemm_options;
  gemm_options.Initialize(1);
  ORT_ENFORCE(gemm_options.IsCompute16F(), "Actual: ", gemm_options.IsCompute16F());
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  ORT_ENFORCE(gemm_options.GetMathMode() == CUBLAS_DEFAULT_MATH, "Actual:", gemm_options.GetMathMode());
  ORT_ENFORCE(gemm_options.GetComputeType() == CUBLAS_COMPUTE_16F, "Actual:", gemm_options.GetComputeType());
#else
  ORT_ENFORCE(gemm_options.GetMathMode() == CUBLAS_TENSOR_OP_MATH, "Actual:", gemm_options.GetMathMode());
  ORT_ENFORCE(gemm_options.GetComputeType() == CUDA_R_16F, "Actual:", gemm_options.GetComputeType());
#endif
}

void CudaGemmOptions_NoReducedPrecision() {
  HalfGemmOptions gemm_options;
  gemm_options.Initialize(2);
  ORT_ENFORCE(!gemm_options.IsCompute16F(), "Actual: ", gemm_options.IsCompute16F());
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  ORT_ENFORCE(gemm_options.GetMathMode() == CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION, "Actual:", gemm_options.GetMathMode());
  ORT_ENFORCE(gemm_options.GetComputeType() == CUBLAS_COMPUTE_32F, "Actual:", gemm_options.GetComputeType());
#else
  ORT_ENFORCE(gemm_options.GetMathMode() == CUBLAS_TENSOR_OP_MATH, "Actual:", gemm_options.GetMathMode());
  ORT_ENFORCE(gemm_options.GetComputeType() == CUDA_R_32F, "Actual:", gemm_options.GetComputeType());
#endif
}

void CudaGemmOptions_Pedantic() {
  HalfGemmOptions gemm_options;
  gemm_options.Initialize(4);
  ORT_ENFORCE(!gemm_options.IsCompute16F(), "Actual: ", gemm_options.IsCompute16F());
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  ORT_ENFORCE(gemm_options.GetMathMode() == CUBLAS_PEDANTIC_MATH, "Actual:", gemm_options.GetMathMode());
  ORT_ENFORCE(gemm_options.GetComputeType() == CUBLAS_COMPUTE_32F_PEDANTIC, "Actual:", gemm_options.GetComputeType());
#else
  ORT_ENFORCE(gemm_options.GetMathMode() == CUBLAS_TENSOR_OP_MATH, "Actual:", gemm_options.GetMathMode());
  ORT_ENFORCE(gemm_options.GetComputeType() == CUDA_R_32F, "Actual:", gemm_options.GetComputeType());
#endif
}

void CudaGemmOptions_Compute16F_Pedantic() {
  HalfGemmOptions gemm_options;
  gemm_options.Initialize(5);
  ORT_ENFORCE(gemm_options.IsCompute16F(), "Actual: ", gemm_options.IsCompute16F());
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  ORT_ENFORCE(gemm_options.GetMathMode() == CUBLAS_PEDANTIC_MATH, "Actual:", gemm_options.GetMathMode());
  ORT_ENFORCE(gemm_options.GetComputeType() == CUBLAS_COMPUTE_16F_PEDANTIC, "Actual:", gemm_options.GetComputeType());
#else
  ORT_ENFORCE(gemm_options.GetMathMode() == CUBLAS_TENSOR_OP_MATH, "Actual:", gemm_options.GetMathMode());
  ORT_ENFORCE(gemm_options.GetComputeType() == CUDA_R_16F, "Actual:", gemm_options.GetComputeType());
#endif
}

void CudaGemmOptions_Compute16F_NoReducedPrecision() {
  HalfGemmOptions gemm_options;
  gemm_options.Initialize(3);
  ORT_ENFORCE(gemm_options.IsCompute16F(), "Actual: ", gemm_options.IsCompute16F());
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  ORT_ENFORCE(gemm_options.GetMathMode() == CUBLAS_DEFAULT_MATH, "Actual:", gemm_options.GetMathMode());
  ORT_ENFORCE(gemm_options.GetComputeType() == CUBLAS_COMPUTE_16F, "Actual:", gemm_options.GetComputeType());
#else
  ORT_ENFORCE(gemm_options.GetMathMode() == CUBLAS_TENSOR_OP_MATH, "Actual:", gemm_options.GetMathMode());
  ORT_ENFORCE(gemm_options.GetComputeType() == CUDA_R_16F, "Actual:", gemm_options.GetComputeType());
#endif
}

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime

#endif
