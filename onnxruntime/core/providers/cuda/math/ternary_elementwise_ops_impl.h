// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "orttraining/training_ops/cpu/activation/gelu_computation_mode.h"
// #include "core/providers/cuda/cu_inc/common.cuh"
// #include "orttraining/training_ops/cuda/activation/gelu_grad_impl_common.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "orttraining/training_ops/cuda/activation/activations_grad_impl.h"
// #include "orttraining/training_ops/cuda/activation/gelu_grad_impl_common.cuh"
// #include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T>
__device__ __inline__ T ComputeGeluGradDefault(T dY, T X, gelu_computation_mode::Default) {
  const T kAlpha = T(M_2_SQRTPI) * T(M_SQRT1_2) * T(0.5);
  return dY * (_Normcdf(X) + X * kAlpha * _Exp(-T(0.5) * X * X));
}

// These macros simplifies coding. To add a new op with following steps:
// 1. Add a new entry in TERNARY_OPS() list
// 2. (optional) Define templated single element operator in binary_elementwise_ops_impl.cu
// 3. (optional) Implement specialized single element operator
// 4. Add op kernel class definition in binary_elementwise_ops.h
// 5. Add op kernel registration and compute specialization in binary_elementwise_ops.cc
// ComputeGeluGradScalar(a, b + c, onnxruntime::gelu_computation_mode::Default{})
#define TERNARY_OPS() \
  TERNARY_OP_NAME_EXPR(BiasGeluGrad_dX, ComputeGeluGradDefault(a, b + c, onnxruntime::gelu_computation_mode::Default{}))

// NOTE that cu files are compiled with nvcc and should not refer to any onnxruntime headers
// so struct BinaryElementwisePreparation cannot be used here

#define TERNARY_ELEMENTWISE_IMPL_DECLARATION(name)   \
  template <typename T>                              \
  void Impl_##name(                                  \
      cudaStream_t stream,                           \
      size_t output_rank_or_simple_broadcast,        \
      BroadcastIndexType cond_index_type,            \
      const TArray<int64_t>& cond_padded_strides,    \
      const T* cond_data,                            \
      BroadcastIndexType x_index_type,               \
      const TArray<int64_t>& x_padded_strides,       \
      const T* x_data,                               \
      BroadcastIndexType y_index_type,               \
      const TArray<int64_t>& y_padded_strides,       \
      const T* y_data,                               \
      const TArray<fast_divmod>& fdm_output_strides, \
      T* output_data,                                \
      size_t count)

#define TERNARY_OP_NAME_EXPR(name, expr) TERNARY_ELEMENTWISE_IMPL_DECLARATION(name);
TERNARY_OPS()
#undef TERNARY_OP_NAME_EXPR

}  // namespace cuda
}  // namespace onnxruntime
