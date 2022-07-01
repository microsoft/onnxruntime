// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
// #include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/ternary_elementwise_impl.cuh"
#include "orttraining/training_ops/cpu/activation/gelu_computation_mode.h"
#include "core/providers/cuda/math/ternary_elementwise_ops_impl_functors.cuh"
#include <cuda_runtime.h>
#include "orttraining/training_ops/cuda/activation/activations_grad_impl.h"
// #include "orttraining/training_ops/cuda/activation/gelu_grad_impl_common.cuh"
#include "core/providers/cuda/cu_inc/common.cuh"

// #include "orttraining/training_ops/cpu/activation/gelu_computation_mode.h"
// #include "orttraining/training_ops/cuda/activation/gelu_grad_impl_common.cuh"

namespace onnxruntime {
namespace cuda {

#define TERNARY_ELEMENTWISE_IMPL(name)                     \
  TERNARY_ELEMENTWISE_IMPL_DECLARATION(name) {             \
    TenaryElementWiseImpl(stream,                          \
                          output_rank_or_simple_broadcast, \
                          cond_index_type,                 \
                          cond_padded_strides,             \
                          cond_data,                       \
                          x_index_type,                    \
                          x_padded_strides,                \
                          x_data,                          \
                          y_index_type,                    \
                          y_padded_strides,                \
                          y_data,                          \
                          fdm_output_strides,              \
                          output_data,                     \
                          count,                           \
                          OP_##name<T>());                 \
  }

#define SPECIALIZED_TERNARY_ELEMENTWISE_IMPL(x, T)   \
  template void Impl_##x<T>(                         \
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
      size_t count);

#define SPECIALIZED_TERNARY_ELEMENTWISE_IMPL_UZILHFD(x) \
  SPECIALIZED_TERNARY_ELEMENTWISE_IMPL(x, half)         \
  SPECIALIZED_TERNARY_ELEMENTWISE_IMPL(x, float)        \
  SPECIALIZED_TERNARY_ELEMENTWISE_IMPL(x, double)       \
  SPECIALIZED_TERNARY_ELEMENTWISE_IMPL(x, BFloat16)

// create declarations for impl
#define TERNARY_OP_NAME_EXPR(name, expr) \
  TERNARY_ELEMENTWISE_IMPL(name)

TERNARY_OPS()

#undef TERNARY_OP_NAME_EXPR

SPECIALIZED_TERNARY_ELEMENTWISE_IMPL_UZILHFD(BiasGeluGrad_dX)

}  // namespace cuda
}  // namespace onnxruntime
