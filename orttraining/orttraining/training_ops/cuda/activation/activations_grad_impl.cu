// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include "orttraining/training_ops/cuda/activation/activations_grad_impl.h"
#include "orttraining/training_ops/cuda/activation/gelu_grad_impl_common.cuh"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/binary_elementwise_impl.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T>
struct OP_GeluGrad : public CtxGeluGrad {
  __device__ __inline__ T operator()(const T& dy, const T& x) const {
    return ComputeGeluGradScalar(dy, x, gelu_computation_mode::Default{});
  }
};

template <typename T>
struct OP_FastGeluGrad : public CtxGeluGrad {
  __device__ __inline__ T operator()(const T& dy, const T& x) const {
    return ComputeGeluGradScalar(dy, x, gelu_computation_mode::Approximation{});
  }
};

template <typename T>
struct OP_ReluGrad : public CtxReluGrad {
  __device__ __inline__ T operator()(const T& dy, const T& x) const {
    return x > T {0} ? dy : T {0};
  }
};

#define BINARY_ELEMENTWISE_IMPL(name)                                                  \
  BINARY_ELEMENTWISE_IMPL_DECLARATION(name) {                                          \
    BinaryElementWiseNoBroadcastImpl(stream,                                           \
                                     lhs_data, rhs_data,                               \
                                     output_data,                                      \
                                     *reinterpret_cast<const OP_##name<T>*>(func_ctx), \
                                     count);                                           \
  }

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL(name, T) \
  template void Impl_##name<T>(cudaStream_t stream, const T* lhs_data, const T* rhs_data, T* output_data, const Ctx##name* func_ctx, size_t count);

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL_HFD(x) \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, half)     \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, float)    \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, double)

#define ACTIVATION_GRAD_OP_NAME(name) \
  BINARY_ELEMENTWISE_IMPL(name);      \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL_HFD(name)

ACTIVATION_GRAD_OPS()
#undef ACTIVATION_GRAD_OP_NAME

}  // namespace cuda
}  // namespace onnxruntime
