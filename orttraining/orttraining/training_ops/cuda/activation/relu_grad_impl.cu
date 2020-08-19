// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include "orttraining/training_ops/cuda/activation/relu_grad_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/binary_elementwise_impl.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T>
struct OP_ReluGrad : public CtxReluGrad {
  __device__ __inline__ T operator()(const T& dy, const T& x) const {
    return dy ? x > 0 : 0;
  }
};

#define BINARY_ELEMENTWISE_IMPL(name)                                                  \
  BINARY_ELEMENTWISE_IMPL_DECLARATION(name) {                                          \
    BinaryElementWiseNoBroadcastImpl(lhs_data, rhs_data,                               \
                                     output_data,                                      \
                                     *reinterpret_cast<const OP_##name<T>*>(func_ctx), \
                                     count);                                           \
  }

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL(name, T) \
  template void Impl_##name<T>(const T* lhs_data, const T* rhs_data, T* output_data, const Ctx##name* func_ctx, size_t count);

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL_HFD(x) \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, half)     \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, float)    \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, double)

//  BINARY_ELEMENTWISE_IMPL(name);      
#define RELU_GRAD_OP_NAME(name) \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL_HFD(name)

RELU_GRAD_OPS()
#undef RELU_GRAD_OP_NAME

}  // namespace cuda
}  // namespace onnxruntime
