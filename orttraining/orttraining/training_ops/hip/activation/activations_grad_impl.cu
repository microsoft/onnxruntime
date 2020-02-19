// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <hip/hip_runtime.h>
#include "orttraining/training_ops/hip/activation/activations_grad_impl.h"
#include "core/providers/hip/cu_inc/common.cuh"
#include "core/providers/hip/cu_inc/binary_elementwise_impl.cuh"

namespace onnxruntime {
namespace hip {

template <typename T>
struct OP_GeluGrad : public CtxGeluGrad {
  __device__ __inline__ T operator()(const T& dy, const T& x) const {
    const T kAlpha = T(M_2_SQRTPI) * T(M_SQRT1_2) * T(0.5);
    return dy * (_Normcdf(x) + x * kAlpha * _Exp(-T(0.5) * x * x));
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

#define ACTIVATION_GRAD_OP_NAME(name) \
  BINARY_ELEMENTWISE_IMPL(name);      \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL_HFD(name)

ACTIVATION_GRAD_OPS()
#undef ACTIVATION_GRAD_OP_NAME

}  // namespace hip
}  // namespace onnxruntime
