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

template <>
struct OP_GeluGrad<half> : public CtxGeluGrad {
  __device__ __inline__ half operator()(const half& dy, const half& x) const {
    return static_cast<half>(
        ComputeGeluGradScalar(static_cast<float>(dy), static_cast<float>(x), gelu_computation_mode::Default{}));
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
    return x > T{0} ? dy : T{0};
  }
};

template <typename T>
struct OP_SigmoidGrad : public CtxSigmoidGrad {
  __device__ __inline__ T operator()(const T& dy, const T& y) const {
    return dy * y * ((T)1 - y);
  }
};

template <typename T>
struct OP_QuickGeluGrad : public CtxQuickGeluGrad {
  __device__ __inline__ T operator()(const T& dy, const T& x) const {
    T v = x * static_cast<T>(alpha);
    T one = static_cast<T>(1.f);
    T zero = static_cast<T>(0.f);
    T sigmoid = v >= zero ? one / (one + _Exp(-v)) : one - one / (one + _Exp(v));
    return dy * sigmoid * (one + v * (one - sigmoid));
  }
};

template <typename T>
struct OP_TanhGrad : public CtxTanhGrad {
  __device__ __inline__ T operator()(const T& dy, const T& y) const {
    return dy * ((T)1 - y * y);
  }
};

template <typename T>
struct OP_LeakyReluGrad : public CtxLeakyReluGrad {
  __device__ __inline__ T operator()(const T& dy, const T& y) const {
    return dy * (y > T{0} ? T{1} : static_cast<T>(alpha));
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

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL_HFDX(x) \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, half)     \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, float)    \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, double)  \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, BFloat16)

#define ACTIVATION_GRAD_OP_NAME(name) \
  BINARY_ELEMENTWISE_IMPL(name);      \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL_HFDX(name)

ACTIVATION_GRAD_OPS()
#undef ACTIVATION_GRAD_OP_NAME

}  // namespace cuda
}  // namespace onnxruntime
