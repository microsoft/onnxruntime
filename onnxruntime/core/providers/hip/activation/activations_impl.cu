// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <hip/hip_runtime.h>
#include "activations_impl.h"
#include "core/providers/hip/cu_inc/common.cuh"
#include "core/providers/hip/cu_inc/unary_elementwise_impl.cuh"

namespace onnxruntime {
namespace hip {

template <typename T>
struct OP_Elu : public CtxElu {
  __device__ __inline__ T operator()(const T& a) const {
    return a > (T)0 ? a : (T)alpha * (_Exp(a) - (T)1);
  }
};

template <typename T>
struct OP_HardSigmoid : public CtxHardSigmoid {
  __device__ __inline__ T operator()(const T& a) const {
    return _Max(_Min((T)alpha * a + (T)beta, (T)1), (T)0);
  }
};

template <typename T>
struct OP_LeakyRelu : public CtxLeakyRelu {
  __device__ __inline__ T operator()(const T& a) const {
    return a > (T)0 ? a : (T)alpha * a;
  }
};

template <typename T>
struct OP_Relu : public CtxRelu {
  __device__ __inline__ T operator()(const T& a) const {
    return _Max(a, (T)0);
  }
};

template <typename T>
struct OP_Selu : public CtxSelu {
  __device__ __inline__ T operator()(const T& a) const {
    return (T)gamma * (_Max(a, (T)0) + _Min((T)alpha * (_Exp(a) - (T)1), (T)0));
  }
};

template <typename T>
struct OP_Sigmoid : public CtxSigmoid {
  __device__ __inline__ T operator()(const T& a) const {
    return a > T(0) ? (T)1 / ((T)1. + _Exp(-_Abs(a))) : (T)1 - (T)1 / ((T)1 + _Exp(-_Abs(a)));
  }
};

template <typename T>
struct OP_Softplus : public CtxSoftplus {
  __device__ __inline__ T operator()(const T& a) const {
    if (a > (T)0)
      return a + _Log(_Exp(-a) + (T)1);
    else
      return _Log(_Exp(a) + (T)1);
  }
};

template <typename T>
struct OP_Softsign : public CtxSoftsign {
  __device__ __inline__ T operator()(const T& a) const {
    return a / ((T)1. + _Abs(a));
  }
};

template <typename T>
struct OP_Tanh : public CtxTanh {
  __device__ __inline__ T operator()(const T& a) const {
    return _Tanh(a);
  }
};

template <typename T>
struct OP_ThresholdedRelu : public CtxThresholdedRelu {
  __device__ __inline__ T operator()(const T& a) const {
    return a > (T)alpha ? a : (T)0;
  }
};

#define UNARY_ACTIVATION_IMPL(name)                                        \
  UNARY_ACTIVATION_IMPL_DECLARATION(name) {                                \
    UnaryElementWiseImpl(input_data,                                       \
                         output_data,                                      \
                         *reinterpret_cast<const OP_##name<T>*>(func_ctx), \
                         count);                                           \
  }

#define SPECIALIZED_UNARY_ACTIVATION_IMPL(name, T) \
  template void Impl_##name<T>(const T* input_data, T* output_data, const Ctx##name* func_ctx, size_t count);

#define SPECIALIZED_UNARY_ACTIVATIONL_HFD(name)  \
  SPECIALIZED_UNARY_ACTIVATION_IMPL(name, half)  \
  SPECIALIZED_UNARY_ACTIVATION_IMPL(name, float) \
  SPECIALIZED_UNARY_ACTIVATION_IMPL(name, double)

#define UNARY_ACTIVATION_OP_NAME(name) \
  UNARY_ACTIVATION_IMPL(name);         \
  SPECIALIZED_UNARY_ACTIVATIONL_HFD(name)

UNARY_ACTIVATION_OPS()
#undef UNARY_ACTIVATION_OP_NAME

}  // namespace hip
}  // namespace onnxruntime
