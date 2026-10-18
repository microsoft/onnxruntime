// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include <type_traits>
#include "activations_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/unary_elementwise_impl.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
struct OP_Affine : public CtxAffine {
  __device__ __inline__ T operator()(const T& a) const {
    return a * (T)alpha + (T)beta;
  }
};

template <typename T>
struct OP_ParametricSoftplus : public CtxParametricSoftplus {
  __device__ __inline__ T operator()(const T& a) const {
    if (a > (T)0)
      return (T)alpha * (a * (T)beta + _Log(_Exp(-a * (T)beta) + (T)1));
    else
      return (T)alpha * _Log(_Exp(a * (T)beta) + (T)1);
  }
};

template <typename T>
struct OP_ScaledTanh : public CtxScaledTanh {
  __device__ __inline__ T operator()(const T& a) const {
    return (T)alpha * _Tanh(a * (T)beta);
  }
};

template <typename T>
struct OP_QuickGelu : public CtxQuickGelu {
  __device__ __inline__ T operator()(const T& a) const {
    using ComputeT = std::conditional_t<std::is_same_v<T, double>, double, float>;
    const ComputeT value = static_cast<ComputeT>(a);
    const ComputeT scaled = value * static_cast<ComputeT>(alpha);
    if constexpr (std::is_same_v<ComputeT, float>) {
      // Float exp enters its subnormal range near -87; keep the final product wide.
      if (scaled < -87.0f) {
        const double wide_value = static_cast<double>(value);
        const double exponent = _Exp(wide_value * static_cast<double>(alpha));
        return static_cast<T>(static_cast<float>(wide_value * exponent / (1.0 + exponent)));
      }
    }
    const ComputeT exponent = _Exp(scaled >= ComputeT(0) ? -scaled : scaled);
    // Preserve negative tails and round only after the fused sigmoid/product.
    const ComputeT sigmoid = scaled >= ComputeT(0) ? ComputeT(1) / (ComputeT(1) + exponent)
                                                   : exponent / (ComputeT(1) + exponent);
    return static_cast<T>(value * sigmoid);
  }
};

#define UNARY_ACTIVATION_IMPL(name)                                        \
  UNARY_ACTIVATION_IMPL_DECLARATION(name) {                                \
    UnaryElementWiseImpl(stream,                                           \
                         input_data,                                       \
                         output_data,                                      \
                         *reinterpret_cast<const OP_##name<T>*>(func_ctx), \
                         count);                                           \
  }

#define SPECIALIZED_UNARY_ACTIVATION_IMPL(name, T) \
  template void Impl_##name<T>(cudaStream_t stream, const T* input_data, T* output_data, const Ctx##name* func_ctx, size_t count);

#define SPECIALIZED_UNARY_ACTIVATIONL_HFD(name)   \
  SPECIALIZED_UNARY_ACTIVATION_IMPL(name, half)   \
  SPECIALIZED_UNARY_ACTIVATION_IMPL(name, float)  \
  SPECIALIZED_UNARY_ACTIVATION_IMPL(name, double) \
  SPECIALIZED_UNARY_ACTIVATION_IMPL(name, BFloat16)

#define UNARY_ACTIVATION_OP_NAME(name) \
  UNARY_ACTIVATION_IMPL(name);         \
  SPECIALIZED_UNARY_ACTIVATIONL_HFD(name)

UNARY_CONTRIB_ACTIVATION_OPS()
#undef UNARY_ACTIVATION_OP_NAME

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
