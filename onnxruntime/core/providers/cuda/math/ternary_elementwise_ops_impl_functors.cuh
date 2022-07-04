// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/math/ternary_elementwise_ops_impl.h"
#include "orttraining/training_ops/cpu/activation/gelu_computation_mode.h"
#include "orttraining/training_ops/cuda/activation/gelu_grad_impl_common.cuh"
namespace onnxruntime {
namespace cuda {

// define the device functors that perform the computation on scalars

#define OP_FUNCTOR_DEFINITION(name, expr)                     \
  template <class T>                                          \
  struct OP_##name {                                          \
    __device__ __inline__ T operator()(T a, T b, T c) const { \
      return (expr);                                          \
    }                                                         \
  };

#define TERNARY_OP_NAME_EXPR(name, expr) \
  OP_FUNCTOR_DEFINITION(name, expr)

TERNARY_OPS()

#undef TERNARY_OP_NAME_EXPR
#undef OP_FUNCTOR_DEFINITION

}  // namespace cuda
}  // namespace onnxruntime
