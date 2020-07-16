// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/math/binary_elementwise_ops_impl.h"

namespace onnxruntime {
namespace cuda {

// define the device functors that perform the computation on scalars

#define OP_FUNCTOR_DEFINITION(name, expr)                 \
  template <class T, class T1>                            \
  struct OP_##name {                                      \
    __device__ __inline__ T operator()(T a, T1 b) const { \
      return (expr);                                      \
    }                                                     \
  };

#define BINARY_OP_NAME_EXPR(name, expr) \
  OP_FUNCTOR_DEFINITION(name, expr)

BINARY_OPS()

OP_FUNCTOR_DEFINITION(Pow, _Pow(a, b))

#undef BINARY_OP_NAME_EXPR
#undef OP_FUNCTOR_DEFINITION

}  // namespace cuda
}  // namespace onnxruntime
