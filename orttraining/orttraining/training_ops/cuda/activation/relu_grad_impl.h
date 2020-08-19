// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/activation/activations_impl.h"

namespace onnxruntime {
namespace cuda {

typedef onnxruntime::cuda::CtxNull CtxReluGrad;

#define RELU_GRAD_OPS() \
  RELU_GRAD_OP_NAME(ReluGrad)

#define BINARY_ELEMENTWISE_IMPL_DECLARATION(name) \
  template <typename T>                           \
  void Impl_##name(const T* lhs_data,             \
                   const T* rhs_data,             \
                   T* output_data,                \
                   const Ctx##name* func_ctx,     \
                   size_t count)

#define RELU_GRAD_OP_NAME(name) BINARY_ELEMENTWISE_IMPL_DECLARATION(name);
RELU_GRAD_OPS()
#undef RELU_GRAD_OP_NAME

}  // namespace cuda
}  // namespace onnxruntime