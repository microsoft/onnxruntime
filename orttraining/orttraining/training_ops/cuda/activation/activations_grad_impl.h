// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/activation/activations_impl.h"

namespace onnxruntime {
namespace cuda {

typedef onnxruntime::cuda::CtxNull CtxGeluGrad;
typedef onnxruntime::cuda::CtxNull CtxFastGeluGrad;

#define ACTIVATION_GRAD_OPS() \
  ACTIVATION_GRAD_OP_NAME(GeluGrad) \
  ACTIVATION_GRAD_OP_NAME(FastGeluGrad)

#define BINARY_ELEMENTWISE_IMPL_DECLARATION(name) \
  template <typename T>                           \
  void Impl_##name(const T* lhs_data,             \
                   const T* rhs_data,             \
                   T* output_data,                \
                   const Ctx##name* func_ctx,     \
                   size_t count)

#define ACTIVATION_GRAD_OP_NAME(name) BINARY_ELEMENTWISE_IMPL_DECLARATION(name);
ACTIVATION_GRAD_OPS()
#undef ACTIVATION_GRAD_OP_NAME

}  // namespace cuda
}  // namespace onnxruntime
