// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/activation/activations_impl.h"
#include "core/providers/cuda/shared_inc/fast_divmod.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

typedef onnxruntime::cuda::CtxAlphaBeta CtxAffine;
typedef onnxruntime::cuda::CtxAlphaBeta CtxParametricSoftplus;
typedef onnxruntime::cuda::CtxAlphaBeta CtxScaledTanh;
typedef onnxruntime::cuda::CtxNull CtxGelu;

#define UNARY_CONTRIB_ACTIVATION_OPS()         \
  UNARY_ACTIVATION_OP_NAME(ScaledTanh)         \
  UNARY_ACTIVATION_OP_NAME(Affine)             \
  UNARY_ACTIVATION_OP_NAME(ParametricSoftplus) \
  UNARY_ACTIVATION_OP_NAME(Gelu)

#define UNARY_ACTIVATION_IMPL_DECLARATION(name) \
  template <typename T>                         \
  void Impl_##name(                             \
      const T* input_data,                      \
      T* output_data,                           \
      const Ctx##name* func_ctx,                \
      size_t count)

#define UNARY_ACTIVATION_OP_NAME(name) UNARY_ACTIVATION_IMPL_DECLARATION(name);
UNARY_CONTRIB_ACTIVATION_OPS()
#undef UNARY_ACTIVATION_OP_NAME

// Put Gradients Related Below
typedef onnxruntime::cuda::CtxNull CtxGeluGrad;

#define ACTIVATION_GRAD_OPS() \
  ACTIVATION_GRAD_OP_NAME(GeluGrad)

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
}  // namespace contrib
}  // namespace onnxruntime
