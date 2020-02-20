// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/hip/activation/activations_impl.h"

namespace onnxruntime {
namespace contrib {
namespace hip {

typedef onnxruntime::hip::CtxAlphaBeta CtxAffine;
typedef onnxruntime::hip::CtxAlphaBeta CtxParametricSoftplus;
typedef onnxruntime::hip::CtxAlphaBeta CtxScaledTanh;
typedef onnxruntime::hip::CtxNull CtxGelu;

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

}  // namespace hip
}  // namespace contrib
}  // namespace onnxruntime
