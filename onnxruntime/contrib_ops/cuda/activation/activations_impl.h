// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/activation/activations_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

typedef onnxruntime::cuda::CtxAlphaBeta CtxAffine;
typedef onnxruntime::cuda::CtxAlphaBeta CtxParametricSoftplus;
typedef onnxruntime::cuda::CtxAlphaBeta CtxScaledTanh;
typedef onnxruntime::cuda::CtxAlpha CtxQuickGelu;
typedef onnxruntime::cuda::CtxNull CtxLog1p;

#define UNARY_CONTRIB_ACTIVATION_OPS()         \
  UNARY_ACTIVATION_OP_NAME(ScaledTanh)         \
  UNARY_ACTIVATION_OP_NAME(Affine)             \
  UNARY_ACTIVATION_OP_NAME(ParametricSoftplus) \
  UNARY_ACTIVATION_OP_NAME(QuickGelu)          \
  UNARY_ACTIVATION_OP_NAME(Log1p)

#define UNARY_ACTIVATION_OP_NAME(name) UNARY_ACTIVATION_IMPL_DECLARATION(name);
UNARY_CONTRIB_ACTIVATION_OPS()
#undef UNARY_ACTIVATION_OP_NAME

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
