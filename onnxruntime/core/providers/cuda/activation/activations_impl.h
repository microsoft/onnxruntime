// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace onnxruntime {
namespace cuda {

struct CtxAlpha {
  float alpha;
};

struct CtxAlphaBeta {
  float alpha;
  float beta;
};

struct CtxAlphaGamma {
  float alpha;
  float gamma;
};

struct CtxNull {
};

typedef CtxAlpha CtxElu;
typedef CtxAlphaBeta CtxHardSigmoid;
typedef CtxAlpha CtxLeakyRelu;
typedef CtxNull CtxRelu;
typedef CtxAlphaGamma CtxSelu;
typedef CtxNull CtxSigmoid;
typedef CtxNull CtxSoftplus;
typedef CtxNull CtxSoftsign;
typedef CtxNull CtxTanh;
typedef CtxAlpha CtxThresholdedRelu;

#define UNARY_ACTIVATION_OPS()          \
  UNARY_ACTIVATION_OP_NAME(Elu)         \
  UNARY_ACTIVATION_OP_NAME(HardSigmoid) \
  UNARY_ACTIVATION_OP_NAME(LeakyRelu)   \
  UNARY_ACTIVATION_OP_NAME(Relu)        \
  UNARY_ACTIVATION_OP_NAME(Selu)        \
  UNARY_ACTIVATION_OP_NAME(Sigmoid)     \
  UNARY_ACTIVATION_OP_NAME(Softplus)    \
  UNARY_ACTIVATION_OP_NAME(Softsign)    \
  UNARY_ACTIVATION_OP_NAME(Tanh)        \
  UNARY_ACTIVATION_OP_NAME(ThresholdedRelu)

#define UNARY_ACTIVATION_IMPL_DECLARATION(name) \
  template <typename T>                         \
  void Impl_##name(                             \
      cudaStream_t stream,                \
      const T* input_data,                      \
      T* output_data,                           \
      const Ctx##name* func_ctx,                \
      size_t count)

#define UNARY_ACTIVATION_OP_NAME(name) UNARY_ACTIVATION_IMPL_DECLARATION(name);
UNARY_ACTIVATION_OPS()
#undef UNARY_ACTIVATION_OP_NAME

}  // namespace cuda
}  // namespace onnxruntime
