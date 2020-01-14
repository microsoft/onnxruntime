// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
<<<<<<< HEAD
#include <stdint.h>
#include "core/providers/cuda/shared_inc/fast_divmod.h"
=======
>>>>>>> c767e264c52c3bac2c319b630d37f541f4d2a677
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
typedef CtxNull CtxGelu;

<<<<<<< HEAD
#define UNARY_ACTIVATION_OPS()              \
  UNARY_ACTIVATION_OP_NAME(Elu)             \
  UNARY_ACTIVATION_OP_NAME(HardSigmoid)     \
  UNARY_ACTIVATION_OP_NAME(LeakyRelu)       \
  UNARY_ACTIVATION_OP_NAME(Relu)            \
  UNARY_ACTIVATION_OP_NAME(Selu)            \
  UNARY_ACTIVATION_OP_NAME(Sigmoid)         \
  UNARY_ACTIVATION_OP_NAME(Softplus)        \
  UNARY_ACTIVATION_OP_NAME(Softsign)        \
  UNARY_ACTIVATION_OP_NAME(Tanh)            \
  UNARY_ACTIVATION_OP_NAME(ThresholdedRelu) \
  UNARY_ACTIVATION_OP_NAME(Gelu)
=======
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
>>>>>>> c767e264c52c3bac2c319b630d37f541f4d2a677

#define UNARY_ACTIVATION_IMPL_DECLARATION(name) \
  template <typename T>                         \
  void Impl_##name(                             \
      const T* input_data,                      \
      T* output_data,                           \
      const Ctx##name* func_ctx,                \
      size_t count)

#define UNARY_ACTIVATION_OP_NAME(name) UNARY_ACTIVATION_IMPL_DECLARATION(name);
UNARY_ACTIVATION_OPS()
#undef UNARY_ACTIVATION_OP_NAME

// Put Gradients Related Below
typedef CtxNull CtxGeluGrad;

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
}  // namespace onnxruntime
