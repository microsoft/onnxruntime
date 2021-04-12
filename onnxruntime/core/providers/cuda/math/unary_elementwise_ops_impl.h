// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace cuda {

// This macro simplifies coding to add a new op with following steps:
// 1. Add a new entry in UNARY_OPS() list
// 2. (optional) Define templated single element operator in unary_elementwise_ops_impl.cu
// 3. (optional) Implement specialized single element operator
// 4. Add op kernel class definition in unary_elementwise_ops.h
// 5. Add op kernel registration and compute specialization in unary_elementwise_ops.cc

#define UNARY_OPS()                        \
  UNARY_OP_NAME_EXPR(Abs, _Abs(a))         \
  UNARY_OP_NAME_EXPR(Neg, -a)              \
  UNARY_OP_NAME_EXPR(Ceil, _Ceil(a))       \
  UNARY_OP_NAME_EXPR(Floor, _Floor(a))     \
  UNARY_OP_NAME_EXPR(Reciprocal, T(1) / a) \
  UNARY_OP_NAME_EXPR(Sqrt, _Sqrt(a))       \
  UNARY_OP_NAME_EXPR(Exp, _Exp(a))         \
  UNARY_OP_NAME_EXPR(Log, _Log(a))         \
  UNARY_OP_NAME_EXPR(Erf, _Erf(a))         \
  UNARY_OP_NAME_EXPR(Not, !a)              \
  UNARY_OP_NAME_EXPR(Round, _Round(a))     \
  UNARY_OP_NAME_EXPR(Sin, _Sin(a))         \
  UNARY_OP_NAME_EXPR(Cos, _Cos(a))

#define UNARY_ELEMENTWISE_IMPL_DECLARATION(name) \
  template <typename T>                          \
  void Impl_##name(                              \
      cudaStream_t stream,                 \
      const T* input_data,                       \
      T* output_data,                            \
      size_t count)

#define UNARY_OP_NAME_EXPR(name, expr) UNARY_ELEMENTWISE_IMPL_DECLARATION(name);
UNARY_OPS()
#undef UNARY_OP_NAME_EXPR

template <typename InT, typename OutT>
void Impl_Cast(
    cudaStream_t stream,
    const InT* input_data,
    OutT* output_data,
    size_t count);

}  // namespace cuda
}  // namespace onnxruntime
