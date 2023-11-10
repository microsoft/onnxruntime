// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/framework/float8.h"
#include <cuda_runtime.h>

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
      cudaStream_t stream,                       \
      const T* input_data,                       \
      T* output_data,                            \
      size_t count)

#define UNARY_OP_NAME_EXPR(name, expr) UNARY_ELEMENTWISE_IMPL_DECLARATION(name);
UNARY_OPS()
#undef UNARY_OP_NAME_EXPR

// Cast

#define DECL_IMPL_CAST(InT, OutT) \
  void Explicit_Impl_Cast(cudaStream_t stream, const InT* input_data, OutT* output_data, size_t count);

#if !defined(DISABLE_FLOAT8_TYPES)

#define DECL_IMPL_CAST_FROM(T)    \
  DECL_IMPL_CAST(T, half)         \
  DECL_IMPL_CAST(T, float)        \
  DECL_IMPL_CAST(T, double)       \
  DECL_IMPL_CAST(T, int8_t)       \
  DECL_IMPL_CAST(T, int16_t)      \
  DECL_IMPL_CAST(T, int32_t)      \
  DECL_IMPL_CAST(T, int64_t)      \
  DECL_IMPL_CAST(T, uint8_t)      \
  DECL_IMPL_CAST(T, uint16_t)     \
  DECL_IMPL_CAST(T, uint32_t)     \
  DECL_IMPL_CAST(T, uint64_t)     \
  DECL_IMPL_CAST(T, bool)         \
  DECL_IMPL_CAST(T, BFloat16)     \
  DECL_IMPL_CAST(T, Float8E4M3FN) \
  DECL_IMPL_CAST(T, Float8E5M2)

#else

#define DECL_IMPL_CAST_FROM(T) \
  DECL_IMPL_CAST(T, half)      \
  DECL_IMPL_CAST(T, float)     \
  DECL_IMPL_CAST(T, double)    \
  DECL_IMPL_CAST(T, int8_t)    \
  DECL_IMPL_CAST(T, int16_t)   \
  DECL_IMPL_CAST(T, int32_t)   \
  DECL_IMPL_CAST(T, int64_t)   \
  DECL_IMPL_CAST(T, uint8_t)   \
  DECL_IMPL_CAST(T, uint16_t)  \
  DECL_IMPL_CAST(T, uint32_t)  \
  DECL_IMPL_CAST(T, uint64_t)  \
  DECL_IMPL_CAST(T, bool)      \
  DECL_IMPL_CAST(T, BFloat16)

#endif

DECL_IMPL_CAST_FROM(half)
DECL_IMPL_CAST_FROM(float)
DECL_IMPL_CAST_FROM(double)
DECL_IMPL_CAST_FROM(int8_t)
DECL_IMPL_CAST_FROM(int16_t)
DECL_IMPL_CAST_FROM(int32_t)
DECL_IMPL_CAST_FROM(int64_t)
DECL_IMPL_CAST_FROM(uint8_t)
DECL_IMPL_CAST_FROM(uint16_t)
DECL_IMPL_CAST_FROM(uint32_t)
DECL_IMPL_CAST_FROM(uint64_t)
DECL_IMPL_CAST_FROM(bool)
DECL_IMPL_CAST_FROM(BFloat16)

#if !defined(DISABLE_FLOAT8_TYPES)

DECL_IMPL_CAST_FROM(Float8E4M3FN)
DECL_IMPL_CAST_FROM(Float8E5M2)

#define DECL_IMPL_CASTSAT(InT, OutT) \
  void Explicit_Impl_CastSat(cudaStream_t stream, const InT* input_data, OutT* output_data, size_t count, bool saturate);

DECL_IMPL_CASTSAT(half, Float8E4M3FN)
DECL_IMPL_CASTSAT(float, Float8E4M3FN)
DECL_IMPL_CASTSAT(half, Float8E5M2)
DECL_IMPL_CASTSAT(float, Float8E5M2)

#endif

template <typename InT, typename OutT>
void Impl_Cast(cudaStream_t stream, const InT* input_data, OutT* output_data, size_t count) {
  Explicit_Impl_Cast(stream, input_data, output_data, count);
}

#if !defined(DISABLE_FLOAT8_TYPES)

template <typename InT, typename OutT>
void Impl_CastSat(
    cudaStream_t stream,
    const InT* input_data,
    OutT* output_data,
    size_t count,
    bool saturate) {
  Explicit_Impl_CastSat(stream, input_data, output_data, count, saturate);
}

#endif

}  // namespace cuda
}  // namespace onnxruntime
