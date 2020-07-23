// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include "binary_elementwise_ops_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/binary_elementwise_impl.cuh"
#include "core/providers/cuda/math/binary_elementwise_ops_impl_functors.cuh"

namespace onnxruntime {
namespace cuda {

#define BINARY_ELEMENTWISE_IMPL(name)                      \
  BINARY_ELEMENTWISE_IMPL_DECLARATION(name) {              \
    BinaryElementWiseImpl(output_rank_or_simple_broadcast, \
                          lhs_padded_strides,              \
                          lhs_data,                        \
                          rhs_padded_strides,              \
                          rhs_data,                        \
                          fdm_output_strides,              \
                          fdm_H,                           \
                          fdm_C,                           \
                          output_data,                     \
                          OP_##name<T, T>(),               \
                          count);                          \
  }

#define BINARY_ELEMENTWISE_IMPL_T1(name)                   \
  BINARY_ELEMENTWISE_IMPL_DECLARATION_T1(name) {           \
    BinaryElementWiseImpl(output_rank_or_simple_broadcast, \
                          lhs_padded_strides,              \
                          lhs_data,                        \
                          rhs_padded_strides,              \
                          rhs_data,                        \
                          fdm_output_strides,              \
                          fdm_H,                           \
                          fdm_C,                           \
                          output_data,                     \
                          OP_##name<T, T1>(),              \
                          count);                          \
  }

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, T)                                         \
  template void Impl_##x<T>(int32_t output_rank,                                          \
                            const TArray<int64_t>* lhs_padded_strides, const T* lhs_data, \
                            const TArray<int64_t>* rhs_padded_strides, const T* rhs_data, \
                            const TArray<fast_divmod>* fdm_output_strides, const fast_divmod& fdm_H, const fast_divmod& fdm_C, T* output_data, size_t count);

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(x, T, T1)                                         \
  template void ImplT1_##x<T, T1>(int32_t output_rank,                                           \
                                  const TArray<int64_t>* lhs_padded_strides, const T* lhs_data,  \
                                  const TArray<int64_t>* rhs_padded_strides, const T1* rhs_data, \
                                  const TArray<fast_divmod>* fdm_output_strides, const fast_divmod& fdm_H, const fast_divmod& fdm_C, T* output_data, size_t count);

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD(x) \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, uint32_t)     \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, uint64_t)     \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, int32_t)      \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, int64_t)      \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, half)         \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, float)        \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, double)

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL_OIL(x) \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, bool)     \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, int32_t)  \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, int64_t)

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL_HFD(x) \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, half)     \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, float)    \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, double)

// create declarations for impl
#define BINARY_OP_NAME_EXPR(name, expr) \
  BINARY_ELEMENTWISE_IMPL(name)

BINARY_OPS()
#undef BINARY_OP_NAME_EXPR

// create specialized impl
// the postfix of means the types supported by the op:
// B: uint8_t
// W: uint16_t
// U: uint32_t
// Z: uint64_t
// C: int8_t
// S: int16_t
// I: int32_t
// L: int64_t
// H: float16
// F: float
// D: double
// O: bool

SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD(Add)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL(Add, bool)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD(Sub)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD(Mul)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD(Div)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_HFD(Pow_7)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL(And, bool)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL(Or, bool)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL(Xor, bool)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_HFD(PRelu)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD(Greater)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_OIL(Equal)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD(Max)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD(Min)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD(Less)

// create declarations for impl for Pow
BINARY_ELEMENTWISE_IMPL_T1(Pow)

SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(Pow, int32_t, int32_t)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(Pow, int32_t, int64_t)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(Pow, int32_t, float)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(Pow, int32_t, double)

SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(Pow, int64_t, int32_t)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(Pow, int64_t, int64_t)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(Pow, int64_t, float)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(Pow, int64_t, double)

SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(Pow, float, int32_t)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(Pow, float, int64_t)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(Pow, float, float)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(Pow, float, double)

SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(Pow, double, int32_t)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(Pow, double, int64_t)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(Pow, double, float)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(Pow, double, double)

}  // namespace cuda
}  // namespace onnxruntime
