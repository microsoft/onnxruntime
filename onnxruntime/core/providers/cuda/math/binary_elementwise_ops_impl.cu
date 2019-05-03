// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include "binary_elementwise_ops_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/binary_elementwise_impl.cuh"

namespace onnxruntime {
namespace cuda {

#define OP(name, expr)                                   \
  template <class T>                                     \
  struct OP_##name {                                     \
    __device__ __inline__ T operator()(T a, T b) const { \
      return (expr);                                     \
    }                                                    \
  };

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
                          OP_##name<T>(),                  \
                          count);                          \
  }

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, T) \
  template void Impl_##x<T>(size_t output_rank, const int64_t* lhs_padded_strides, const T* lhs_data, const int64_t* rhs_padded_strides, const T* rhs_data, const fast_divmod* fdm_output_strides, const fast_divmod& fdm_H, const fast_divmod& fdm_C, T* output_data, size_t count);

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD(x) \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, uint32_t)     \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, uint64_t)     \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, int32_t)      \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, int64_t)      \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, half)         \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, float)        \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, double)

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL_HFD(x) \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, half)     \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, float)    \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, double)

// create declarations for op and impl
#define BINARY_OP_NAME_EXPR(name, expr) \
  OP(name, expr)                        \
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
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD(Sub)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD(Mul)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD(Div)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_HFD(Pow)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL(And, bool)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL(Or, bool)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL(Xor, bool)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_HFD(PRelu)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD(Greater)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_HFD(Max)

}  // namespace cuda
}  // namespace onnxruntime
