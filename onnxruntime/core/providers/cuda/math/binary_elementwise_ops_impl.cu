// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include "core/providers/cuda/math/binary_elementwise_ops_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/binary_elementwise_impl.cuh"
#include "core/providers/cuda/math/binary_elementwise_ops_impl_functors.cuh"

namespace onnxruntime {
namespace cuda {

#define BINARY_ELEMENTWISE_IMPL(name)                      \
  BINARY_ELEMENTWISE_IMPL_DECLARATION(name) {              \
    BinaryElementWiseImpl(stream,                          \
                          output_rank_or_simple_broadcast, \
                          lhs_padded_strides,              \
                          lhs_data,                        \
                          rhs_padded_strides,              \
                          rhs_data,                        \
                          fdm_output_strides,              \
                          fdm_H,                           \
                          fdm_C,                           \
                          output_data,                     \
                          OP_##name<T, T, T>(),            \
                          count);                          \
  }

#define BINARY_ELEMENTWISE_IMPL_T1(name)                   \
  BINARY_ELEMENTWISE_IMPL_DECLARATION_T1(name) {           \
    BinaryElementWiseImpl(stream,                          \
                          output_rank_or_simple_broadcast, \
                          lhs_padded_strides,              \
                          lhs_data,                        \
                          rhs_padded_strides,              \
                          rhs_data,                        \
                          fdm_output_strides,              \
                          fdm_H,                           \
                          fdm_C,                           \
                          output_data,                     \
                          OP_##name<T, T, T1>(),           \
                          count);                          \
  }

#define BINARY_ELEMENTWISE_IMPL_T2(name)                   \
  BINARY_ELEMENTWISE_IMPL_DECLARATION_T2(name) {           \
    BinaryElementWiseImpl(stream,                          \
                          output_rank_or_simple_broadcast, \
                          lhs_padded_strides,              \
                          lhs_data,                        \
                          rhs_padded_strides,              \
                          rhs_data,                        \
                          fdm_output_strides,              \
                          fdm_H,                           \
                          fdm_C,                           \
                          output_data,                     \
                          OP_##name<T, T1, T2>(),          \
                          count);                          \
  }

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, T)                                         \
  template void Impl_##x<T>(cudaStream_t stream,                                          \
                            int32_t output_rank,                                          \
                            const TArray<int64_t>* lhs_padded_strides, const T* lhs_data, \
                            const TArray<int64_t>* rhs_padded_strides, const T* rhs_data, \
                            const TArray<fast_divmod>* fdm_output_strides, const fast_divmod& fdm_H, const fast_divmod& fdm_C, T* output_data, size_t count);

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(x, T, T1)                                         \
  template void ImplT1_##x<T, T1>(cudaStream_t stream,                                           \
                                  int32_t output_rank,                                           \
                                  const TArray<int64_t>* lhs_padded_strides, const T* lhs_data,  \
                                  const TArray<int64_t>* rhs_padded_strides, const T1* rhs_data, \
                                  const TArray<fast_divmod>* fdm_output_strides, const fast_divmod& fdm_H, const fast_divmod& fdm_C, T* output_data, size_t count);

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T2(x, T, T1, T2)                                         \
  template void ImplT2_##x<T, T1, T2>(cudaStream_t stream,                                           \
                                      int32_t output_rank,                                           \
                                      const TArray<int64_t>* lhs_padded_strides, const T1* lhs_data, \
                                      const TArray<int64_t>* rhs_padded_strides, const T2* rhs_data, \
                                      const TArray<fast_divmod>* fdm_output_strides, const fast_divmod& fdm_H, const fast_divmod& fdm_C, T* output_data, size_t count);

#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL_BF16(x) SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, nv_bfloat16)
#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T2_BF16(name) SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T2(name, bool, nv_bfloat16, nv_bfloat16)
#else
#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL_BF16(x)
#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T2_BF16(name)
#endif

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD(x) \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, uint32_t)     \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, uint64_t)     \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, int32_t)      \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, int64_t)      \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, half)         \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL_BF16(x)          \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, float)        \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, double)

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1_ILHFD(x, T) \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(x, T, int32_t)    \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(x, T, int64_t)    \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(x, T, half)       \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(x, T, float)      \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1(x, T, double)

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL_OIL(x) \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, bool)     \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, int32_t)  \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, int64_t)

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL_HFD(x) \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, half)     \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL_BF16(x)      \
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
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD(Max)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD(Min)

// create declarations for impl for Pow
BINARY_ELEMENTWISE_IMPL_T1(Pow)

SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1_ILHFD(Pow, int32_t)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1_ILHFD(Pow, int64_t)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1_ILHFD(Pow, float)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1_ILHFD(Pow, double)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T1_ILHFD(Pow, half)

// create declarations for impl2
#define BINARY_OP_NAME_EXPR2(name, expr) \
  BINARY_ELEMENTWISE_IMPL_T2(name)

BINARY_OPS2()
#undef BINARY_OP_NAME_EXPR2

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD2(name)               \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T2(name, bool, uint32_t, uint32_t) \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T2(name, bool, uint64_t, uint64_t) \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T2(name, bool, int32_t, int32_t)   \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T2(name, bool, int64_t, int64_t)   \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T2(name, bool, half, half)         \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T2_BF16(name)                      \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T2(name, bool, float, float)       \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T2(name, bool, double, double)

SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD2(Greater)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD2(Equal)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_T2(Equal, bool, bool, bool)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD2(Less)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD2(GreaterOrEqual)
SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD2(LessOrEqual)

}  // namespace cuda
}  // namespace onnxruntime
