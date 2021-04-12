// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include "unary_elementwise_ops_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/unary_elementwise_impl.cuh"

namespace onnxruntime {
namespace cuda {

#define OP(name, expr)                                     \
  template <typename T>                                    \
  struct OP_##name {                                       \
    __device__ __inline__ T operator()(const T& a) const { \
      return expr;                                         \
    }                                                      \
  };

#define UNARY_ELEMENTWISE_IMPL(name)         \
  UNARY_ELEMENTWISE_IMPL_DECLARATION(name) { \
    UnaryElementWiseImpl(stream,             \
                         input_data,         \
                         output_data,        \
                         OP_##name<T>(),     \
                         count);             \
  }

#define SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, T) \
  template void Impl_##name<T>(cudaStream_t stream, const T* input_data, T* output_data, size_t count);

#define UNARY_OP_NAME_EXPR(name, expr) \
  OP(name, expr)                       \
  UNARY_ELEMENTWISE_IMPL(name)

UNARY_OPS()
#undef UNARY_OP_NAME_EXPR

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

#define SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(name) \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, half)     \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, float)    \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, double)

#define SPECIALIZED_UNARY_ELEMENTWISE_IMPL_CSILHFD(name) \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, int8_t)       \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, int16_t)      \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, int32_t)      \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, int64_t)      \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(name)

#define SPECIALIZED_UNARY_ELEMENTWISE_IMPL_BWUZCSILHFD(name) \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, uint8_t)          \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, uint16_t)         \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, uint32_t)         \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, uint64_t)         \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL_CSILHFD(name)

SPECIALIZED_UNARY_ELEMENTWISE_IMPL_BWUZCSILHFD(Abs)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_CSILHFD(Neg)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Floor)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Ceil)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Reciprocal)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Sqrt)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Log)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Exp)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Erf)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Round)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Sin)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Cos)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL(Not, bool)

// When casting, half needs to be converted via float type from most other types
template <typename T>
struct ViaTypeMap {
  typedef T ViaT;
};

template <>
struct ViaTypeMap<half> {
  typedef float ViaT;
};

#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
template <>
struct ViaTypeMap<nv_bfloat16> {
  typedef float ViaT;
};
#endif

template <typename InT, typename OutT>
struct OP_Cast {
  __device__ __inline__ OutT operator()(const InT& a) const {
    const bool any_float16 = std::is_same<half, InT>::value || std::is_same<half, OutT>::value;
#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    const bool any_bf16 = std::is_same<nv_bfloat16, InT>::value || std::is_same<nv_bfloat16, OutT>::value;
    typedef typename std::conditional<any_bf16, nv_bfloat16, OutT>::type T1;
    typedef typename std::conditional<any_float16, half, T1>::type T;
#else
    typedef typename std::conditional<any_float16, half, OutT>::type T;
#endif
    typedef typename ViaTypeMap<T>::ViaT ViaT;
    return (OutT)((ViaT)a);
  }
};

template <typename InT, typename OutT>
void Impl_Cast(
    cudaStream_t stream,
    const InT* input_data,
    OutT* output_data,
    size_t count) {
  UnaryElementWiseImpl(stream,
                       input_data,
                       output_data,
                       OP_Cast<InT, OutT>(),
                       count);
}

#define SPECIALIZED_CAST_IMPL2(InT, OutT) \
  template void Impl_Cast<InT, OutT>(cudaStream_t stream, const InT* input_data, OutT* output_data, size_t count);

#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
#define SPECIALIZED_CAST_IMPL2_BF16(T) SPECIALIZED_CAST_IMPL2(T, nv_bfloat16)
#else
#define SPECIALIZED_CAST_IMPL2_BF16(T)
#endif

#define SPECIALIZED_CAST_FROM(T)      \
  SPECIALIZED_CAST_IMPL2(T, half)     \
  SPECIALIZED_CAST_IMPL2_BF16(T)      \
  SPECIALIZED_CAST_IMPL2(T, float)    \
  SPECIALIZED_CAST_IMPL2(T, double)   \
  SPECIALIZED_CAST_IMPL2(T, int8_t)   \
  SPECIALIZED_CAST_IMPL2(T, int16_t)  \
  SPECIALIZED_CAST_IMPL2(T, int32_t)  \
  SPECIALIZED_CAST_IMPL2(T, int64_t)  \
  SPECIALIZED_CAST_IMPL2(T, uint8_t)  \
  SPECIALIZED_CAST_IMPL2(T, uint16_t) \
  SPECIALIZED_CAST_IMPL2(T, uint32_t) \
  SPECIALIZED_CAST_IMPL2(T, uint64_t) \
  SPECIALIZED_CAST_IMPL2(T, bool)

SPECIALIZED_CAST_FROM(half)
SPECIALIZED_CAST_FROM(float)
SPECIALIZED_CAST_FROM(double)
SPECIALIZED_CAST_FROM(int8_t)
SPECIALIZED_CAST_FROM(int16_t)
SPECIALIZED_CAST_FROM(int32_t)
SPECIALIZED_CAST_FROM(int64_t)
SPECIALIZED_CAST_FROM(uint8_t)
SPECIALIZED_CAST_FROM(uint16_t)
SPECIALIZED_CAST_FROM(uint32_t)
SPECIALIZED_CAST_FROM(uint64_t)
SPECIALIZED_CAST_FROM(bool)
#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
SPECIALIZED_CAST_FROM(nv_bfloat16)
#endif

}  // namespace cuda
}  // namespace onnxruntime
