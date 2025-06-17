// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include "cu_inc/unary_elementwise_impl.cuh"

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
#include "cuda_fp8.h"
#endif
#include <cuda_fp16.h>

namespace onnxruntime {

namespace cuda {

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
// X: BFloat16

// When casting, half needs to be converted via float type from most other types
template <typename T>
struct ViaTypeMap {
  typedef T ViaT;
};

template <>
struct ViaTypeMap<half> {
  typedef float ViaT;
};

template <typename InT, typename OutT>
struct OP_Cast {
  __device__ __inline__ OutT operator()(const InT& a) const {
    const bool any_float16 = std::is_same<half, InT>::value || std::is_same<half, OutT>::value;
    typedef typename std::conditional<any_float16, half, OutT>::type T;
    typedef typename ViaTypeMap<T>::ViaT ViaT;
    return (OutT)((ViaT)a);
  }
};

#define IMPL_CAST_IMPL(InT, OutT)                                                                        \
  void Explicit_Impl_Cast(cudaStream_t stream, const InT* input_data, OutT* output_data, size_t count) { \
    UnaryElementWiseImpl(stream, input_data, output_data, OP_Cast<InT, OutT>(), count);                  \
  }

#define IMPL_CAST_IMPL_THROW(InT, OutT)                                                              \
  void Explicit_Impl_Cast(cudaStream_t /*stream*/, const InT* /*input_data*/, OutT* /*output_data*/, \
                          size_t /*count*/) {                                                        \
    ORT_THROW("Cast from " #InT " to " #OutT " must define saturate.");                              \
  }

#define IMPL_CAST_IMPL_FROM(T) \
  IMPL_CAST_IMPL(T, half)      \
  IMPL_CAST_IMPL(T, float)     \
  IMPL_CAST_IMPL(T, double)    \
  IMPL_CAST_IMPL(T, int8_t)    \
  IMPL_CAST_IMPL(T, int16_t)   \
  IMPL_CAST_IMPL(T, int32_t)   \
  IMPL_CAST_IMPL(T, int64_t)   \
  IMPL_CAST_IMPL(T, uint8_t)   \
  IMPL_CAST_IMPL(T, uint16_t)  \
  IMPL_CAST_IMPL(T, uint32_t)  \
  IMPL_CAST_IMPL(T, uint64_t)  \
  IMPL_CAST_IMPL(T, bool)      \
  //IMPL_CAST_IMPL(T, BFloat16)

IMPL_CAST_IMPL_FROM(half)
IMPL_CAST_IMPL_FROM(float)
IMPL_CAST_IMPL_FROM(double)
IMPL_CAST_IMPL_FROM(int8_t)
IMPL_CAST_IMPL_FROM(int16_t)
IMPL_CAST_IMPL_FROM(int32_t)
IMPL_CAST_IMPL_FROM(int64_t)
IMPL_CAST_IMPL_FROM(uint8_t)
IMPL_CAST_IMPL_FROM(uint16_t)
IMPL_CAST_IMPL_FROM(uint32_t)
IMPL_CAST_IMPL_FROM(uint64_t)
IMPL_CAST_IMPL_FROM(bool)
//IMPL_CAST_IMPL_FROM(BFloat16)

}  // namespace cuda
}  // namespace onnxruntime
