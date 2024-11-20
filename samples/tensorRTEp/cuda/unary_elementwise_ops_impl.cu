// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
//#include "unary_elementwise_ops_impl.h"
//#include "core/providers/cuda/cu_inc/common.cuh"
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

//template <>
//struct ViaTypeMap<BFloat16> {
  //typedef float ViaT;
//};

template <typename InT, typename OutT>
struct OP_Cast {
  __device__ __inline__ OutT operator()(const InT& a) const {
    //const bool any_float16 = std::is_same<half, InT>::value || std::is_same<half, OutT>::value;
    //const bool any_bf16 = std::is_same<BFloat16, InT>::value || std::is_same<BFloat16, OutT>::value;
    //typedef typename std::conditional<any_bf16, BFloat16, OutT>::type T1;
    //typedef typename std::conditional<any_float16, half, T1>::type T;
    //typedef typename ViaTypeMap<T>::ViaT ViaT;
    typedef typename ViaTypeMap<OutT>::ViaT ViaT;
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

//template <typename InT, typename OutT>
//struct OP_CastSat {
  //__device__ __inline__ OutT operator()(const InT& a) const;
//};

//template <typename InT, typename OutT>
//struct OP_CastNoSat {
  //__device__ __inline__ OutT operator()(const InT& a) const;
//};

//#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080

//#define OP_CAST(T, NVT)                                                                                     \
  //template <>                                                                                               \
  //struct OP_CastSat<half, T> {                                                                              \
    //__device__ __inline__ T operator()(const half& v) const {                                               \
      //return T(static_cast<unsigned char>(__nv_cvt_halfraw_to_fp8(v, __NV_SATFINITE, NVT)), T::FromBits()); \
    //}                                                                                                       \
  //};                                                                                                        \
  //template <>                                                                                               \
  //struct OP_CastNoSat<half, T> {                                                                            \
    //__device__ __inline__ T operator()(const half& v) const {                                               \
      //return T(static_cast<unsigned char>(__nv_cvt_halfraw_to_fp8(v, __NV_NOSAT, NVT)), T::FromBits());     \
    //}                                                                                                       \
  //};                                                                                                        \
  //template <>                                                                                               \
  //struct OP_CastSat<float, T> {                                                                             \
    //__device__ __inline__ T operator()(const float& v) const {                                              \
      //return T(static_cast<unsigned char>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, NVT)), T::FromBits());   \
    //}                                                                                                       \
  //};                                                                                                        \
  //template <>                                                                                               \
  //struct OP_CastNoSat<float, T> {                                                                           \
    //__device__ __inline__ T operator()(const float& v) const {                                              \
      //return T(static_cast<unsigned char>(__nv_cvt_float_to_fp8(v, __NV_NOSAT, NVT)), T::FromBits());       \
    //}                                                                                                       \
  //};

//#else

//#define OP_CAST(T, NVT)                                        \
  //template <>                                                  \
  //struct OP_CastSat<half, T> {                                 \
    //__device__ __inline__ T operator()(const half& v) const {  \
      //return T(__half2float(v), true);                         \
    //}                                                          \
  //};                                                           \
  //template <>                                                  \
  //struct OP_CastNoSat<half, T> {                               \
    //__device__ __inline__ T operator()(const half& v) const {  \
      //return T(__half2float(v), false);                        \
    //}                                                          \
  //};                                                           \
  //template <>                                                  \
  //struct OP_CastSat<float, T> {                                \
    //__device__ __inline__ T operator()(const float& v) const { \
      //return T(v, true);                                       \
    //}                                                          \
  //};                                                           \
  //template <>                                                  \
  //struct OP_CastNoSat<float, T> {                              \
    //__device__ __inline__ T operator()(const float& v) const { \
      //return T(v, false);                                      \
    //}                                                          \
  //};

//#endif

}  // namespace cuda
}  // namespace onnxruntime
