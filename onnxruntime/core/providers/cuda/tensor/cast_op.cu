// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>

#include "core/providers/cuda/cu_inc/common.cuh"

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
#include "cuda_fp8.h"
#endif

namespace onnxruntime {
namespace cuda {

template <typename OutT, typename InT>
struct CastStd;

template <typename OutT, typename InT>
struct CastSat;

template <typename OutT, typename InT>
struct CastNoSat;

#if !defined(DISABLE_FLOAT8_TYPES)

#if CUDA_VERSION < 11080
#error Float 8 types are defined with CUDA>=11.8. Set flag DISABLE_FLOAT8_TYPES to disable them.
#endif

template <>
struct CastStd<float, Float8E4M3FN> {
  __device__ __forceinline__ float operator()(Float8E4M3FN v) const {
    return __half2float(__nv_cvt_fp8_to_halfraw(v.val, __NV_E4M3));
  }
};

template <>
struct CastStd<half, Float8E4M3FN> {
  __device__ __forceinline__ half operator()(Float8E4M3FN v) const {
    return __nv_cvt_fp8_to_halfraw(v.val, __NV_E4M3);
  }
};

template <>
struct CastStd<float, Float8E5M2> {
  __device__ __forceinline__ float operator()(Float8E5M2 v) const {
    return __half2float(__nv_cvt_fp8_to_halfraw(v.val, __NV_E5M2));
  }
};

template <>
struct CastStd<half, Float8E5M2> {
  __device__ __forceinline__ half operator()(Float8E5M2 v) const {
    return __nv_cvt_fp8_to_halfraw(v.val, __NV_E5M2);
  }
};

template <>
struct CastSat<Float8E4M3FN, float> {
  __device__ __forceinline__ Float8E4M3FN operator()(float v, bool saturate) const {
    return Float8E4M3FN(static_cast<unsigned char>(__nv_cvt_float_to_fp8(v, saturate ? __NV_SATFINITE : __NV_NOSAT, __NV_E4M3)), Float8E4M3FN::FromBits());
  }
};

template <>
struct CastSat<Float8E4M3FN, half> {
  __device__ __forceinline__ Float8E4M3FN operator()(half v, bool saturate) const {
    return Float8E4M3FN(static_cast<unsigned char>(__nv_cvt_halfraw_to_fp8(v, saturate ? __NV_SATFINITE : __NV_NOSAT, __NV_E4M3)), Float8E4M3FN::FromBits());
  }
};

template <>
struct CastSat<Float8E5M2, float> {
  __device__ __forceinline__ Float8E5M2 operator()(float v, bool saturate) const {
    return Float8E5M2(static_cast<unsigned char>(__nv_cvt_float_to_fp8(v, saturate ? __NV_SATFINITE : __NV_NOSAT, __NV_E4M3)), Float8E5M2::FromBits());
  }
};

template <>
struct CastSat<Float8E5M2, half> {
  __device__ __forceinline__ Float8E5M2 operator()(half v, bool saturate) const {
    return Float8E5M2(static_cast<unsigned char>(__nv_cvt_halfraw_to_fp8(v, saturate ? __NV_SATFINITE : __NV_NOSAT, __NV_E4M3)), Float8E5M2::FromBits());
  }
};

#else

template <>
struct CastStd<float, Float8E4M3FN> {
  __device__ __forceinline__ float operator()(Float8E4M3FN v) const {
    return v.ToFloat();
  }
};

template <>
struct CastStd<half, Float8E4M3FN> {
  __device__ __forceinline__ half operator()(Float8E4M3FN v) const {
    return __float2half(v.ToFloat());
  }
};

template <>
struct CastStd<float, Float8E5M2> {
  __device__ __forceinline__ float operator()(Float8E5M2 v) const {
    return v.ToFloat();
  }
};

template <>
struct CastStd<half, Float8E5M2> {
  __device__ __forceinline__ half operator()(Float8E5M2 v) const {
    return __float2half(v.ToFloat());
  }
};

template <>
struct CastSat<Float8E4M3FN, float> {
  __device__ __forceinline__ Float8E4M3FN operator()(float v, bool saturate) const {
    return Float8E4M3FN(v, saturate);
  }
};

template <>
struct CastSat<Float8E4M3FN, half> {
  __device__ __forceinline__ Float8E4M3FN operator()(half v, bool saturate) const {
    return Float8E4M3FN(__half2float(v), saturate);
  }
};

template <>
struct CastSat<Float8E5M2, float> {
  __device__ __forceinline__ Float8E5M2 operator()(float v, bool saturate) const {
    return Float8E5M2(v, saturate);
  }
};

template <>
struct CastSat<Float8E5M2, half> {
  __device__ __forceinline__ Float8E5M2 operator()(half v, bool saturate) const {
    return Float8E5M2(__half2float(v), saturate);
  }
};

#endif

template <int NumThreadsPerBlock, int NumElementsPerThread, typename OutT, typename InT>
__global__ void CastKernelStd(const InT* input, OutT* output, CUDA_LONG N, CastStd<OutT, InT> cast) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output[id] = cast(input[id]);
      id += NumThreadsPerBlock;
    }
  }
}

template <class OutT, class InT>
Status CudaCastStd(cudaStream_t stream, const InT* input, OutT* output, size_t num_of_element) {
  if (num_of_element <= 0)
    return Status::OK();

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CastKernelStd<GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread, OutT, InT><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      static_cast<int>(num_of_element),
      CastStd<OutT, InT>()
      );
  return Status::OK();
}

#if !defined(DISABLE_FLOAT8_TYPES)

template <int NumThreadsPerBlock, int NumElementsPerThread, typename OutT, typename InT>
__global__ void CastKernelSat(const InT* input, OutT* output, CUDA_LONG N, CastSat<OutT, InT> cast, bool saturate) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output[id] = cast(input[id], saturate);
      id += NumThreadsPerBlock;
    }
  }
}

template <class OutT, class InT>
Status CudaCastSat(cudaStream_t stream, const InT* input, OutT* output, size_t num_of_element, bool saturate) {
  if (num_of_element <= 0)
    return Status::OK();

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CastKernelSat<GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread, OutT, InT><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      static_cast<int>(num_of_element),
      CastSat<OutT, InT>(),
      saturate
      );
  return Status::OK();
}

template Status CudaCastStd<float, Float8E4M3FN>(cudaStream_t stream, const Float8E4M3FN* input, float* output, size_t num_of_element);
template Status CudaCastStd<half, Float8E4M3FN>(cudaStream_t stream, const Float8E4M3FN* input, half* output, size_t num_of_element);

template Status CudaCastSat<Float8E4M3FN, float>(cudaStream_t stream, const float* input, Float8E4M3FN* output, size_t num_of_element, bool saturate);
template Status CudaCastSat<Float8E4M3FN, half>(cudaStream_t stream, const half* input, Float8E4M3FN* output, size_t num_of_element, bool saturate);

template Status CudaCastStd<float, Float8E5M2>(cudaStream_t stream, const Float8E5M2* input, float* output, size_t num_of_element);
template Status CudaCastStd<half, Float8E5M2>(cudaStream_t stream, const Float8E5M2* input, half* output, size_t num_of_element);

template Status CudaCastSat<Float8E5M2, float>(cudaStream_t stream, const float* input, Float8E5M2* output, size_t num_of_element, bool saturate);
template Status CudaCastSat<Float8E5M2, half>(cudaStream_t stream, const half* input, Float8E5M2* output, size_t num_of_element, bool saturate);

#endif

}  // namespace cuda
}  // namespace onnxruntime
