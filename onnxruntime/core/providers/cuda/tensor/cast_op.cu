// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>

#include "core/providers/cuda/cu_inc/common.cuh"

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
#include "cuda_fp8.h"
#endif

#include "core/framework/float4.h"

#include "cast_op.h"

namespace onnxruntime {
namespace cuda {
namespace cast_helper_impl {
template <typename OutT, typename InT>
struct CastStd;

template <typename OutT, typename InT>
struct CastSat;

template <typename OutT, typename InT>
struct CastNoSat;

#if !defined(DISABLE_FLOAT8_TYPES)

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080

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

#endif  // DISABLE_FLOAT8_TYPES

#if !defined(DISABLE_FLOAT4_TYPES)

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080

template <>
struct CastStd<float2, Float4E2M1x2> {
  __device__ __forceinline__ float2 operator()(Float4E2M1x2 v) const {
    return v.ToCudaFloat2();
  }
};

template <>
struct CastStd<Float4E2M1x2, float2> {
  __device__ __forceinline__ Float4E2M1x2 operator()(float2 v) const {
    return Float4E2M1x2(v);
  }
};

template <>
struct CastStd<float, Float4E2M1x2> {
  __device__ __forceinline__ float operator()(Float4E2M1x2 v) const {
    return v.ToCudaFloat2().x;
  }
};

template <>
struct CastStd<Float4E2M1x2, float> {
  __device__ __forceinline__ Float4E2M1x2 operator()(float v) const {
    return Float4E2M1x2(v, 0);
  }
};

#else
template <>
struct CastStd<float2, Float4E2M1x2> {
  __device__ __forceinline__ float2 operator()(Float4E2M1x2 v) const {
    auto float_pair = v.ToFloat2();

    float2 res;
    res.x = float_pair.first;
    res.y = float_pair.second;

    return res;
  }
};

template <>
struct CastStd<Float4E2M1x2, float2> {
  __device__ __forceinline__ Float4E2M1x2 operator()(float2 v) const {
    return Float4E2M1x2(v.x, v.y);
  }
};

template <>
struct CastStd<float, Float4E2M1x2> {
  __device__ __forceinline__ float operator()(Float4E2M1x2 v) const {
    auto float_pair = v.ToFloat2();
    return float_pair.x;
  }
};

template <>
struct CastStd<Float4E2M1x2, float> {
  __device__ __forceinline__ Float4E2M1x2 operator()(float v) const {
    return Float4E2M1x2(v, 0);
  }
};

#endif

#endif  // DISABLE_FLOAT4_TYPES

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
Status CudaCastStd(cudaStream_t stream, const InT* input, OutT* output, size_t num_of_elements) {
  if (num_of_elements <= 0)
    return Status::OK();

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_elements, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CastKernelStd<GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread, OutT, InT><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      static_cast<int>(num_of_elements),
      CastStd<OutT, InT>());
  return Status::OK();
}

#if !defined(DISABLE_FLOAT4_TYPES)

template <int NumElementsPerThread, int NumThreadsPerBlock, bool is_odd, typename OutPairType, typename InPairType,
          typename OutSingleType, typename InSingleType>
__global__ void CudaCastPairwiseKernel(const InPairType* input, OutPairType* output,
                                       CUDA_LONG pair_count,
                                       CastStd<OutPairType, InPairType> pair_caster,
                                       CastStd<OutSingleType, InSingleType> singleton_caster) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < pair_count) {
      output[id] = pair_caster(input[id]);
      id += NumThreadsPerBlock;
    }
    if constexpr (is_odd) {
      if (id == pair_count) {
        *reinterpret_cast<OutSingleType*>(&output[id]) = singleton_caster(*reinterpret_cast<const InSingleType*>(&input[id]));
      }
    }
  }
}

template <class OutT, class InT>
Status CudaCastPairwise(cudaStream_t stream, const InT* input, OutT* output, size_t num_of_elements) {
  // There is no generic implementation - specialized implementation for the packed type(s) follow
  return Status::OK();
}

template <>
Status CudaCastPairwise(cudaStream_t stream, const Float4E2M1x2* input, float* output, size_t num_of_elements) {
  if (num_of_elements <= 0)
    return Status::OK();

  bool is_odd = (num_of_elements & 0x01) != 0;

  int pair_count = static_cast<int>(num_of_elements / 2);

  int blocksPerGrid = static_cast<int>(CeilDiv(pair_count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));

  // Have enough threads/blocks to process the last singleton element
  if (is_odd && (pair_count == blocksPerGrid * GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread)) {
    // This block will process last singleton element
    blocksPerGrid += 1;
  }

  if (is_odd) {
    CudaCastPairwiseKernel<GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread, true,
                           float2, Float4E2M1x2, float, Float4E2M1x2>
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            input, reinterpret_cast<float2*>(output), pair_count,
            CastStd<float2, Float4E2M1x2>(),
            CastStd<float, Float4E2M1x2>());
  } else {
    CudaCastPairwiseKernel<GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread, false,
                           float2, Float4E2M1x2, float, Float4E2M1x2>
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            input, reinterpret_cast<float2*>(output), pair_count,
            CastStd<float2, Float4E2M1x2>(),
            CastStd<float, Float4E2M1x2>());
  }

  return Status::OK();
}

template <>
Status CudaCastPairwise(cudaStream_t stream, const float* input, Float4E2M1x2* output, size_t num_of_elements) {
  if (num_of_elements <= 0)
    return Status::OK();

  bool is_odd = (num_of_elements & 0x01) != 0;

  int pair_count = static_cast<int>(num_of_elements / 2);

  int blocksPerGrid = static_cast<int>(CeilDiv(pair_count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));

  // Have enough threads/blocks to process the last singleton element
  if (is_odd && (pair_count == blocksPerGrid * GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread)) {
    // This block will process last singleton element
    blocksPerGrid += 1;
  }

  if (is_odd) {
    CudaCastPairwiseKernel<GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread, true,
                           Float4E2M1x2, float2, Float4E2M1x2, float>
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            reinterpret_cast<const float2*>(input), output, pair_count,
            CastStd<Float4E2M1x2, float2>(),
            CastStd<Float4E2M1x2, float>());
  } else {
    CudaCastPairwiseKernel<GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread, false,
                           Float4E2M1x2, float2, Float4E2M1x2, float>
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            reinterpret_cast<const float2*>(input), output, pair_count,
            CastStd<Float4E2M1x2, float2>(),
            CastStd<Float4E2M1x2, float>());
  }

  return Status::OK();
}

template Status CudaCastPairwise<float, Float4E2M1x2>(cudaStream_t stream, const Float4E2M1x2* input, float* output, size_t num_of_element);
template Status CudaCastPairwise<Float4E2M1x2, float>(cudaStream_t stream, const float* input, Float4E2M1x2* output, size_t num_of_element);

#endif

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
      saturate);
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

}  // namespace cast_helper_impl
}  // namespace cuda
}  // namespace onnxruntime
