// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cast_op.cuh"

#include <limits>

#include "core/providers/cuda/cu_inc/common.cuh"

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
#include "cuda_fp8.h"
#endif

namespace onnxruntime {
namespace cuda {

template <typename OutT, typename InT>
struct Cast;

template <>
struct Cast<Float8E4M3FN, float> {
  __device__ __forceinline__ Float8E4M3FN operator()(float v) const {
    return Float8E4M3FN(__nv_cvt_float_to_fp8(v, __NV_NOSAT, __NV_E4M3));
  }
};

template <>
struct Cast<float, Float8E4M3FN> {
  __device__ __forceinline__ float operator()(Float8E4M3FN v) const {
    return __half2float(__nv_cvt_fp8_to_halfraw(v.val, __NV_E4M3));
  }
};

template <int NumThreadsPerBlock, int NumElementsPerThread, typename OutT, typename InT>
__global__ void CastKernel(const InT* input, OutT* output, CUDA_LONG N, Cast<OutT, InT> cast) {
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
Status CudaCast(cudaStream_t stream, const InT* input, OutT* output, size_t num_of_element) {
  if (num_of_element <= 0)
    return Status::OK();

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CastKernel<GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      static_cast<int>(num_of_element),
      Cast<OutT, InT>());
  return Status::OK();
}

template Status CudaCast<Float8E4M3FN, float>(cudaStream_t stream, const float* input, Float8E4M3FN* output, size_t num_of_element);
template Status CudaCast<float, Float8E4M3FN>(cudaStream_t stream, const Float8E4M3FN* input, float* output, size_t num_of_element);

}  // namespace cuda
}  // namespace onnxruntime
