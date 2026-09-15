// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "range_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void RangeKernel(const T start, const T delta, const int64_t count, T* output) {
  // Use a 64-bit grid-stride loop so counts larger than the launch grid (and larger than
  // INT_MAX) are handled correctly without index truncation or grid-dimension overflow.
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = index; i < count; i += stride) {
    output[i] = start + delta * static_cast<T>(i);
  }
}

template <typename T>
Status RangeImpl(cudaStream_t stream, const T start, const T delta, const int64_t count, T* output) {
  if (count <= 0) {
    return Status::OK();
  }
  constexpr int block_size = GridDim::maxThreadsPerBlock;
  int64_t num_blocks = (count + block_size - 1) / block_size;
  // CUDA limits the x-dimension of the launch grid to 2^31 - 1 blocks. Cap the grid to that
  // maximum; the grid-stride loop in RangeKernel covers any remaining elements when the count
  // requires more blocks than can be launched at once.
  constexpr int64_t kMaxGridDimX = 2147483647;
  int grid_size = static_cast<int>(num_blocks < kMaxGridDimX ? num_blocks : kMaxGridDimX);
  RangeKernel<T><<<grid_size, block_size, 0, stream>>>(start, delta, count, output);
  return CUDA_CALL(cudaGetLastError());
}

#define SPECIALIZED_IMPL(T) \
  template Status RangeImpl<T>(cudaStream_t stream, const T start, const T delta, const int64_t count, T* output);

SPECIALIZED_IMPL(int16_t)
SPECIALIZED_IMPL(int32_t)
SPECIALIZED_IMPL(int64_t)
SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)

}  // namespace cuda
}  // namespace onnxruntime
