// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "range_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace cuda {

template <typename T, unsigned TPB>
__global__ void RangeKernel(const T start, const T delta, const size_t n, T* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
      T value = start + delta * index;
      T value_delta = delta * TPB;
      for (int i = index; i < n; i += TPB, value += value_delta) {
        output[i] = value;
      }
  }
}

template <typename T>
bool RangeImpl(const T start, const T delta, const size_t n, T* output) {
  constexpr int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  RangeKernel<T, block_size><<<grid_size, block_size, 0>>>(start, delta, n, output);
  return CUDA_CALL(cudaPeekAtLastError());
}

#define SPECIALIZED_IMPL(T) \
  template bool RangeImpl<T>(const T start, const T delta, const size_t n, T* output);

SPECIALIZED_IMPL(int16_t)
SPECIALIZED_IMPL(int32_t)
SPECIALIZED_IMPL(int64_t)
SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)

}  // namespace cuda
}  // namespace onnxruntime
