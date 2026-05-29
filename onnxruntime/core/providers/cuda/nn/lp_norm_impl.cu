// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "lp_norm_impl.h"

namespace onnxruntime {
namespace cuda {

// Each block handles one normalization vector.
// norm_size: length along the normalization axis
// stride: stride between consecutive elements along the normalization axis
template <typename T, int P>
__global__ void LpNormKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int64_t norm_size,
    const int64_t num_norms,
    const int64_t stride) {
  const int64_t norm_idx = static_cast<int64_t>(blockIdx.x);
  if (norm_idx >= num_norms) return;

  // Compute base offset for this normalization vector.
  // norm_idx = (outer_idx * stride + inner_idx) where inner_idx < stride
  const int64_t outer_idx = norm_idx / stride;
  const int64_t inner_idx = norm_idx % stride;
  const int64_t base = outer_idx * norm_size * stride + inner_idx;

  // Step 1: Each thread accumulates partial norm over its assigned elements.
  T thread_sum = T(0);
  for (int64_t i = static_cast<int64_t>(threadIdx.x); i < norm_size; i += static_cast<int64_t>(blockDim.x)) {
    T val = input[base + i * stride];
    if constexpr (P == 1) {
      thread_sum += _Abs(val);
    } else {
      thread_sum += val * val;
    }
  }

  // Step 2: Block-level reduction using shared memory.
  extern __shared__ char shared_mem[];
  T* sdata = reinterpret_cast<T*>(shared_mem);
  sdata[threadIdx.x] = thread_sum;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }

  T norm = sdata[0];
  if constexpr (P == 2) {
    norm = _Sqrt(norm);
  }

  // Step 3: Normalize.
  if (norm != T(0)) {
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < norm_size; i += static_cast<int64_t>(blockDim.x)) {
      output[base + i * stride] = input[base + i * stride] / norm;
    }
  } else {
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < norm_size; i += static_cast<int64_t>(blockDim.x)) {
      output[base + i * stride] = T(0);
    }
  }
}

template <typename T>
void LpNormImpl(
    cudaStream_t stream,
    const T* input,
    T* output,
    int64_t norm_size,
    int64_t num_norms,
    int64_t stride,
    int p) {
  if (num_norms == 0) return;

  const int threads_per_block = std::min(static_cast<int64_t>(256), norm_size);
  const int blocks = static_cast<int>(num_norms);
  const size_t shared_mem_size = threads_per_block * sizeof(T);

  if (p == 1) {
    LpNormKernel<T, 1><<<blocks, threads_per_block, shared_mem_size, stream>>>(
        input, output, norm_size, num_norms, stride);
  } else {
    LpNormKernel<T, 2><<<blocks, threads_per_block, shared_mem_size, stream>>>(
        input, output, norm_size, num_norms, stride);
  }
}

// Explicit instantiations.
template void LpNormImpl<float>(cudaStream_t, const float*, float*, int64_t, int64_t, int64_t, int);
template void LpNormImpl<double>(cudaStream_t, const double*, double*, int64_t, int64_t, int64_t, int);
template void LpNormImpl<half>(cudaStream_t, const half*, half*, int64_t, int64_t, int64_t, int);

}  // namespace cuda
}  // namespace onnxruntime
