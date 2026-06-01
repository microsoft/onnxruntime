// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/accumulation_type.h"
#include "core/providers/cuda/nn/lp_norm_impl.h"

namespace onnxruntime {
namespace cuda {

// Round up to the next power of two (for block-level reduction correctness).
inline int NextPowerOfTwo(int v) {
  --v;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

// Each block handles one normalization vector.
// norm_size: length along the normalization axis
// stride: stride between consecutive elements along the normalization axis
// AccT is the accumulation type (float for half, T otherwise).
template <typename T, typename AccT, int P>
__global__ void LpNormKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int64_t norm_size,
    const int64_t num_norms,
    const int64_t stride) {
  // Grid-stride loop so the kernel works even when num_norms exceeds the grid size.
  for (int64_t norm_idx = static_cast<int64_t>(blockIdx.x);
       norm_idx < num_norms;
       norm_idx += static_cast<int64_t>(gridDim.x)) {
    // Compute base offset for this normalization vector.
    // norm_idx = (outer_idx * stride + inner_idx) where inner_idx < stride
    const int64_t outer_idx = norm_idx / stride;
    const int64_t inner_idx = norm_idx % stride;
    const int64_t base = outer_idx * norm_size * stride + inner_idx;

    // Step 1: Each thread accumulates partial norm over its assigned elements
    // using a wider accumulation type to avoid overflow (e.g. half -> float).
    AccT thread_sum = AccT(0);
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < norm_size; i += static_cast<int64_t>(blockDim.x)) {
      AccT val = static_cast<AccT>(input[base + i * stride]);
      if constexpr (P == 1) {
        thread_sum += _Abs(val);
      } else {
        thread_sum += val * val;
      }
    }

    // Step 2: Block-level reduction using shared memory.
    // blockDim.x is always a power of two so the halving reduction is correct.
    extern __shared__ char shared_mem[];
    AccT* sdata = reinterpret_cast<AccT*>(shared_mem);
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
      }
      __syncthreads();
    }

    AccT norm = sdata[0];
    if constexpr (P == 2) {
      norm = _Sqrt(norm);
    }

    // Sync before Step 3 to prevent a fast thread in the next grid-stride iteration
    // from overwriting sdata[] while a slower thread still reads norm from sdata[0].
    __syncthreads();

    // Step 3: Normalize (division in accumulation type, cast back to T).
    if (norm != AccT(0)) {
      for (int64_t i = static_cast<int64_t>(threadIdx.x); i < norm_size; i += static_cast<int64_t>(blockDim.x)) {
        output[base + i * stride] = static_cast<T>(static_cast<AccT>(input[base + i * stride]) / norm);
      }
    } else {
      // Zero norm: output zeros to match the CPU kernel behavior (yVec.setZero()).
      // The ONNX spec would produce inf/nan here, but ORT intentionally returns 0.
      for (int64_t i = static_cast<int64_t>(threadIdx.x); i < norm_size; i += static_cast<int64_t>(blockDim.x)) {
        output[base + i * stride] = T(0);
      }
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
  using AccT = AccumulationType_t<T>;

  if (num_norms == 0) return;

  // Block size must be a power of two for the shared-memory reduction.
  const int raw_threads = static_cast<int>(std::min(static_cast<int64_t>(256), norm_size));
  const int threads_per_block = std::max(1, NextPowerOfTwo(raw_threads));
  // Cap grid size to avoid exceeding CUDA limits; the kernel uses a grid-stride loop.
  constexpr int64_t kMaxGridDim = 65535;
  const int blocks = static_cast<int>(std::min(num_norms, kMaxGridDim));
  const size_t shared_mem_size = static_cast<size_t>(threads_per_block) * sizeof(AccT);

  if (p == 1) {
    LpNormKernel<T, AccT, 1><<<blocks, threads_per_block, shared_mem_size, stream>>>(
        input, output, norm_size, num_norms, stride);
  } else {
    LpNormKernel<T, AccT, 2><<<blocks, threads_per_block, shared_mem_size, stream>>>(
        input, output, norm_size, num_norms, stride);
  }
}

// Explicit instantiations.
template void LpNormImpl<float>(cudaStream_t, const float*, float*, int64_t, int64_t, int64_t, int);
template void LpNormImpl<half>(cudaStream_t, const half*, half*, int64_t, int64_t, int64_t, int);

}  // namespace cuda
}  // namespace onnxruntime
