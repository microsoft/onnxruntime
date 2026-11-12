// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/math/softmax_warpwise_impl.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T, template <typename> class Reduction>
__device__ T SoftmaxGradientBlockReduce(T value) {
  using BlockReduce = cub::BlockReduce<T, 256>;
  __shared__ typename BlockReduce::TempStorage storage;
  __shared__ T result;
  __syncthreads();
  const T reduced = BlockReduce(storage).Reduce(value, Reduction<T>());
  if (threadIdx.x == 0) result = reduced;
  __syncthreads();
  return result;
}

// Keep the dominant gradient as a small residual instead of subtracting rounded probabilities from one.
template <typename GradT, typename OutputT, typename AccT, int Batch, int Iterations, int WarpSize,
          bool IsLogSoftmax, int Unroll = 1>
__device__ void StableDominantGradient(const GradT (&grad)[Batch][Iterations], const OutputT (&output)[Batch][Iterations],
                                       AccT (&sum)[Batch], int count, int (&pivot)[Batch],
                                       AccT (&pivot_grad)[Batch], AccT (&pivot_dy)[Batch]) {
  for (int i = 0; i < Batch; ++i) {
    pivot[i] = count;
    for (int it = 0; it < Iterations; ++it) {
      const int index = (it / Unroll * WarpSize + threadIdx.x) * Unroll + it % Unroll;
      const AccT value = static_cast<AccT>(output[i][it]);
      if (index < count && value > (IsLogSoftmax ? AccT(-0.6931471805599453) : AccT(0.5))) {
        pivot[i] = min(pivot[i], index);
      }
    }
  }
  warp_reduce<int, Batch, WarpSize, Min>(pivot);
  for (int i = 0; i < Batch; ++i) {
    pivot_dy[i] = 0;
    pivot_grad[i] = 0;
  }
  AccT residual[Batch]{};
  if constexpr (IsLogSoftmax) {
    for (int i = 0; i < Batch; ++i) {
      for (int it = 0; it < Iterations; ++it) {
        const int index = (it / Unroll * WarpSize + threadIdx.x) * Unroll + it % Unroll;
        if (index < count && index != pivot[i]) {
          residual[i] += static_cast<AccT>(grad[i][it]) -
                         std::exp(static_cast<AccT>(output[i][it])) * sum[i];
        }
      }
    }
    warp_reduce<AccT, Batch, WarpSize, Add>(residual);
    for (int i = 0; i < Batch; ++i) pivot_grad[i] = -residual[i];
    return;
  }

  for (int i = 0; i < Batch; ++i) {
    for (int it = 0; it < Iterations; ++it) {
      const int index = (it / Unroll * WarpSize + threadIdx.x) * Unroll + it % Unroll;
      if (index == pivot[i] && index < count) {
        pivot_dy[i] = static_cast<AccT>(grad[i][it]);
      }
    }
  }
  warp_reduce<AccT, Batch, WarpSize, Add>(pivot_dy);
  for (int i = 0; i < Batch; ++i) {
    for (int it = 0; it < Iterations; ++it) {
      const int index = (it / Unroll * WarpSize + threadIdx.x) * Unroll + it % Unroll;
      if (index < count && index != pivot[i]) {
        residual[i] += static_cast<AccT>(output[i][it]) *
                       (static_cast<AccT>(grad[i][it]) - pivot_dy[i]);
      }
    }
  }
  warp_reduce<AccT, Batch, WarpSize, Add>(residual);
  for (int i = 0; i < Batch; ++i) {
    if (pivot[i] < count) sum[i] = residual[i];
    residual[i] = 0;
    for (int it = 0; it < Iterations; ++it) {
      const int index = (it / Unroll * WarpSize + threadIdx.x) * Unroll + it % Unroll;
      if (index < count && index != pivot[i]) {
        residual[i] += static_cast<AccT>(output[i][it]) *
                       ((static_cast<AccT>(grad[i][it]) - pivot_dy[i]) - sum[i]);
      }
    }
  }
  warp_reduce<AccT, Batch, WarpSize, Add>(residual);
  for (int i = 0; i < Batch; ++i) pivot_grad[i] = -residual[i];
}

}  // namespace cuda
}  // namespace onnxruntime
