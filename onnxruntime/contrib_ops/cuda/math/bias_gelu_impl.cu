// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/bias_gelu_impl.h"

#include "core/providers/cuda/cu_inc/common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr int kElementsPerThread = GridDim::maxElementsPerThread;
#ifdef USE_ROCM
constexpr int kThreadsPerBlock = 512;
#else
constexpr int kThreadsPerBlock = GridDim::maxThreadsPerBlock;
#endif

}  // namespace

template <typename T>
__global__ void BiasGeluKernel(int64_t bias_size, const T* X, const T* B, T* Y) {
  const auto kElementsPerBlock = kElementsPerThread * blockDim.x;
  const auto input_base_idx = bias_size * blockIdx.x + kElementsPerBlock * blockIdx.y + threadIdx.x;
  const auto bias_base_idx = kElementsPerBlock * blockIdx.y + threadIdx.x;
  const auto element_stride = blockDim.x;

  T reg_X[kElementsPerThread];
  T reg_B[kElementsPerThread];

  {
    auto input_idx = input_base_idx;
    auto bias_idx = bias_base_idx;
#pragma unroll
    for (int element_idx = 0; element_idx < kElementsPerThread; ++element_idx) {
      if (bias_idx < bias_size) {
        reg_X[element_idx] = X[input_idx];
        reg_B[element_idx] = B[bias_idx];
        input_idx += element_stride;
        bias_idx += element_stride;
      }
    }
  }

  {
    auto input_idx = input_base_idx;
    auto bias_idx = bias_base_idx;
#pragma unroll
    for (int element_idx = 0; element_idx < kElementsPerThread; ++element_idx) {
      if (bias_idx < bias_size) {
        Y[input_idx] = _Gelu(reg_X[element_idx] + reg_B[element_idx]);
        input_idx += element_stride;
        bias_idx += element_stride;
      }
    }
  }
}

template <typename T>
__global__ void VectorizedBiasGeluKernel(int64_t bias_size, const T* X, const T* B, T* Y) {
  const auto kElementsPerBlock = kElementsPerThread * blockDim.x;
  const auto bias_idx = kElementsPerBlock * blockIdx.y + kElementsPerThread * threadIdx.x;
  if (bias_idx >= bias_size) {
    return;
  }

  const auto input_idx = bias_size * blockIdx.x + kElementsPerBlock * blockIdx.y + kElementsPerThread * threadIdx.x;

  using LoadT = aligned_vector<T, kElementsPerThread>;

  T reg_X[kElementsPerThread];
  T reg_B[kElementsPerThread];
  T reg_Y[kElementsPerThread];

  LoadT* value_X = reinterpret_cast<LoadT*>(&reg_X);
  LoadT* value_B = reinterpret_cast<LoadT*>(&reg_B);
  *value_X = *reinterpret_cast<const LoadT*>(&X[input_idx]);
  *value_B = *reinterpret_cast<const LoadT*>(&B[bias_idx]);

#pragma unroll
  for (int element_idx = 0; element_idx < kElementsPerThread; ++element_idx) {
    reg_Y[element_idx] = _Gelu(reg_X[element_idx] + reg_B[element_idx]);
  }

  *(reinterpret_cast<LoadT*>(&Y[input_idx])) = *reinterpret_cast<LoadT*>(&reg_Y[0]);
}

template <typename T>
void LaunchBiasGeluKernel(cudaStream_t stream, int64_t input_size, int64_t bias_size, const T* X, const T* B, T* Y) {
  // given a 2D grid of blocks:
  // each grid column handles bias_size elements
  // there are input_size / bias_size columns.
  int num_threads_per_block = std::min<int>(static_cast<int>(CeilDiv(bias_size, kElementsPerThread)), kThreadsPerBlock);
  const auto grid_width = CeilDiv(bias_size, kElementsPerThread * num_threads_per_block);
  const auto grid_height = input_size / bias_size;
  const dim3 grid_dim{static_cast<uint32_t>(grid_height), static_cast<uint32_t>(grid_width)};

  constexpr int vec_alignment = std::alignment_of<aligned_vector<T, kElementsPerThread>>::value;
  if (bias_size % kElementsPerThread == 0 && reinterpret_cast<uint64_t>(X) % vec_alignment == 0 &&
      reinterpret_cast<uint64_t>(B) % vec_alignment == 0 && reinterpret_cast<uint64_t>(Y) % vec_alignment == 0) {
    VectorizedBiasGeluKernel<T><<<grid_dim, num_threads_per_block, 0, stream>>>(bias_size, X, B, Y);
  } else {
    BiasGeluKernel<T><<<grid_dim, num_threads_per_block, 0, stream>>>(bias_size, X, B, Y);
  }
}

// explicit instantiations
#define SPECIALIZED_BIAS_GELU_IMPL(T)                                                                           \
  template void LaunchBiasGeluKernel<T>(cudaStream_t stream, int64_t input_size, int64_t bias_size, const T* X, \
                                        const T* B, T* Y)

SPECIALIZED_BIAS_GELU_IMPL(half);
SPECIALIZED_BIAS_GELU_IMPL(float);
SPECIALIZED_BIAS_GELU_IMPL(double);
SPECIALIZED_BIAS_GELU_IMPL(BFloat16);

#undef SPECIALIZED_BIAS_GELU_IMPL

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
