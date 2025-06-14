// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/nn/group_norm_impl.h"

namespace onnxruntime {
namespace cuda {

constexpr int32_t kWarpSize = 32;
constexpr int32_t kBlockSize = 256;

template <typename T, typename U>
__global__ void GroupNormKernel(
    const T* input,
    const T* scale,
    const T* bias,
    T* output,
    int64_t batch_size,
    int64_t num_channels,
    int64_t spatial_size,
    int64_t num_groups,
    int64_t channels_per_group,
    int64_t stash_type,
    U epsilon) {
  
  const int64_t group_size = channels_per_group * spatial_size;
  const int64_t batch_idx = blockIdx.y;
  const int64_t group_idx = blockIdx.x;
  
  if (batch_idx >= batch_size || group_idx >= num_groups) {
    return;
  }
  
  // Calculate group start channel
  const int64_t group_start_channel = group_idx * channels_per_group;
  
  // Stage 1: Calculate mean and variance using float32 precision when stash_type=1
  __shared__ float shared_sum[kBlockSize];
  __shared__ float shared_sum_sq[kBlockSize];
  
  float sum = 0.0f;
  float sum_sq = 0.0f;
  
  // Each thread processes multiple elements
  for (int64_t idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
    const int64_t channel_offset = idx / spatial_size;
    const int64_t spatial_offset = idx % spatial_size;
    const int64_t channel_idx = group_start_channel + channel_offset;
    
    if (channel_idx < num_channels) {
      const int64_t input_idx = batch_idx * num_channels * spatial_size + 
                               channel_idx * spatial_size + spatial_offset;
      
      // Cast to float for precision as per stash_type=1 specification
      const float val = static_cast<float>(input[input_idx]);
      sum += val;
      sum_sq += val * val;
    }
  }
  
  shared_sum[threadIdx.x] = sum;
  shared_sum_sq[threadIdx.x] = sum_sq;
  __syncthreads();
  
  // Reduce sums within block
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
      shared_sum_sq[threadIdx.x] += shared_sum_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }
  
  float group_mean = 0.0f;
  float group_var = 0.0f;
  float inv_std = 0.0f;
  
  if (threadIdx.x == 0) {
    group_mean = shared_sum[0] / static_cast<float>(group_size);
    group_var = shared_sum_sq[0] / static_cast<float>(group_size) - group_mean * group_mean;
    inv_std = rsqrtf(group_var + static_cast<float>(epsilon));
  }
  
  // Broadcast mean and inv_std to all threads
  __shared__ float broadcast_mean;
  __shared__ float broadcast_inv_std;
  
  if (threadIdx.x == 0) {
    broadcast_mean = group_mean;
    broadcast_inv_std = inv_std;
  }
  __syncthreads();
  
  group_mean = broadcast_mean;
  inv_std = broadcast_inv_std;
  
  // Stage 2: Apply normalization with scale and bias
  // y = scale * (x - mean) / sqrt(variance + epsilon) + bias
  for (int64_t idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
    const int64_t channel_offset = idx / spatial_size;
    const int64_t spatial_offset = idx % spatial_size;
    const int64_t channel_idx = group_start_channel + channel_offset;
    
    if (channel_idx < num_channels) {
      const int64_t input_idx = batch_idx * num_channels * spatial_size + 
                               channel_idx * spatial_size + spatial_offset;
      
      // Normalize using float32 precision as per stash_type=1
      const float x_float = static_cast<float>(input[input_idx]);
      const float normalized = (x_float - group_mean) * inv_std;
      
      // Apply scale and bias in original type precision
      const float scale_val = static_cast<float>(scale[channel_idx]);
      const float bias_val = static_cast<float>(bias[channel_idx]);
      const float result = normalized * scale_val + bias_val;
      
      output[input_idx] = static_cast<T>(result);
    }
  }
}

template <typename T, typename U>
Status GroupNormImpl(
    cudaStream_t stream,
    const T* input_data,
    const T* scale_data,
    const T* bias_data,
    T* output_data,
    int64_t batch_size,
    int64_t num_channels,
    int64_t spatial_size,
    int64_t num_groups,
    int64_t stash_type,
    double epsilon) {
  
  const int64_t channels_per_group = num_channels / num_groups;
  
  // Launch kernel with batch_size x num_groups grid
  dim3 grid(static_cast<unsigned int>(num_groups), static_cast<unsigned int>(batch_size));
  dim3 block(kBlockSize);
  
  GroupNormKernel<<<grid, block, 0, stream>>>(
      input_data,
      scale_data,
      bias_data,
      output_data,
      batch_size,
      num_channels,
      spatial_size,
      num_groups,
      channels_per_group,
      stash_type,
      static_cast<U>(epsilon));
  
  return CUDA_CALL(cudaGetLastError());
}

// Explicit template instantiations
template Status GroupNormImpl<float, float>(cudaStream_t, const float*, const float*, const float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t, double);
template Status GroupNormImpl<double, float>(cudaStream_t, const double*, const double*, const double*, double*, int64_t, int64_t, int64_t, int64_t, int64_t, double);
template Status GroupNormImpl<half, float>(cudaStream_t, const half*, const half*, const half*, half*, int64_t, int64_t, int64_t, int64_t, int64_t, double);
template Status GroupNormImpl<__nv_bfloat16, float>(cudaStream_t, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int64_t, int64_t, int64_t, int64_t, int64_t, double);

}  // namespace cuda
}  // namespace onnxruntime