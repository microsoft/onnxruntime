// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "block_quantization_impl.cuh"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

#include <cub/cub.cuh>

namespace onnxruntime {
namespace contrib {
namespace cuda {

typedef struct __align__(8) {
  half x;
  half y;
  half z;
  half w;
}
half4;

__device__ __forceinline__ float warpReduceMaxMultipleGroup(float max_value, const int width) {
  constexpr unsigned FULLMASK = 0xFFFFFFFF;
  switch (width) {
    case 32:
      max_value = max(max_value, __shfl_xor_sync(FULLMASK, max_value, 16));
    case 16:
      max_value = max(max_value, __shfl_xor_sync(FULLMASK, max_value, 8));
    case 8:
      max_value = max(max_value, __shfl_xor_sync(FULLMASK, max_value, 4));
    case 4:
      max_value = max(max_value, __shfl_xor_sync(FULLMASK, max_value, 2));
    case 2:
      max_value = max(max_value, __shfl_xor_sync(FULLMASK, max_value, 1));
    default:
      break;
  }
  return max_value;
}

static constexpr int INT8MAX = 127;

// vectorized read/write value in 4 each thread.
// when datablock size <= 128, each warp will handle one full datablock or more than one full datablocks.
template <typename T, typename TInVec4>
__global__ void
BlockQuantizeKernelInWarp(const T* x, unsigned datablock_size, unsigned datablock_count, T* scale, int8_t* y) {
  const int64_t offset = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  const int64_t datablock_id = offset / datablock_size;
  if (datablock_id >= datablock_count) return;

  TInVec4 x4 = *(const TInVec4*)(x + offset);
  float max_abs_val = fabsf((float)x4.x);
  max_abs_val = fmaxf(max_abs_val, fabsf((float)x4.y));
  max_abs_val = fmaxf(max_abs_val, fabsf((float)x4.z));
  max_abs_val = fmaxf(max_abs_val, fabsf((float)x4.w));

  const int width = static_cast<int>(datablock_size >> 2);
  max_abs_val = warpReduceMaxMultipleGroup(max_abs_val, width);
  const float block_scale_value = max_abs_val / INT8MAX;

  char4 y4{0, 0, 0, 0};
  if (block_scale_value) {
    y4.x = static_cast<char>(__float2int_rn((float)x4.x / block_scale_value));
    y4.y = static_cast<char>(__float2int_rn((float)x4.y / block_scale_value));
    y4.z = static_cast<char>(__float2int_rn((float)x4.z / block_scale_value));
    y4.w = static_cast<char>(__float2int_rn((float)x4.w / block_scale_value));
  }
  *(char4*)(y + offset) = y4;

  if (offset == datablock_id * datablock_size) {  // first thread for a datablock
    scale[datablock_id] = (T)block_scale_value;
  }
}

// vectorized read/write value in 4 each thread.
// when datablock size >= 256, one cuda block will handle on datablock.
template <typename T, typename TInVec4, int TPB>
__global__ void
BlockQuantizeKernelCrossWarp(const T* x, unsigned datablock_size, T* scale, int8_t* y) {
  if (threadIdx.x * 4 >= datablock_size) return;
  const int64_t datablock_id = blockIdx.x;
  const int64_t offset = (datablock_id * datablock_size + threadIdx.x) * 4;

  __shared__ float block_scale_value;
  TInVec4 x4 = *(const TInVec4*)(x + offset);
  float max_abs_val = 0.0f;
  max_abs_val = fmaxf(max_abs_val, fabsf((float)x4.x));
  max_abs_val = fmaxf(max_abs_val, fabsf((float)x4.y));
  max_abs_val = fmaxf(max_abs_val, fabsf((float)x4.z));
  max_abs_val = fmaxf(max_abs_val, fabsf((float)x4.w));

  typedef cub::BlockReduce<float, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  float block_max_abs_val = BlockReduce(temp_storage).Reduce(max_abs_val, cub::Max());
  if (threadIdx.x == 0) {
    block_scale_value = block_max_abs_val / INT8MAX;
  }
  __syncthreads();

  char4 y4{0, 0, 0, 0};
  if (block_scale_value) {
    y4.x = static_cast<char>(__float2int_rn((float)x4.x / block_scale_value));
    y4.y = static_cast<char>(__float2int_rn((float)x4.y / block_scale_value));
    y4.z = static_cast<char>(__float2int_rn((float)x4.z / block_scale_value));
    y4.w = static_cast<char>(__float2int_rn((float)x4.w / block_scale_value));
  }
  *(char4*)(y + offset) = y4;

  if (offset == datablock_id * datablock_size) {  // first thread for a datablock
    scale[datablock_id] = (T)block_scale_value;
  }
}

template <>
Status CudaBlockQuantize(
    cudaStream_t stream,
    const cudaDeviceProp& /*device_prop*/,
    const half* x,
    unsigned const datablock_size,
    unsigned const datablock_count,
    half* scale,
    int8_t* y) {
  if (datablock_size <= 128) {
    constexpr unsigned TPB = 256;
    constexpr unsigned EPB = TPB * 4;
    const unsigned cuda_blocks = ((int64_t)(datablock_size)*datablock_count + EPB - 1) / EPB;
    BlockQuantizeKernelInWarp<half, half4><<<cuda_blocks, TPB, 0, stream>>>(
        x, datablock_size, datablock_count, scale, y);
  } else if (datablock_size == 256) {
    constexpr unsigned TPB = 256 / 4;
    BlockQuantizeKernelCrossWarp<half, half4, TPB><<<datablock_count, TPB, 0, stream>>>(
        x, datablock_size, scale, y);
  } else if (datablock_size == 512) {
    constexpr unsigned TPB = 512 / 4;
    BlockQuantizeKernelCrossWarp<half, half4, TPB><<<datablock_count, TPB, 0, stream>>>(
        x, datablock_size, scale, y);
  } else if (datablock_size == 1024) {
    constexpr unsigned TPB = 1024 / 4;
    BlockQuantizeKernelCrossWarp<half, half4, TPB><<<datablock_count, TPB, 0, stream>>>(
        x, datablock_size, scale, y);
  } else {
    ORT_ENFORCE(false, "Datablock size too large, currently not supported!");
  }

  return CUDA_CALL(cudaGetLastError());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
