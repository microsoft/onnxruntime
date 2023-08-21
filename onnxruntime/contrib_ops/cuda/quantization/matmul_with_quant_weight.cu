// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "attention_quantization_impl.cuh"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <class T>
__global__ void Dequantize4BitsKernel(
    T* output,
    const uint8_t* quant_data,
    const T* scale_data,
    const uint8_t* zero_points,
    int k,
    int k_block,
    int block_size,
    int block_blob_size);

template <>
__global__ void Dequantize4BitsKernel<half>(
    half* output,
    const uint8_t* quant_data,
    const half* scale_data,
    const uint8_t* zero_points,
    int k,
    int k_block,
    int block_size,
    int block_blob_size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int n_idx = id / k_block;
  int k_block_idx = id % k_block;
  int k_idx = k_block_idx * block_size;
  const uint8_t* quant_data_cur = quant_data + id * block_blob_size;
  half scale = scale_data[id];
  half zero_point = zero_points ? static_cast<half>(float(zero_points[id])) : static_cast<half>(8.0);

  output = output + n_idx * k;
  for (int i = k_idx; i < k_idx + block_size; i += 2) {
    uint8_t value = *(quant_data_cur++);
    half x0 = static_cast<half>(float(value & 0xF));
    half x1 = static_cast<half>(float(value >> 4));
    if (i < k) {
      x0 = scale * (x0 - zero_point);
      output[i] = x0;
    }
    if (i + 1 < k) {
      x1 = scale * (x1 - zero_point);
      output[i + 1] = x1;
    }
  }
}

template <>
__global__ void Dequantize4BitsKernel<float>(
    float* output,
    const uint8_t* quant_data,
    const float* scale_data,
    const uint8_t* zero_points,
    int k,
    int k_block,
    int block_size,
    int block_blob_size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int n_idx = id / k_block;
  int k_block_idx = id % k_block;
  int k_idx = k_block_idx * block_size;
  const uint8_t* quant_data_cur = quant_data + id * block_blob_size;
  float scale = *(scale_data + id);
  float zero_point = zero_points ? float(zero_points[id]) : static_cast<float>(8.0);

  output = output + n_idx * k;
  for (int i = k_idx; i < k_idx + block_size; i+=2) {
    uint8_t value = *(quant_data_cur++);
    float x0 = static_cast<float>(float(value & 0xF));
    float x1 = static_cast<float>(float(value >> 4));
    if (i < k) {
      x0 = scale * (x0 - zero_point);
      output[i] = x0;
    }
    if (i + 1 < k) {
      x1 = scale * (x1 - zero_point);
      output[i + 1] = x1;
    }
  }
}

template <class T>
Status Dequantize4Bits(
    T* output,
    const uint8_t* quant_data,
    const T* scales_data,
    const uint8_t* zero_points,
    int k,
    int n,
    int block_size,
    cudaStream_t stream) {
  int k_blocks = (k + block_size - 1) / block_size;
  int blocks_per_grid = static_cast<int>(CeilDiv(n * k_blocks, GridDim::maxThreadsPerBlock));
  int block_blob_size = block_size / 2;

  Dequantize4BitsKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      output,
      quant_data,
      scales_data,
      zero_points,
      k,
      k_blocks,
      block_size,
      block_blob_size);

  return Status::OK();
}

template Status Dequantize4Bits<float>(
    float* output,
    const uint8_t* quant_data,
    const float* scales_data,
    const uint8_t* zero_points,
    int k,
    int n,
    int block_size,
    cudaStream_t stream);

template Status Dequantize4Bits<half>(
    half* output,
    const uint8_t* quant_data,
    const half* scales_data,
    const uint8_t* zero_points,
    int k,
    int n,
    int block_size,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
