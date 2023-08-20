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
__global__ void Dequantize4BitsKernel(T* output, const uint8_t* quant_data, int k, int k_block, int block_size, bool has_zero_point, int block_blob_size);

template <>
__global__ void Dequantize4BitsKernel<half>(half* output, const uint8_t* quant_data, int k, int k_block, int block_size, bool has_zero_point, int block_blob_size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int n_idx = id / k_block;
  int k_block_idx = id % k_block;
  int k_idx = k_block_idx * block_size;
  const uint8_t* quant_data_cur = quant_data + id * block_blob_size;
  uint16_t scale_low = *(quant_data_cur++);
  uint16_t scale_high = *(quant_data_cur++);
  uint16_t scale_data = scale_high << 8 | scale_low;
  half scale = *(reinterpret_cast<half*>(&scale_data));
  half zero_point = static_cast<half>(8.0);
  if (has_zero_point) {
    zero_point = static_cast<half>(float(*(quant_data_cur++)));
  }

  int output_offset = n_idx * k + k_idx;
  for (int i = 0; i < block_size / 2; i++) {
    uint8_t value = *(quant_data_cur++);
    half x0 = static_cast<half>(float(value & 0xF));
    half x1 = static_cast<half>(float(value >> 4));
    if (k_idx++ < k) {
      x0 = scale * (x0 - zero_point);
      output[output_offset++] = x0;
    }
    if (k_idx++ < k) {
      x1 = scale * (x1 - zero_point);
      output[output_offset++] = x1;
    }
  }
}

template <>
__global__ void Dequantize4BitsKernel<float>(float* output, const uint8_t* quant_data, int k, int k_block, int block_size, bool has_zero_point, int block_blob_size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int n_idx = id / k_block;
  int k_block_idx = id % k_block;
  int k_idx = k_block_idx * block_size;
  const uint8_t* quant_data_cur = quant_data + id * block_blob_size;
  uint32_t scale_0 = *(quant_data_cur++);
  uint32_t scale_1 = *(quant_data_cur++);
  uint32_t scale_2 = *(quant_data_cur++);
  uint32_t scale_3 = *(quant_data_cur++);
  uint32_t scale_data = scale_3 << 24 | scale_2 << 16 | scale_1 << 8 | scale_0;
  float scale = *(reinterpret_cast<float*>(&scale_data));
  float zero_point = static_cast<float>(8.0);
  if (has_zero_point) {
    zero_point = float(*(quant_data_cur++));
  }

  output = output + n_idx * k;
  for (int i = k_idx; i < k_idx + block_size / 2; i++) {
    uint8_t value = *(quant_data_cur++);
    float x0 = static_cast<float>(float(value & 0xF));
    float x1 = static_cast<float>(float(value >> 4));
    if (i < k) {
      x0 = scale * (x0 - zero_point);
      output[i] = x0;
    }
    if (i + block_size / 2 < k) {
      x1 = scale * (x1 - zero_point);
      output[i + block_size / 2] = x1;
    }
  }
}

template <class T>
Status Dequantize4Bits(T* output, const uint8_t* quant_data, int k, int n, int block_size, bool has_zero_point, cudaStream_t stream) {
  int k_blocks = (k + block_size - 1) / block_size;
  int blocks_per_grid = static_cast<int>(CeilDiv(n * k_blocks, GridDim::maxThreadsPerBlock));
  int k_block = (k + block_size - 1) / block_size;
  int block_blob_size = block_size / 2 + sizeof(T) + (has_zero_point ? 1 : 0);

  Dequantize4BitsKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(output, quant_data, k, k_block, block_size, has_zero_point, block_blob_size);

  return Status::OK();
}

template Status Dequantize4Bits<float>(float* output, const uint8_t* quant_data, int k, int n, int block_size, bool has_zero_point, cudaStream_t stream);
template Status Dequantize4Bits<half>(half* output, const uint8_t* quant_data, int k, int n, int block_size, bool has_zero_point, cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
