// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "matmul_with_quant_weight.cuh"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <class T, int block_size>
__global__ void MatMul4BitsWeightKernel(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k) {
  int k_block_id = blockIdx.x;
  int n_block_id = blockIdx.y;
  int m_id = blockIdx.z;
  int n_id = n_block_id * blockDim.x + threadIdx.x;
  int k_id = k_block_id * block_size;

  if (n_id >= n) {
    return;
  }

  // load A to share
  __shared__ T a_data_vec[block_size];
  a_data += m_id * k + k_id;

  for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
    if (i + k_id < k) {
      a_data_vec[i] = a_data[i];
    } else {
      a_data_vec[i] = static_cast<T>(0.f);
    }
  }
  __syncthreads();
  // dequantize a blob
  T weight[block_size];
  const uint8_t* quant_data_cur = b_data_quant + (n_id * gridDim.x + k_block_id) * block_size / 2;
  T scale = *(scales_data + n_id * gridDim.x + k_block_id);
  T zero_point = static_cast<T>(zero_points ? float(zero_points[n_id * gridDim.x + k_block_id]) : 8.f);
  for (int kk = 0; kk < block_size; kk += 2) {
    uint8_t value = *(quant_data_cur++);
    T x0 = static_cast<T>(float(value & 0xF));
    T x1 = static_cast<T>(float(value >> 4));
    if (kk + k_id < k) {
      x0 = scale * (x0 - zero_point);
      weight[kk] = x0;
    }
    if (kk + 1 + k_id < k) {
      x1 = scale * (x1 - zero_point);
      weight[kk + 1] = x1;
    }
  }

  T res = static_cast<T>(0.f);
  for (int kk = 0; kk < block_size; kk++) {
    res += weight[kk] * a_data_vec[kk];
  }

  atomicAdd(output + m_id * n + n_id, res);
}

template <class T>
Status MatMul4BitsWeight(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    cudaStream_t stream) {
  constexpr int n_block_size = 128;
  dim3 blocks((k + block_size - 1) / block_size, (n + n_block_size - 1) / n_block_size, m);
  dim3 threads(min(n, n_block_size));

  if (16 == block_size) {
    MatMul4BitsWeightKernel<T, 16><<<blocks, threads, 0, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k);
  } else if (32 == block_size) {
    MatMul4BitsWeightKernel<T, 32><<<blocks, threads, 0, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k);
  } else if (64 == block_size) {
    MatMul4BitsWeightKernel<T, 64><<<blocks, threads, 0, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k);
  } else if (128 == block_size) {
    MatMul4BitsWeightKernel<T, 128><<<blocks, threads, 0, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "block size ", block_size, " is not supported");
  }

  return Status::OK();
}

template Status MatMul4BitsWeight<float>(
    float* output,
    const float* a_data,
    const uint8_t* b_data_quant,
    const float* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    cudaStream_t stream);

template Status MatMul4BitsWeight<half>(
    half* output,
    const half* a_data,
    const uint8_t* b_data_quant,
    const half* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
