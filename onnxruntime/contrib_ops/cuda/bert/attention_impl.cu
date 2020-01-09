/*
 The implementation of this file is based on qkvToContext plugin in TensorRT demo:
 https://github.com/NVIDIA/TensorRT/tree/release/5.1/demo/BERT/

Copyright 2019 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "attention_impl.h"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

static size_t AlignTo(size_t a, size_t b) {
  return CeilDiv(a, b) * b;
}

size_t ScratchSize(size_t element_size, int batch_size, int num_heads, int sequence_length) {
  const size_t len = batch_size * num_heads * sequence_length * sequence_length;
  const size_t bytes = len * element_size;

  const size_t alignment = 256;
  const size_t bytesAligned = AlignTo(bytes, alignment);
  return bytesAligned;
}

size_t GetAttentionWorkspaceSize(size_t element_size, int batch_size, int num_heads, int head_size, int sequence_length) {
  size_t qkv_size = 3 * batch_size * sequence_length * num_heads * head_size * element_size;
  return qkv_size + 2 * ScratchSize(element_size, batch_size, num_heads, sequence_length);
}

template <typename T, unsigned TPB>
__device__ inline void Softmax(const int ld, const int num_valid, const T* input, T* output) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  float thread_data_max(-CUDART_INF_F);

  // e^x is represented as infinity if x is large enough, like 100.f.
  // Infinity divided by Infinity is a NAN. Thus, softmax gets a NAN if one or more item are large enough.
  // a math transform as below is leveraged to get a stable softmax:
  // e^xi/(e^x1 + ...e^xn) = e^(xi - max) / (e^(x1 - max) + ... + e^(xn - max))
  const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * ld;
  for (int i = threadIdx.x; i < num_valid; i += TPB) {
    const int index = offset + i;
    if (thread_data_max < float(input[index])) {
      thread_data_max = float(input[index]);
    }
  }

  const auto max = BlockReduce(tmp_storage).Reduce(thread_data_max, cub::Max());

  // Store max value
  if (threadIdx.x == 0) {
    max_block = max;
  }
  __syncthreads();

  float thread_data_sum(0.f);
  for (int i = threadIdx.x; i < num_valid; i += TPB) {
    const int index = offset + i;
    const float val = input[index];
    thread_data_sum += expf(val - max_block);
  }

  const auto sum = BlockReduce(tmp_storage).Reduce(thread_data_sum, cub::Sum());
  if (threadIdx.x == 0) {
    sum_reverse_block = 1.f / sum;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int index = offset + i;
    const float val = (i < num_valid) ? expf(float(input[index]) - max_block) * sum_reverse_block : 0.f;
    output[index] = T(val);
  }
}

template <typename T, unsigned TPB>
__device__ inline void SoftmaxSmall(const int ld, const int num_valid, const T* input, T* output) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * ld;
  const int index = offset + threadIdx.x;

  // e^x is represented as infinity if x is large enough, like 100.f.
  // Infinity divided by Infinity is a NAN. Thus, softmax gets a NAN if one or more item are large enough.
  // a math transform as below is leveraged to get a stable softmax:
  // e^xi/(e^x1 + ...e^xn) = e^(xi - max) / (e^(x1 - max) + ... + e^(xn - max))
  float thread_data_max(-CUDART_INF_F);
  if (threadIdx.x < num_valid) {
    thread_data_max = input[index];
  }

  const auto max = BlockReduce(tmp_storage).Reduce(thread_data_max, cub::Max(), num_valid);

  // Store max value
  if (threadIdx.x == 0) {
    max_block = max;
  }
  __syncthreads();

  float thread_data_exp(0.f);
  if (threadIdx.x < num_valid) {
    const float val = input[index];
    thread_data_exp = expf(val - max_block);
  }

  const auto sum = BlockReduce(tmp_storage).Reduce(thread_data_exp, cub::Sum(), num_valid);

  // Store max value
  if (threadIdx.x == 0) {
    sum_reverse_block = (1.f) / sum;
  }
  __syncthreads();

  if (threadIdx.x < ld) {
    // this will be 0 for threadIdx.x >= num_valid
    output[index] = T(thread_data_exp * sum_reverse_block);
  }
}

template <typename T, unsigned TPB>
__global__ void MaskedSoftmaxKernelSmall(const int sequence_length, const int* mask_index, const T* input, T* output) {
  __shared__ int num_valid;

  if (threadIdx.x == 0) {
    num_valid = min(sequence_length, mask_index[blockIdx.y]);
  }
  __syncthreads();

  SoftmaxSmall<T, TPB>(sequence_length, num_valid, input, output);
}

template <typename T, unsigned TPB>
__global__ void MaskedSoftmaxKernel(const int sequence_length, const int* mask_index, const T* input, T* output) {
  __shared__ int num_valid;

  if (threadIdx.x == 0) {
    num_valid = min(sequence_length, mask_index[blockIdx.y]);
  }
  __syncthreads();

  Softmax<T, TPB>(sequence_length, num_valid, input, output);
}

template <typename T>
bool ComputeMaskedSoftmax(cudaStream_t stream, const int sequence_length, const int batch_size, const int num_heads,
                          const int* mask_index, const T* input, T* output) {
  // Mask is of length batch_size and assumes the valid region is contiguous starting
  // from the beginning of the sequence

  const dim3 grid(sequence_length * num_heads, batch_size, 1);

  if (sequence_length <= 32) {
    const int blockSize = 32;
    MaskedSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(sequence_length, mask_index, input, output);
  } else if (sequence_length <= 128) {
    const int blockSize = 128;
    MaskedSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(sequence_length, mask_index, input, output);
  } else if (sequence_length == 384) {
    const int blockSize = 384;
    MaskedSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(sequence_length, mask_index, input, output);
  } else {
    const int blockSize = 256;
    MaskedSoftmaxKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(sequence_length, mask_index, input, output);
  }

  return CUDA_CALL(cudaPeekAtLastError());
}

template <typename T>
__global__ void TransposeCtx(const int H, const T* input, T* output) {
  // Input:  BxNxSxH
  // Output: BxSxNxH

  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;

  int num_heads = blockDim.y;
  int sequence_length = gridDim.x;

  const int NH = num_heads * H;
  const int NHS = NH * sequence_length;
  const int in_offset = s * H + n * sequence_length * H + b * NHS;
  const int out_offset = n * H + s * NH + b * NHS;

  const int i = threadIdx.x;
  if (i < H) {
    output[out_offset + i] = input[in_offset + i];
  }
}

bool LaunchTransCtx(cudaStream_t stream,
                    const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                    const float* input, float* output) {
  const dim3 grid(sequence_length, batch_size, 1);
  if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    const float2* input2 = reinterpret_cast<const float2*>(input);
    float2* output2 = reinterpret_cast<float2*>(output);
    const dim3 block(H, num_heads, 1);
    TransposeCtx<float2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else {
    const dim3 block(head_size, num_heads, 1);
    TransposeCtx<float><<<grid, block, 0, stream>>>(head_size, input, output);
  }
  return CUDA_CALL(cudaPeekAtLastError());
}

bool LaunchTransCtx(cudaStream_t stream,
                    const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                    const half* input, half* output) {
  const dim3 grid(sequence_length, batch_size, 1);
  if (0 == (head_size % 4)) {
    const int H = head_size / 4;
    const dim3 block(H, num_heads, 1);
    const float2* input2 = reinterpret_cast<const float2*>(input);
    float2* output2 = reinterpret_cast<float2*>(output);
    TransposeCtx<float2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    const dim3 block(H, num_heads, 1);
    const half2* input2 = reinterpret_cast<const half2*>(input);
    half2* output2 = reinterpret_cast<half2*>(output);
    TransposeCtx<half2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else {  // this should be an "odd" case. probably not worth catching it in the half2 kernel.
    const dim3 block(head_size, num_heads, 1);
    TransposeCtx<half><<<grid, block, 0, stream>>>(head_size, input, output);
  }

  return CUDA_CALL(cudaPeekAtLastError());
}

template <typename T>
__global__ void TransposeQKV(const int H, const T* input, T* output) {
  // Input:  BxSx3xNxH
  // Output: 3xBxNxSxH

  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int NH = num_heads * H;
  const int NHS = NH * sequence_length;
  const int in_offset = n * H + m * NH + s * 3 * NH + b * NHS * 3;
  const int out_offset = s * H + n * sequence_length * H + b * NHS + m * NHS * batch_size;

  const int i = threadIdx.x;
  if (i < H) {
    output[out_offset + i] = input[in_offset + i];
  }
}

bool LaunchTransQkv(cudaStream_t stream,
                    const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                    const float* input, float* output) {
  const dim3 grid(sequence_length, batch_size, 3);
  if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    const float2* input2 = reinterpret_cast<const float2*>(input);
    float2* output2 = reinterpret_cast<float2*>(output);
    const dim3 block(H, num_heads, 1);
    TransposeQKV<float2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else {
    const dim3 block(head_size, num_heads, 1);
    TransposeQKV<float><<<grid, block, 0, stream>>>(head_size, input, output);
  }
  return CUDA_CALL(cudaPeekAtLastError());
}

bool LaunchTransQkv(cudaStream_t stream,
                    const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                    const half* input, half* output) {
  const dim3 grid(sequence_length, batch_size, 3);
  if (0 == (head_size % 4)) {
    const int H = head_size / 4;
    const dim3 block(H, num_heads, 1);
    const float2* input2 = reinterpret_cast<const float2*>(input);
    float2* output2 = reinterpret_cast<float2*>(output);
    TransposeQKV<float2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    const dim3 block(H, num_heads, 1);
    const half2* input2 = reinterpret_cast<const half2*>(input);
    half2* output2 = reinterpret_cast<half2*>(output);
    TransposeQKV<half2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else {  // this should be an "odd" case. probably not worth catching it in the half2 kernel..
    const dim3 block(head_size, num_heads, 1);
    TransposeQKV<half><<<grid, block, 0, stream>>>(head_size, input, output);
  }
  return CUDA_CALL(cudaPeekAtLastError());
}

cublasStatus_t inline CublasGemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float alpha,
    const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB,
    const float beta, float* C, int ldc, long long int strideC, int batchCount) {
  return cublasSgemmStridedBatched(
      handle, transa, transb, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batchCount);
}

cublasStatus_t inline CublasGemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const half alpha,
    const half* A, int lda, long long int strideA, const half* B, int ldb, long long int strideB,
    const half beta, half* C, int ldc, long long int strideC, int batchCount) {
  return cublasHgemmStridedBatched(
      handle, transa, transb, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batchCount);
}

template <typename T>
bool QkvToContext(
    cublasHandle_t& cublas, cudaStream_t stream,
    const int batch_size, const int sequence_length, const int num_heads, const int head_size, const size_t element_size,
    const T* input, T* output, T* workspace,
    const int* mask_index) {
  const size_t bytes = ScratchSize(element_size, batch_size, num_heads, sequence_length);
  T* scratch1 = workspace;
  T* scratch2 = scratch1 + (bytes / element_size);
  T* scratch3 = scratch2 + (bytes / element_size);

  // input should be BxSx3xNxH => scratch3: 3xBxNxSxH
  if (!LaunchTransQkv(stream, sequence_length, batch_size, head_size, num_heads, input, scratch3)) {
    return false;
  }

  // now scratch3 has Q, K, V: each has size BxNxSxH
  const int batches = batch_size * num_heads;
  const int size_per_batch = sequence_length * head_size;
  const int total_size = batches * size_per_batch;
  const int temp_matrix_size = sequence_length * sequence_length;

  const T* q = scratch3;
  const T* k = q + total_size;
  const T* v = k + total_size;

  cublasSetStream(cublas, stream);
  CublasMathModeSetter helper(cublas, CUBLAS_TENSOR_OP_MATH);

  // compute Q*K' (as K'*Q), scaled by 1/sqrt(H) and store in scratch1: BxNxSxS
  const float rsqrt_head_size = 1.f / sqrt(static_cast<float>(head_size));
  if (!CUBLAS_CALL(CublasGemmStridedBatched(
          cublas, CUBLAS_OP_T, CUBLAS_OP_N, sequence_length, sequence_length, head_size, rsqrt_head_size, k, head_size, size_per_batch,
          q, head_size, size_per_batch, 0.f, scratch1, sequence_length, temp_matrix_size, batches))) {
    return false;
  }

  // apply softmax and store result P to scratch2: BxNxSxS
  if (!ComputeMaskedSoftmax<T>(stream, sequence_length, batch_size, num_heads, mask_index, scratch1, scratch2)) {
    return false;
  }

  // compute P*V (as V*P), and store in scratch3: BxNxSxH
  if (!CUBLAS_CALL(CublasGemmStridedBatched(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, head_size, sequence_length, sequence_length, 1.f, v, head_size, size_per_batch,
          scratch2, sequence_length, temp_matrix_size, 0.f, scratch3, head_size, size_per_batch, batches))) {
    return false;
  }

  // scratch3 is BxNxSxH, transpose to output BxSxNxH
  return LaunchTransCtx(stream, sequence_length, batch_size, head_size, num_heads, scratch3, output);
}

bool LaunchAttentionKernel(
    const void* input,
    const int* mask_index,
    void* output,
    const int batch_size,
    const int sequence_length,
    const int num_heads,
    const int head_size,
    void* workspace,
    cublasHandle_t& cublas,
    const size_t element_size) {
  // use default stream
  const cudaStream_t stream = nullptr;

  if (element_size == 2) {
    return QkvToContext(cublas, stream,
                        batch_size, sequence_length, num_heads, head_size, element_size,
                        reinterpret_cast<const half*>(input), reinterpret_cast<half*>(output), reinterpret_cast<half*>(workspace),
                        mask_index);
  } else {
    return QkvToContext(cublas, stream,
                        batch_size, sequence_length, num_heads, head_size, element_size,
                        reinterpret_cast<const float*>(input), reinterpret_cast<float*>(output), reinterpret_cast<float*>(workspace),
                        mask_index);
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
