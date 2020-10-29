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

size_t ScratchSize(size_t element_size, int batch_size, int num_heads, int sequence_length, int all_sequence_length) {
  const size_t len = batch_size * num_heads * sequence_length * all_sequence_length;
  const size_t bytes = len * element_size;

  const size_t alignment = 256;
  const size_t bytesAligned = AlignTo(bytes, alignment);
  return bytesAligned;
}

size_t GetAttentionWorkspaceSize(
    size_t element_size,
    int batch_size,
    int num_heads,
    int head_size,
    int sequence_length,
    int past_sequence_length) {
  size_t qkv_size = 3 * batch_size * sequence_length * num_heads * head_size * element_size;
  return qkv_size + 2 * ScratchSize(element_size, batch_size, num_heads, sequence_length, past_sequence_length + sequence_length);
}

template <typename T, unsigned TPB>
__device__ inline void Softmax(const int all_sequence_length,
                               const int sequence_length,
                               const int valid_end,
                               const int valid_start,
                               const T* input,
                               T* output) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  float thread_data_max(-CUDART_INF_F);

  // e^x is represented as infinity if x is large enough, like 100.f.
  // Infinity divided by Infinity is a NAN. Thus, softmax gets a NAN if one or more item are large enough.
  // a math transform as below is leveraged to get a stable softmax:
  // e^xi/(e^x1 + ...e^xn) = e^(xi - max) / (e^(x1 - max) + ... + e^(xn - max))
  const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * all_sequence_length;
  for (int i = threadIdx.x; i < valid_end; i += TPB) {
    if (i >= valid_start) {
      const int index = offset + i;
      if (thread_data_max < float(input[index])) {
        thread_data_max = float(input[index]);
      }
    }
  }

  const auto max = BlockReduce(tmp_storage).Reduce(thread_data_max, cub::Max());

  // Store max value
  if (threadIdx.x == 0) {
    max_block = max;
  }
  __syncthreads();

  float thread_data_sum(0.f);
  for (int i = threadIdx.x; i < valid_end; i += TPB) {
    if (i >= valid_start) {
      const int index = offset + i;
      const float val = input[index];
      thread_data_sum += expf(val - max_block);
    }
  }

  const auto sum = BlockReduce(tmp_storage).Reduce(thread_data_sum, cub::Sum());
  if (threadIdx.x == 0) {
    sum_reverse_block = 1.f / sum;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < all_sequence_length; i += TPB) {
    const int index = offset + i;
    const float val = (i >= valid_start && i < valid_end) ? expf(float(input[index]) - max_block) * sum_reverse_block : 0.f;
    output[index] = T(val);
  }
}

template <typename T, unsigned TPB>
__device__ inline void SoftmaxSmall(const int all_sequence_length,
                                    const int sequence_length,
                                    const int valid_end,
                                    const int valid_start,
                                    const T* input,
                                    T* output,
                                    bool is_unidirectional) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  // Input dimension is BxNxSxS*; blockIdx.y is batch index b; gridDim.x=N*S;  blockIdx.x is index within N*S;
  const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * all_sequence_length;
  const int index = offset + threadIdx.x;

  bool is_valid = false;  // whether it has attention mask == 1.

  // Update end position for unidirectional.
  int end = valid_end;
  if (is_unidirectional) {
    int end_unid = all_sequence_length - sequence_length + (blockIdx.x % sequence_length) + 1;
    if (end_unid <= valid_start) {
      // In this situation, mask of [0, end_unid) and [valid_start, valid_end) has -10000, and [end_unid, valid_start) and [valid_end, all_seq_len) has -20000.
      // So [0, end_unid) will also have value after softmax.
      is_valid = threadIdx.x < end_unid;
    } else {
      end = min(valid_end, end_unid);
    }
  }

  is_valid = is_valid || (threadIdx.x >= valid_start && threadIdx.x < end);

  // e^x is represented as infinity if x is large enough, like 100.f.
  // Infinity divided by Infinity is a NAN. Thus, softmax gets a NAN if one or more item are large enough.
  // a math transform as below is leveraged to get a stable softmax:
  // e^xi/(e^x1 + ...e^xn) = e^(xi - max) / (e^(x1 - max) + ... + e^(xn - max))
  float thread_data_max = is_valid ? float(input[index]) : float(-CUDART_INF_F);
  const auto max = BlockReduce(tmp_storage).Reduce(thread_data_max, cub::Max(), end);

  // Store max value
  if (threadIdx.x == 0) {
    max_block = max;
  }
  __syncthreads();

  float thread_data_exp(0.f);
  if (is_valid) {
    thread_data_exp = expf(float(input[index]) - max_block);
  }

  const auto sum = BlockReduce(tmp_storage).Reduce(thread_data_exp, cub::Sum(), end);

  // Store value of 1.0/sum.
  if (threadIdx.x == 0) {
    sum_reverse_block = (1.f) / sum;
  }
  __syncthreads();

  // threadIdx.x might be larger than all_sequence_length due to alignment to 32x.
  if (threadIdx.x < all_sequence_length) {
    output[index] = T(thread_data_exp * sum_reverse_block);
  }
}

template <typename T, unsigned TPB>
__device__ inline void SoftmaxWithMask2DSmall(const int all_sequence_length,
                                              const int sequence_length,
                                              const int* attention_mask,  // 2D attention mask
                                              const T* input,
                                              T* output,
                                              const bool is_unidirectional,
                                              const float scalar) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  // Input dimension is BxNxSxS*; blockIdx.y is batch index b; gridDim.x=N*S;  blockIdx.x is index within N*S;
  int index = (blockIdx.y * gridDim.x + blockIdx.x) * all_sequence_length + threadIdx.x;

  float thread_data = -CUDART_INF_F;
  if (threadIdx.x < all_sequence_length) {
    const int& mask = attention_mask[blockIdx.y * all_sequence_length + threadIdx.x];
    float mask_value = mask > 0 ? 0.0f : -10000.0f;

    if (is_unidirectional) {
      int from_index = all_sequence_length - sequence_length + (blockIdx.x % sequence_length);  // offset of from token in all sequence length.
      if (threadIdx.x > from_index) {
        mask_value += -10000.0f;
      }
    }

    thread_data = float(input[index]) * scalar + mask_value;
  }

  const float max = BlockReduce(tmp_storage).Reduce(thread_data, cub::Max(), all_sequence_length);

  // Store max value
  if (threadIdx.x == 0) {
    max_block = max;
  }
  __syncthreads();

  float thread_data_exp = threadIdx.x < all_sequence_length ? expf(thread_data - max_block) : 0.0f;
  const auto sum = BlockReduce(tmp_storage).Reduce(thread_data_exp, cub::Sum(), all_sequence_length);

  // Store value of 1.0/sum
  if (threadIdx.x == 0) {
    sum_reverse_block = (1.f) / sum;
  }
  __syncthreads();

  if (threadIdx.x < all_sequence_length) {
    output[index] = T(thread_data_exp * sum_reverse_block);
  }
}

template <typename T, unsigned TPB>
__global__ void SoftmaxKernelSmall(const int all_sequence_length, const int sequence_length, const T* input, T* output, bool is_unidirectional) {
  SoftmaxSmall<T, TPB>(all_sequence_length, sequence_length, all_sequence_length, 0, input, output, is_unidirectional);
}

template <typename T, unsigned TPB>
__global__ void SoftmaxKernel(const int all_sequence_length, const int sequence_length, const T* input, T* output) {
  Softmax<T, TPB>(all_sequence_length, sequence_length, all_sequence_length, 0, input, output);
}

template <typename T>
bool ComputeSoftmax(
    cudaStream_t stream, const int all_sequence_length, const int sequence_length, const int batch_size, const int num_heads,
    const T* input, T* output, bool is_unidirectional) {
  const dim3 grid(sequence_length * num_heads, batch_size, 1);
  if (all_sequence_length <= 32) {
    const int blockSize = 32;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, input, output, is_unidirectional);
  } else if (all_sequence_length <= 64) {
    const int blockSize = 64;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, input, output, is_unidirectional);
  } else if (all_sequence_length <= 128) {
    const int blockSize = 128;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, input, output, is_unidirectional);
  } else if (all_sequence_length <= 256) {
    const int blockSize = 256;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, input, output, is_unidirectional);
  } else if (all_sequence_length <= 512) {
    const int blockSize = 512;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, input, output, is_unidirectional);
  } else if (all_sequence_length <= 1024) {
    const int blockSize = 1024;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, input, output, is_unidirectional);
  } else if (!is_unidirectional) {
    const int blockSize = 1024;
    SoftmaxKernel<T, blockSize><<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, input, output);
  } else {
    ORT_THROW("Attention CUDA operator does not support unidirectional with total sequence length > 1024.");
  }

  return CUDA_CALL(cudaPeekAtLastError());
}

template <typename T, unsigned TPB>
__global__ void MaskedSoftmaxKernelSmall(const int all_sequence_length, const int sequence_length, const int* mask_end, const int* mask_start, const T* input, T* output, bool is_unidirectional) {
  __shared__ int start_position;
  __shared__ int end_position;

  if (threadIdx.x == 0) {
    const int batch = blockIdx.y;
    start_position = mask_start != nullptr ? max(0, mask_start[batch]) : 0;
    end_position = min(all_sequence_length, mask_end[batch]);

    // Attend to no word has same effect as attend to all words. This is added to get parity with CPU result.
    if (start_position >= end_position) {
      start_position = 0;
      end_position = all_sequence_length;
    }
  }
  __syncthreads();

  SoftmaxSmall<T, TPB>(all_sequence_length, sequence_length, end_position, start_position, input, output, is_unidirectional);
}

template <typename T, unsigned TPB>
__global__ void MaskedSoftmaxKernel(const int all_sequence_length, const int sequence_length, const int* mask_end, const int* mask_start, const T* input, T* output) {
  __shared__ int start_position;
  __shared__ int end_position;

  if (threadIdx.x == 0) {
    const int batch = blockIdx.y;
    start_position = mask_start != nullptr ? max(0, mask_start[batch]) : 0;
    end_position = min(all_sequence_length, mask_end[batch]);

    // Attend to no word has same effect as attend to all words. This is added to get parity with CPU result.
    if (start_position >= end_position) {
      start_position = 0;
      end_position = all_sequence_length;
    }
  }
  __syncthreads();

  Softmax<T, TPB>(all_sequence_length, sequence_length, end_position, start_position, input, output);
}

template <typename T, unsigned TPB>
__global__ void SoftmaxWithMask2DSmallKernel(const int all_sequence_length, const int sequence_length, const int* attention_mask, const T* input, T* output, const bool is_unidirectional, const float scalar) {
  SoftmaxWithMask2DSmall<T, TPB>(all_sequence_length, sequence_length, attention_mask, input, output, is_unidirectional, scalar);
}

template <typename T>
bool ComputeSoftmaxWithMask1D(cudaStream_t stream, const int all_sequence_length, const int sequence_length, const int batch_size, const int num_heads,
                              const int* mask_index, const int* mask_start, const T* input, T* output, const bool is_unidirectional) {
  const dim3 grid(sequence_length * num_heads, batch_size, 1);

  if (all_sequence_length <= 32) {
    const int blockSize = 32;
    MaskedSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start, input, output, is_unidirectional);
  } else if (all_sequence_length <= 64) {
    const int blockSize = 64;
    MaskedSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start, input, output, is_unidirectional);
  } else if (all_sequence_length <= 128) {
    const int blockSize = 128;
    MaskedSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start, input, output, is_unidirectional);
  } else if (all_sequence_length <= 256) {
    const int blockSize = 256;
    MaskedSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start, input, output, is_unidirectional);
  } else if (all_sequence_length <= 512) {
    const int blockSize = 512;
    MaskedSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start, input, output, is_unidirectional);
  } else if (all_sequence_length <= 1024) {
    const int blockSize = 1024;
    MaskedSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start, input, output, is_unidirectional);
  } else if (!is_unidirectional) {
    const int blockSize = 1024;
    MaskedSoftmaxKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start, input, output);
  } else {
    ORT_THROW("Attention CUDA operator does not support unidirectional with total sequence length > 1024.");
  }

  return CUDA_CALL(cudaPeekAtLastError());
}

template <typename T>
bool ComputeSoftmaxWithMask2D(cudaStream_t stream, const int all_sequence_length, const int sequence_length, const int batch_size, const int num_heads,
                              const int* attention_mask, const T* input, T* output, const bool is_unidirectional, const float scalar) {
  const dim3 grid(sequence_length * num_heads, batch_size, 1);

  if (all_sequence_length <= 32) {
    const int blockSize = 32;
    SoftmaxWithMask2DSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, attention_mask, input, output, is_unidirectional, scalar);
  } else if (all_sequence_length <= 64) {
    const int blockSize = 64;
    SoftmaxWithMask2DSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, attention_mask, input, output, is_unidirectional, scalar);
  } else if (all_sequence_length <= 128) {
    const int blockSize = 128;
    SoftmaxWithMask2DSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, attention_mask, input, output, is_unidirectional, scalar);
  } else if (all_sequence_length <= 256) {
    const int blockSize = 256;
    SoftmaxWithMask2DSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, attention_mask, input, output, is_unidirectional, scalar);
  } else if (all_sequence_length <= 512) {
    const int blockSize = 512;
    SoftmaxWithMask2DSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, attention_mask, input, output, is_unidirectional, scalar);
  } else if (all_sequence_length <= 1024) {
    const int blockSize = 1024;
    SoftmaxWithMask2DSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, attention_mask, input, output, is_unidirectional, scalar);
  } else {
    ORT_THROW("Attention CUDA operator does not supported 2D attention mask with total sequence length > 1024.");
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

template <typename T>
__global__ void ConcatPastToPresent(const int sequence_length,
                                    const T* past,
                                    const T* k_v,
                                    T* present) {
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;
  const int is_v = blockIdx.z;  // 0 for k, 1 for v

  const int all_sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int num_heads = blockDim.y;
  const int H = blockDim.x;

  // past:    2 x BxNxS'xH   (past_k and past_v)
  // k_v:     2 x BxNxSxH    (k and v)
  // present: 2 x BxNxS*xH   (present_k and present_v)
  const int past_sequence_length = all_sequence_length - sequence_length;

  const int present_SH = all_sequence_length * H;
  const int present_NSH = num_heads * present_SH;
  int out_offset = b * present_NSH + n * present_SH + s * H + h + is_v * (present_NSH * batch_size);
  if (s < past_sequence_length) {
    const int past_SH = past_sequence_length * H;
    const int past_NSH = num_heads * past_SH;
    const int in_offset = b * past_NSH + n * past_SH + s * H + h + is_v * (past_NSH * batch_size);
    present[out_offset] = past[in_offset];
  } else if (s < all_sequence_length) {
    const int SH = sequence_length * H;
    const int NSH = num_heads * SH;
    const int in_offset = b * NSH + n * SH + (s - past_sequence_length) * H + h + is_v * (NSH * batch_size);
    present[out_offset] = k_v[in_offset];
  }
}

bool LaunchConcatPastToPresent(cudaStream_t stream,
                               const int all_sequence_length,
                               const int sequence_length,
                               const int batch_size,
                               const int head_size,
                               const int num_heads,
                               const float* past,
                               const float* k_v,
                               float* present) {
  const dim3 grid(all_sequence_length, batch_size, 2);
  if (0 == (head_size & 1)) {
    const dim3 block(head_size / 2, num_heads, 1);
    ConcatPastToPresent<float2><<<grid, block, 0, stream>>>(sequence_length, reinterpret_cast<const float2*>(past), reinterpret_cast<const float2*>(k_v), reinterpret_cast<float2*>(present));
  } else {
    const dim3 block(head_size, num_heads, 1);
    ConcatPastToPresent<float><<<grid, block, 0, stream>>>(sequence_length, past, k_v, present);
  }
  return CUDA_CALL(cudaPeekAtLastError());
}

bool LaunchConcatPastToPresent(cudaStream_t stream,
                               const int all_sequence_length,
                               const int sequence_length,
                               const int batch_size,
                               const int head_size,
                               const int num_heads,
                               const half* past,
                               const half* k_v,
                               half* present) {
  const dim3 grid(all_sequence_length, batch_size, 2);
  if (0 == (head_size % 4)) {
    const dim3 block(head_size / 4, num_heads, 1);
    ConcatPastToPresent<float2><<<grid, block, 0, stream>>>(sequence_length, reinterpret_cast<const float2*>(past), reinterpret_cast<const float2*>(k_v), reinterpret_cast<float2*>(present));
  } else if (0 == (head_size & 1)) {
    const dim3 block(head_size / 2, num_heads, 1);
    ConcatPastToPresent<half2><<<grid, block, 0, stream>>>(sequence_length, reinterpret_cast<const half2*>(past), reinterpret_cast<const half2*>(k_v), reinterpret_cast<half2*>(present));
  } else {  // this should be an "odd" case. probably not worth catching it in the half2 kernel.
    const dim3 block(head_size, num_heads, 1);
    ConcatPastToPresent<half><<<grid, block, 0, stream>>>(sequence_length, past, k_v, present);
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
    const cudaDeviceProp& prop, cublasHandle_t& cublas, cudaStream_t stream,
    const int batch_size, const int sequence_length, const int num_heads, const int head_size, const size_t element_size,
    const T* input, T* output, T* workspace,
    const int* mask_index, const std::vector<int64_t>* mask_index_dims,
    bool is_unidirectional, int past_sequence_length, const T* past, T* present) {
  const int all_sequence_length = past_sequence_length + sequence_length;
  const size_t bytes = ScratchSize(element_size, batch_size, num_heads, sequence_length, all_sequence_length);
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

  const T* q = scratch3;
  const T* k = q + total_size;
  const T* v = k + total_size;

  cublasSetStream(cublas, stream);
  CublasMathModeSetter helper(prop, cublas, CUBLAS_TENSOR_OP_MATH);

  // Concat past (2xBxNxS'xH) to present (2xBxNxS*xH):
  // past_k (BxNxS'xH) + k (BxNxSxH) => present_k (BxNxS*xH)
  // past_v (BxNxS'xH) + v (BxNxSxH) => present_v (BxNxS*xH)
  const int present_size_per_batch = all_sequence_length * head_size;
  if (nullptr != present) {
    if (!LaunchConcatPastToPresent(stream, all_sequence_length, sequence_length, batch_size, head_size, num_heads, past, k, present)) {
      return false;
    }

    // update pointers to present_k and present_v.
    k = present;
    v = present + batches * present_size_per_batch;
  }

  bool use_2d_attention_mask = (nullptr != mask_index && nullptr != mask_index_dims && mask_index_dims->size() == 2);

  // compute Q*K' (as K'*Q), scaled by 1/sqrt(H) and store in scratch1: BxNxSxS*
  // Q: BxNxSxH, K (present_k): BxNxS*xH, Q*K': BxNxSxS*
  const float rsqrt_head_size = 1.f / sqrt(static_cast<float>(head_size));
  const int temp_matrix_size = sequence_length * all_sequence_length;
  T alpha = (T)(use_2d_attention_mask ? 1.0f : rsqrt_head_size);
  if (!CUBLAS_CALL(CublasGemmStridedBatched(
          cublas, CUBLAS_OP_T, CUBLAS_OP_N, all_sequence_length, sequence_length, head_size, alpha, k, head_size, present_size_per_batch,
          q, head_size, size_per_batch, 0.f, scratch1, all_sequence_length, temp_matrix_size, batches))) {
    return false;
  }

  // apply softmax and store result P to scratch2: BxNxSxS*
  if (use_2d_attention_mask) {  // 2d attention mask
    if (!ComputeSoftmaxWithMask2D<T>(stream, all_sequence_length, sequence_length, batch_size, num_heads, mask_index, scratch1, scratch2, is_unidirectional, rsqrt_head_size)) {
      return false;
    }
  } else if (nullptr != mask_index) {  // 1d mask index
    ORT_ENFORCE(nullptr != mask_index_dims && mask_index_dims->size() == 1);
    // mask_index has 1D shape: either (batch_size) or (2*batch_size). Only the later one has start postions.
    const int* mask_start = (mask_index_dims->at(0) > batch_size) ? mask_index + batch_size : nullptr;
    if (!ComputeSoftmaxWithMask1D<T>(stream, all_sequence_length, sequence_length, batch_size, num_heads, mask_index, mask_start, scratch1, scratch2, is_unidirectional)) {
      return false;
    }
  } else {  // no mask
    if (!ComputeSoftmax<T>(stream, all_sequence_length, sequence_length, batch_size, num_heads, scratch1, scratch2, is_unidirectional)) {
      return false;
    }
  }

  // compute P*V (as V*P), and store in scratch3: BxNxSxH
  if (!CUBLAS_CALL(CublasGemmStridedBatched(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, head_size, sequence_length, all_sequence_length, 1.f, v, head_size, present_size_per_batch,
          scratch2, all_sequence_length, temp_matrix_size, 0.f, scratch3, head_size, size_per_batch, batches))) {
    return false;
  }

  // scratch3 is BxNxSxH, transpose to output BxSxNxH
  return LaunchTransCtx(stream, sequence_length, batch_size, head_size, num_heads, scratch3, output);
}

bool LaunchAttentionKernel(
    const cudaDeviceProp& prop,
    const void* input,
    const int* mask_index,
    const std::vector<int64_t>* mask_index_dims,
    void* output,
    const int batch_size,
    const int sequence_length,
    const int num_heads,
    const int head_size,
    void* workspace,
    cublasHandle_t& cublas,
    const size_t element_size,
    bool is_unidirectional,
    int past_sequence_length,
    const void* past,
    void* present) {
  // use default stream
  const cudaStream_t stream = nullptr;

  if (element_size == 2) {
    return QkvToContext(prop, cublas, stream,
                        batch_size, sequence_length, num_heads, head_size, element_size,
                        reinterpret_cast<const half*>(input), reinterpret_cast<half*>(output), reinterpret_cast<half*>(workspace),
                        mask_index, mask_index_dims, is_unidirectional,
                        past_sequence_length, reinterpret_cast<const half*>(past), reinterpret_cast<half*>(present));
  } else {
    return QkvToContext(prop, cublas, stream,
                        batch_size, sequence_length, num_heads, head_size, element_size,
                        reinterpret_cast<const float*>(input), reinterpret_cast<float*>(output), reinterpret_cast<float*>(workspace),
                        mask_index, mask_index_dims, is_unidirectional,
                        past_sequence_length, reinterpret_cast<const float*>(past), reinterpret_cast<float*>(present));
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
