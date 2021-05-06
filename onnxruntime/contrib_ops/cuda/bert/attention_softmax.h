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

#pragma once

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

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
__device__ inline void SoftmaxWithRawMaskSmall(const int all_sequence_length,
                                               const int sequence_length,
                                               const int* attention_mask,  // 2D or 3D attention mask
                                               const T* input,
                                               T* output,
                                               const bool is_unidirectional,
                                               const float scalar,
                                               const int mask_dimension) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  // Input dimension is BxNxSxS*; blockIdx.y is batch index b; gridDim.x=N*S;  blockIdx.x is index within N*S;
  int index = (blockIdx.y * gridDim.x + blockIdx.x) * all_sequence_length + threadIdx.x;

  float thread_data = -CUDART_INF_F;
  if (threadIdx.x < all_sequence_length) {
    const int batch_index = blockIdx.y;
    const int sequence_index = blockIdx.x % sequence_length;
    int mask_offset = 0;
    if (mask_dimension == 2) {
      mask_offset = batch_index * all_sequence_length + threadIdx.x;
    } else if (mask_dimension == 3) {
      mask_offset = batch_index * sequence_length * all_sequence_length + sequence_index * all_sequence_length + threadIdx.x;
    } else if (mask_dimension == 4){
      mask_offset = batch_index * 1024 * 1024 + (all_sequence_length - sequence_length) * 1024 + sequence_index * 1024 + threadIdx.x;
    }
    //const int mask_offset = (mask_dimension == 2) ? batch_index * all_sequence_length + threadIdx.x : batch_index * sequence_length * all_sequence_length + sequence_index * all_sequence_length + threadIdx.x;

    const int& mask = attention_mask[mask_offset];
    float mask_value = mask > 0 ? 0.0f : -10000.0f;

    if (is_unidirectional) {
      int from_index = all_sequence_length - sequence_length + sequence_index;  // offset of from token in all sequence length.
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
__global__ void SoftmaxWithRawMaskSmallKernel(const int all_sequence_length, const int sequence_length, const int* attention_mask, const T* input, T* output, const bool is_unidirectional, const float scalar, const int mask_dimension) {
  SoftmaxWithRawMaskSmall<T, TPB>(all_sequence_length, sequence_length, attention_mask, input, output, is_unidirectional, scalar, mask_dimension);
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
bool ComputeSoftmaxWithRawMask(cudaStream_t stream, const int all_sequence_length, const int sequence_length, const int batch_size, const int num_heads,
                               const int* attention_mask, const T* input, T* output, const bool is_unidirectional, const float scalar,
                               const int mask_dimension) {
  const dim3 grid(sequence_length * num_heads, batch_size, 1);

  if (all_sequence_length <= 32) {
    const int blockSize = 32;
    SoftmaxWithRawMaskSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, attention_mask, input, output, is_unidirectional, scalar, mask_dimension);
  } else if (all_sequence_length <= 64) {
    const int blockSize = 64;
    SoftmaxWithRawMaskSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, attention_mask, input, output, is_unidirectional, scalar, mask_dimension);
  } else if (all_sequence_length <= 128) {
    const int blockSize = 128;
    SoftmaxWithRawMaskSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, attention_mask, input, output, is_unidirectional, scalar, mask_dimension);
  } else if (all_sequence_length <= 256) {
    const int blockSize = 256;
    SoftmaxWithRawMaskSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, attention_mask, input, output, is_unidirectional, scalar, mask_dimension);
  } else if (all_sequence_length <= 512) {
    const int blockSize = 512;
    SoftmaxWithRawMaskSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, attention_mask, input, output, is_unidirectional, scalar, mask_dimension);
  } else if (all_sequence_length <= 1024) {
    const int blockSize = 1024;
    SoftmaxWithRawMaskSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, attention_mask, input, output, is_unidirectional, scalar, mask_dimension);
  } else {
    ORT_THROW("Attention CUDA operator does not supported 2D attention mask with total sequence length > 1024.");
  }

  return CUDA_CALL(cudaPeekAtLastError());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
