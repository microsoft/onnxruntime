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

#include <cub/cub.cuh>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/math/softmax.h"
#include "contrib_ops/cuda/bert/attention_softmax.h"
using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace attention_softmax_cuda {

template <typename T, unsigned TPB>
__device__ inline void Softmax(const int all_sequence_length,
                               const int valid_end,
                               const int valid_start,
                               const T* rel_pos_bias,
                               const bool broadcast_rel_pos_bias,
                               const T* input,
                               T* output) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  float thread_data_max(-CUDART_INF_F);

  const bool no_rpb = (rel_pos_bias == nullptr);

  // e^x is represented as infinity if x is large enough, like 100.f.
  // Infinity divided by Infinity is a NAN. Thus, softmax gets a NAN if one or more item are large enough.
  // a math transform as below is leveraged to get a stable softmax:
  // e^xi/(e^x1 + ...e^xn) = e^(xi - max) / (e^(x1 - max) + ... + e^(xn - max))
  const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * all_sequence_length;
  const int size_per_batch = gridDim.x * all_sequence_length;
  for (int i = threadIdx.x; i < valid_end; i += TPB) {
    if (i >= valid_start) {
      const int index = offset + i;
      float input_at_idx = no_rpb
                               ? float(input[index])
                               : float(input[index] + (broadcast_rel_pos_bias
                                                           ? rel_pos_bias[index % size_per_batch]
                                                           : rel_pos_bias[index]));
      if (thread_data_max < input_at_idx) {
        thread_data_max = input_at_idx;
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
      float val = no_rpb ? input[index] : input[index] + rel_pos_bias[index % size_per_batch];
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
    float input_at_idx = no_rpb ? float(input[index]) : float(input[index] + rel_pos_bias[index % size_per_batch]);
    const float val = (i >= valid_start && i < valid_end) ? expf(input_at_idx - max_block) * sum_reverse_block : 0.f;
    output[index] = T(val);
  }
}

template <typename T, unsigned TPB>
__device__ inline void SoftmaxSmall(const int all_sequence_length,
                                    const int sequence_length,
                                    const int valid_end,
                                    const int valid_start,
                                    const T* rel_pos_bias,
                                    const bool broadcast_rel_pos_bias,
                                    const T* input,
                                    T* output,
                                    bool causal) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  // Input dimension is BxNxSxS*; blockIdx.y is batch index b; gridDim.x=N*S;  blockIdx.x is index within N*S;
  const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * all_sequence_length;
  const int index = offset + threadIdx.x;

  // Update end position for causal.
  int end = valid_end;
  if (causal) {
    const int end_causal = all_sequence_length - sequence_length + (blockIdx.x % sequence_length) + 1;
    if (end_causal < end) {
      end = end_causal;
    }
  }

  const bool is_valid = (threadIdx.x >= valid_start && threadIdx.x < end);

  // e^x is represented as infinity if x is large enough, like 100.f.
  // Infinity divided by Infinity is a NAN. Thus, softmax gets a NAN if one or more item are large enough.
  // a math transform as below is leveraged to get a stable softmax:
  // e^xi/(e^x1 + ...e^xn) = e^(xi - max) / (e^(x1 - max) + ... + e^(xn - max))
  const bool no_rpb = (rel_pos_bias == nullptr);
  const int size_per_batch = gridDim.x * all_sequence_length;
  float input_data = no_rpb
                         ? float(input[index])
                         : float(input[index] + (broadcast_rel_pos_bias
                                                     ? rel_pos_bias[index % size_per_batch]
                                                     : rel_pos_bias[index]));
  float thread_data_max = is_valid ? input_data : float(-CUDART_INF_F);
  const auto max = BlockReduce(tmp_storage).Reduce(thread_data_max, cub::Max(), end);

  // Store max value
  if (threadIdx.x == 0) {
    max_block = max;
  }
  __syncthreads();

  float thread_data_exp(0.f);
  if (is_valid) {
    thread_data_exp = expf(input_data - max_block);
  }

  const auto sum = BlockReduce(tmp_storage).Reduce(thread_data_exp, cub::Sum(), end);

  // Store value of 1.0/sum.
  if (threadIdx.x == 0) {
    sum_reverse_block = (1.f) / sum;
  }
  __syncthreads();

  // threadIdx.x might be larger than all_sequence_length due to alignment to 32x.
  if (threadIdx.x < all_sequence_length) {
    output[index] = is_valid ? T(thread_data_exp * sum_reverse_block) : T(0.f);
  }
}

template <typename T, unsigned TPB>
__global__ void SoftmaxLargeKernel(const int all_sequence_length,
                                   const int sequence_length,
                                   const int valid_end,
                                   const int valid_start,
                                   const T* rel_pos_bias,
                                   const bool broadcast_rel_pos_bias,
                                   const T* input,
                                   T* output,
                                   bool causal) {
  extern __shared__ float cached_data[];  // float[all_sequence_length]

  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  // Update end position for causal.
  int end = valid_end;
  if (causal) {
    int end_causal = all_sequence_length - sequence_length + (blockIdx.x % sequence_length) + 1;
    if (end_causal < end) {
      end = end_causal;
    }
  }

  // Input dimension is BxNxSxS*; blockIdx.y is batch index b; gridDim.x=N*S;  blockIdx.x is index within N*S;
  const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * all_sequence_length;
  const int size_per_batch = gridDim.x * all_sequence_length;

  float thread_data_max = -CUDART_INF_F;
  for (int seq_idx = threadIdx.x; seq_idx < all_sequence_length; seq_idx += TPB) {
    const int index = offset + seq_idx;
    const bool is_valid = (seq_idx >= valid_start && seq_idx < end);

    // e^x is represented as infinity if x is large enough, like 100.f.
    // Infinity divided by Infinity is a NAN. Thus, softmax gets a NAN if one or more item are large enough.
    // a math transform as below is leveraged to get a stable softmax:
    // e^xi/(e^x1 + ...e^xn) = e^(xi - max) / (e^(x1 - max) + ... + e^(xn - max))
    float input_data = is_valid
                           ? (rel_pos_bias
                                  ? float(input[index] + (broadcast_rel_pos_bias
                                                              ? rel_pos_bias[index % size_per_batch]
                                                              : rel_pos_bias[index]))
                                  : float(input[index]))
                           : float(-CUDART_INF_F);
    cached_data[seq_idx] = input_data;
    thread_data_max = max(thread_data_max, input_data);
  }
  const auto max = BlockReduce(tmp_storage).Reduce(thread_data_max, cub::Max(), end);

  // Store max value
  if (threadIdx.x == 0) {
    max_block = max;
  }
  __syncthreads();

  float thread_data_exp(0.f);
  for (int seq_idx = threadIdx.x; seq_idx < all_sequence_length; seq_idx += TPB) {
    const bool is_valid = (seq_idx >= valid_start && seq_idx < end);
    cached_data[seq_idx] = is_valid ? expf(cached_data[seq_idx] - max_block) : 0.0f;
    thread_data_exp += cached_data[seq_idx];
  }
  const auto sum = BlockReduce(tmp_storage).Reduce(thread_data_exp, cub::Sum(), end);

  // Store value of 1.0/sum.
  if (threadIdx.x == 0) {
    sum_reverse_block = (1.f) / sum;
  }
  __syncthreads();

  // threadIdx.x might be larger than all_sequence_length due to alignment to 32x.
  for (int seq_idx = threadIdx.x; seq_idx < all_sequence_length; seq_idx += TPB) {
    const bool is_valid = (seq_idx >= valid_start && seq_idx < end);
    output[offset + seq_idx] = is_valid ? T(cached_data[seq_idx] * sum_reverse_block) : T(0.f);
  }
}

template <typename T, int TPB>
__global__ void SoftmaxWithRawMaskLargeKernel(const int all_sequence_length,
                                              const int sequence_length,
                                              const int* attention_mask,  // 2D, 3D or 4D attention mask
                                              const bool* key_padding_mask,
                                              const T* rel_pos_bias,
                                              const bool broadcast_rel_pos_bias,
                                              const T* input,
                                              T* output,
                                              const bool causal,
                                              const float rsqrt_head_size,
                                              const int mask_dimension,
                                              const int max_sequence_length,
                                              const bool skip_softmax,
                                              const float mask_filter_value) {
  extern __shared__ float cached_data[];  // float[all_sequence_length]

  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  float max_thread_data = -CUDART_INF_F;
  const int size_per_batch = gridDim.x * all_sequence_length;

  // Input dimension is BxNxSxS*; blockIdx.y is batch index b; gridDim.x=N*S;  blockIdx.x is index within N*S;
  int base_index = (blockIdx.y * gridDim.x + blockIdx.x) * all_sequence_length;
  for (int seq_idx = threadIdx.x; seq_idx < all_sequence_length; seq_idx += TPB) {
    float thread_data = -CUDART_INF_F;
    int index = base_index + seq_idx;
    if (rel_pos_bias == nullptr) {
      thread_data = float(input[index]) * rsqrt_head_size;
    } else {
      T rel_pos_bias_value = broadcast_rel_pos_bias ? rel_pos_bias[index % size_per_batch] : rel_pos_bias[index];
      thread_data = float(input[index] + rel_pos_bias_value) * rsqrt_head_size;
    }

    const int sequence_index = blockIdx.x % sequence_length;
    if (causal) {
      int from_index = all_sequence_length - sequence_length + sequence_index;  // offset in all sequence length.
      if (seq_idx > from_index) {
        thread_data = -CUDART_INF_F;
      }
    }

    int mask_offset = 0;
    const int batch_index = blockIdx.y;
    if (mask_dimension == 2) {
      mask_offset = batch_index * all_sequence_length + seq_idx;
    } else if (mask_dimension == 3) {
      mask_offset = (batch_index * sequence_length + sequence_index) * all_sequence_length + seq_idx;
    } else if (mask_dimension == 4) {
      int from_index = all_sequence_length - sequence_length + sequence_index;
      mask_offset = (batch_index * max_sequence_length + from_index) * max_sequence_length + seq_idx;
    }

    if (nullptr == key_padding_mask) {
      const int& mask = attention_mask[mask_offset];
      if (mask == 0)
        thread_data += mask_filter_value;
    } else {
      const bool mask = key_padding_mask[mask_offset];
      if (mask) {
        thread_data = -CUDART_INF_F;
      }
    }

    if (skip_softmax) {
      output[index] = T(thread_data);
    }
    cached_data[seq_idx] = thread_data;
    max_thread_data = max(max_thread_data, thread_data);
  }

  if (skip_softmax) {
    return;
  }

  const float max = BlockReduce(tmp_storage).Reduce(max_thread_data, cub::Max(), all_sequence_length);

  // Store max value
  if (threadIdx.x == 0) {
    max_block = max;
  }
  __syncthreads();

  float sum_thread_data_exp = 0.0f;
  for (int seq_idx = threadIdx.x; seq_idx < all_sequence_length; seq_idx += TPB) {
    auto ev = expf(cached_data[seq_idx] - max_block);
    cached_data[seq_idx] = ev;
    sum_thread_data_exp += ev;
  }
  const auto sum = BlockReduce(tmp_storage).Reduce(sum_thread_data_exp, cub::Sum(), TPB);

  // Store value of 1.0/sum
  if (threadIdx.x == 0) {
    sum_reverse_block = (1.f) / sum;
  }
  __syncthreads();

  for (int seq_idx = threadIdx.x; seq_idx < all_sequence_length; seq_idx += TPB) {
    output[base_index + seq_idx] = T(cached_data[seq_idx] * sum_reverse_block);
  }
}

template <typename T, unsigned TPB>
__device__ inline void SoftmaxWithRawMaskSmall(const int all_sequence_length,
                                               const int sequence_length,
                                               const int* attention_mask,  // 2D, 3D or 4D attention mask
                                               const bool* key_padding_mask,
                                               const T* rel_pos_bias,
                                               const bool broadcast_rel_pos_bias,
                                               const T* input,
                                               T* output,
                                               const bool causal,
                                               const float rsqrt_head_size,
                                               const int mask_dimension,
                                               const int max_sequence_length,
                                               const bool skip_softmax,
                                               const float mask_filter_value) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  // Input dimension is BxNxSxS*; blockIdx.y is batch index b; gridDim.x=N*S;  blockIdx.x is index within N*S;
  int index = (blockIdx.y * gridDim.x + blockIdx.x) * all_sequence_length + threadIdx.x;
  const int size_per_batch = gridDim.x * all_sequence_length;

  float thread_data = -CUDART_INF_F;
  if (threadIdx.x < all_sequence_length) {
    thread_data = float(input[index]) * rsqrt_head_size;

    const int sequence_index = blockIdx.x % sequence_length;
    if (causal) {
      int from_index = all_sequence_length - sequence_length + sequence_index;  // offset in all sequence length.
      if (threadIdx.x > from_index) {
        thread_data = -CUDART_INF_F;
      }
    }

    int mask_offset = 0;
    const int batch_index = blockIdx.y;
    if (mask_dimension == 2) {
      mask_offset = batch_index * all_sequence_length + threadIdx.x;
    } else if (mask_dimension == 3) {
      mask_offset = (batch_index * sequence_length + sequence_index) * all_sequence_length + threadIdx.x;
    } else if (mask_dimension == 4) {
      int from_index = all_sequence_length - sequence_length + sequence_index;
      mask_offset = (batch_index * max_sequence_length + from_index) * max_sequence_length + threadIdx.x;
    }

    if (nullptr == key_padding_mask) {
      const int& mask = attention_mask[mask_offset];
      if (mask == 0)
        thread_data += mask_filter_value;
    } else {
      const bool mask = key_padding_mask[mask_offset];
      if (mask) {
        thread_data = -CUDART_INF_F;
      }
    }

    if (rel_pos_bias != nullptr) {
      float bias = broadcast_rel_pos_bias ? float(rel_pos_bias[index % size_per_batch]) : float(rel_pos_bias[index]);
      thread_data += bias;
    }
  }

  if (skip_softmax) {
    if (threadIdx.x < all_sequence_length) {
      output[index] = T(thread_data);
    }
    return;
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
__global__ void SoftmaxKernelSmall(const int all_sequence_length,
                                   const int sequence_length,
                                   const T* rel_pos_bias,
                                   const bool broadcast_rel_pos_bias,
                                   const T* input,
                                   T* output,
                                   bool causal) {
  SoftmaxSmall<T, TPB>(all_sequence_length, sequence_length, all_sequence_length, 0,
                       rel_pos_bias, broadcast_rel_pos_bias, input, output, causal);
}

template <typename T, unsigned TPB>
__global__ void SoftmaxKernel(const int all_sequence_length,
                              const T* rel_pos_bias,
                              const bool broadcast_rel_pos_bias,
                              const T* input,
                              T* output) {
  Softmax<T, TPB>(all_sequence_length, all_sequence_length, 0,
                  rel_pos_bias, broadcast_rel_pos_bias, input, output);
}

template <typename T>
Status ComputeSoftmax(cudaStream_t stream, const int all_sequence_length, const int sequence_length,
                      const int batch_size, const int num_heads, const T* rel_pos_bias,
                      const bool broadcast_rel_pos_bias, T* input, T* output, bool causal) {
  const dim3 grid(sequence_length * num_heads, batch_size, 1);
  if (all_sequence_length <= 32) {
    const int blockSize = 32;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(
        all_sequence_length, sequence_length, rel_pos_bias, broadcast_rel_pos_bias, input, output, causal);
  } else if (all_sequence_length <= 64) {
    const int blockSize = 64;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(
        all_sequence_length, sequence_length, rel_pos_bias, broadcast_rel_pos_bias, input, output, causal);
  } else if (all_sequence_length <= 128) {
    const int blockSize = 128;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(
        all_sequence_length, sequence_length, rel_pos_bias, broadcast_rel_pos_bias, input, output, causal);
  } else if (all_sequence_length <= 256) {
    const int blockSize = 256;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(
        all_sequence_length, sequence_length, rel_pos_bias, broadcast_rel_pos_bias, input, output, causal);
  } else if (all_sequence_length <= 512) {
    const int blockSize = 512;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(
        all_sequence_length, sequence_length, rel_pos_bias, broadcast_rel_pos_bias, input, output, causal);
  } else if (all_sequence_length <= 1024) {
    const int blockSize = 1024;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(
        all_sequence_length, sequence_length, rel_pos_bias, broadcast_rel_pos_bias, input, output, causal);
  } else if (!causal) {
    const int blockSize = 1024;
    SoftmaxKernel<T, blockSize><<<grid, blockSize, 0, stream>>>(
        all_sequence_length, rel_pos_bias, broadcast_rel_pos_bias, input, output);
  } else {
    const int blockSize = 256;
    const int sh_bytes = sizeof(float) * all_sequence_length;
    SoftmaxLargeKernel<T, blockSize><<<grid, blockSize, sh_bytes, stream>>>(
        all_sequence_length, sequence_length, all_sequence_length, 0, rel_pos_bias, broadcast_rel_pos_bias,
        input, output, true);
  }

  return CUDA_CALL(cudaGetLastError());
}

template <typename T, unsigned TPB>
__global__ void MaskedSoftmaxKernelSmall(const int all_sequence_length,
                                         const int sequence_length,
                                         const int* mask_end,
                                         const int* mask_start,
                                         const T* rel_pos_bias,
                                         const bool broadcast_rel_pos_bias,
                                         const T* input,
                                         T* output,
                                         bool causal) {
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

  SoftmaxSmall<T, TPB>(all_sequence_length, sequence_length, end_position, start_position,
                       rel_pos_bias, broadcast_rel_pos_bias, input, output, causal);
}

template <typename T, unsigned TPB>
__device__ inline void SoftmaxSmallPacked(const int sequence_length,
                                          const int end,
                                          const T* rel_pos_bias,
                                          const bool broadcast_rel_pos_bias,
                                          const T* input,
                                          T* output) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  // Input dimension is BxNxSxS*; blockIdx.y is batch index b; gridDim.x=N*S;  blockIdx.x is index within N*S;
  const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * sequence_length;
  const int index = offset + threadIdx.x;

  bool is_valid = threadIdx.x < end;

  // e^x is represented as infinity if x is large enough, like 100.f.
  // Infinity divided by Infinity is a NAN. Thus, softmax gets a NAN if one or more item are large enough.
  // a math transform as below is leveraged to get a stable softmax:
  // e^xi/(e^x1 + ...e^xn) = e^(xi - max) / (e^(x1 - max) + ... + e^(xn - max))
  const bool no_rpb = (rel_pos_bias == nullptr);
  const int size_per_batch = gridDim.x * sequence_length;
  float input_data = no_rpb
                         ? float(input[index])
                         : float(input[index] + (broadcast_rel_pos_bias
                                                     ? rel_pos_bias[index % size_per_batch]
                                                     : rel_pos_bias[index]));

  float thread_data_max = is_valid ? input_data : float(-CUDART_INF_F);
  const auto max = BlockReduce(tmp_storage).Reduce(thread_data_max, cub::Max(), end);

  // Store max value
  if (threadIdx.x == 0) {
    max_block = max;
  }
  __syncthreads();

  float thread_data_exp(0.f);
  if (is_valid) {
    thread_data_exp = expf(input_data - max_block);
  }

  const auto sum = BlockReduce(tmp_storage).Reduce(thread_data_exp, cub::Sum(), end);

  // Store value of 1.0/sum.
  if (threadIdx.x == 0) {
    sum_reverse_block = (1.f) / sum;
  }
  __syncthreads();

  // threadIdx.x might be larger than all_sequence_length due to alignment to 32x.
  if (threadIdx.x < sequence_length) {
    output[index] = T(thread_data_exp * sum_reverse_block);
  }
}

template <typename T, unsigned TPB>
__global__ void SoftmaxKernelSmallWithCumSeqLen(const T* input,
                                                const T* rel_pos_bias, const bool broadcast_rel_pos_bias,
                                                const int* cum_seq_length, const int sequence_length,
                                                T* output) {
  __shared__ int end_position;

  if (threadIdx.x == 0) {
    const int batch = blockIdx.y;
    end_position = cum_seq_length[batch + 1] - cum_seq_length[batch];
  }
  __syncthreads();

  SoftmaxSmallPacked<T, TPB>(sequence_length, end_position,
                             rel_pos_bias, broadcast_rel_pos_bias,
                             input, output);
}

template <typename T, unsigned TPB>
__global__ void SoftmaxKernelWithCumSeqLen(const T* input,
                                           const T* rel_pos_bias, const bool broadcast_rel_pos_bias,
                                           const int* cum_seq_length, const int sequence_length,
                                           T* output) {
  __shared__ int end_position;

  if (threadIdx.x == 0) {
    const int batch = blockIdx.y;
    end_position = cum_seq_length[batch + 1] - cum_seq_length[batch];
  }
  __syncthreads();

  Softmax<T, TPB>(sequence_length, end_position, 0 /*start_position*/,
                  rel_pos_bias, broadcast_rel_pos_bias, input, output);
}

template <typename T, unsigned TPB>
__global__ void MaskedSoftmaxKernel(const int all_sequence_length,
                                    const int* mask_end,
                                    const int* mask_start,
                                    const T* rel_pos_bias,
                                    const bool broadcast_rel_pos_bias,
                                    const T* input, T* output) {
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

  Softmax<T, TPB>(all_sequence_length, end_position, start_position,
                  rel_pos_bias, broadcast_rel_pos_bias, input, output);
}

template <typename T, unsigned TPB>
__global__ void SoftmaxWithRawMaskSmallKernel(const int all_sequence_length,
                                              const int sequence_length,
                                              const int* attention_mask,
                                              const bool* key_padding_mask,
                                              const T* rel_pos_bias,
                                              const bool broadcast_rel_pos_bias,
                                              const T* input,
                                              T* output,
                                              const bool causal,
                                              const float rsqrt_head_size,
                                              const int mask_dimension,
                                              const int max_sequence_length,
                                              const bool skip_softmax,
                                              const float mask_filter_value) {
  SoftmaxWithRawMaskSmall<T, TPB>(
      all_sequence_length, sequence_length,
      attention_mask, key_padding_mask, rel_pos_bias, broadcast_rel_pos_bias, input, output,
      causal, rsqrt_head_size, mask_dimension, max_sequence_length,
      skip_softmax, mask_filter_value);
}

template <typename T>
Status ComputeSoftmaxWithCumSeqLength(
    const T* input,
    const T* rel_pos_bias,
    const bool broadcast_rel_pos_bias,
    const int32_t* cum_seq_length,
    const int batch_size,
    const int sequence_length,
    const int num_heads,
    T* output, cudaStream_t stream) {
  const dim3 grid(sequence_length * num_heads, batch_size, 1);

  if (sequence_length <= 32) {
    const int blockSize = 32;
    SoftmaxKernelSmallWithCumSeqLen<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(input, rel_pos_bias, broadcast_rel_pos_bias,
                                         cum_seq_length, sequence_length, output);

  } else if (sequence_length <= 64) {
    const int blockSize = 64;
    SoftmaxKernelSmallWithCumSeqLen<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(input, rel_pos_bias, broadcast_rel_pos_bias,
                                         cum_seq_length, sequence_length, output);
  } else if (sequence_length <= 128) {
    const int blockSize = 128;
    SoftmaxKernelSmallWithCumSeqLen<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(input, rel_pos_bias, broadcast_rel_pos_bias,
                                         cum_seq_length, sequence_length, output);
  } else if (sequence_length <= 256) {
    const int blockSize = 256;
    SoftmaxKernelSmallWithCumSeqLen<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(input, rel_pos_bias, broadcast_rel_pos_bias,
                                         cum_seq_length, sequence_length, output);
  } else if (sequence_length <= 512) {
    const int blockSize = 512;
    SoftmaxKernelSmallWithCumSeqLen<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(input, rel_pos_bias, broadcast_rel_pos_bias,
                                         cum_seq_length, sequence_length, output);
  } else if (sequence_length <= 1024) {
    const int blockSize = 1024;
    SoftmaxKernelSmallWithCumSeqLen<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(input, rel_pos_bias, broadcast_rel_pos_bias,
                                         cum_seq_length, sequence_length, output);
  } else {
    SoftmaxKernelWithCumSeqLen<T, 1024>
        <<<grid, 1024, 0, stream>>>(input, rel_pos_bias, broadcast_rel_pos_bias,
                                    cum_seq_length, sequence_length, output);
  }

  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status ComputeSoftmaxWithMask1D(cudaStream_t stream,
                                const int all_sequence_length,
                                const int sequence_length,
                                const int batch_size,
                                const int num_heads,
                                const int* mask_index,
                                const int* mask_start,
                                const T* rel_pos_bias,
                                const bool broadcast_rel_pos_bias,
                                const T* input,
                                T* output,
                                const bool causal) {
  const dim3 grid(sequence_length * num_heads, batch_size, 1);

  if (all_sequence_length <= 32) {
    const int blockSize = 32;
    MaskedSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start,
                                         rel_pos_bias, broadcast_rel_pos_bias, input, output, causal);
  } else if (all_sequence_length <= 64) {
    const int blockSize = 64;
    MaskedSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start,
                                         rel_pos_bias, broadcast_rel_pos_bias, input, output, causal);
  } else if (all_sequence_length <= 128) {
    const int blockSize = 128;
    MaskedSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start,
                                         rel_pos_bias, broadcast_rel_pos_bias, input, output, causal);
  } else if (all_sequence_length <= 256) {
    const int blockSize = 256;
    MaskedSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start,
                                         rel_pos_bias, broadcast_rel_pos_bias, input, output, causal);
  } else if (all_sequence_length <= 512) {
    const int blockSize = 512;
    MaskedSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start,
                                         rel_pos_bias, broadcast_rel_pos_bias, input, output, causal);
  } else if (all_sequence_length <= 1024) {
    const int blockSize = 1024;
    MaskedSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start,
                                         rel_pos_bias, broadcast_rel_pos_bias, input, output, causal);
  } else if (!causal) {
    const int blockSize = 1024;
    MaskedSoftmaxKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, mask_index, mask_start,
                                         rel_pos_bias, broadcast_rel_pos_bias, input, output);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Attention CUDA operator does not support total sequence length > 1024.");
  }

  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status ComputeSoftmaxWithRawMask(Stream* ort_stream,
                                 const int all_sequence_length,
                                 const int sequence_length,
                                 const int batch_size,
                                 const int num_heads,
                                 const int* attention_mask,
                                 const bool* key_padding_mask,
                                 const T* rel_pos_bias,
                                 const bool broadcast_rel_pos_bias,
                                 const T* input,
                                 T* output,
                                 const bool causal,
                                 const float rsqrt_head_size,
                                 const int mask_dimension,
                                 const int max_sequence_length,
                                 const bool use_persistent_softmax,
                                 T* persistent_softmax_workspace,
                                 const float mask_filter_value) {
  auto stream = static_cast<cudaStream_t>(ort_stream->GetHandle());
  const dim3 grid(sequence_length * num_heads, batch_size, 1);

  T* out = use_persistent_softmax ? persistent_softmax_workspace : output;
  if (all_sequence_length <= 32) {
    const int blockSize = 32;
    SoftmaxWithRawMaskSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length,
                                         attention_mask, key_padding_mask, rel_pos_bias, broadcast_rel_pos_bias, input,
                                         out, causal, rsqrt_head_size, mask_dimension, max_sequence_length,
                                         use_persistent_softmax, mask_filter_value);
  } else if (all_sequence_length <= 64) {
    const int blockSize = 64;
    SoftmaxWithRawMaskSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length,
                                         attention_mask, key_padding_mask, rel_pos_bias, broadcast_rel_pos_bias, input,
                                         out, causal, rsqrt_head_size, mask_dimension, max_sequence_length,
                                         use_persistent_softmax, mask_filter_value);
  } else if (all_sequence_length <= 128) {
    const int blockSize = 128;
    SoftmaxWithRawMaskSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length,
                                         attention_mask, key_padding_mask, rel_pos_bias, broadcast_rel_pos_bias, input,
                                         out, causal, rsqrt_head_size, mask_dimension, max_sequence_length,
                                         use_persistent_softmax, mask_filter_value);
  } else if (all_sequence_length <= 256) {
    const int blockSize = 256;
    SoftmaxWithRawMaskSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length,
                                         attention_mask, key_padding_mask, rel_pos_bias, broadcast_rel_pos_bias, input,
                                         out, causal, rsqrt_head_size, mask_dimension, max_sequence_length,
                                         use_persistent_softmax, mask_filter_value);
  } else if (all_sequence_length <= 512) {
    const int blockSize = 512;
    SoftmaxWithRawMaskSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length,
                                         attention_mask, key_padding_mask, rel_pos_bias, broadcast_rel_pos_bias, input,
                                         out, causal, rsqrt_head_size, mask_dimension, max_sequence_length,
                                         use_persistent_softmax, mask_filter_value);
  } else if (all_sequence_length <= 1024) {
    const int blockSize = 1024;
    SoftmaxWithRawMaskSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length,
                                         attention_mask, key_padding_mask, rel_pos_bias, broadcast_rel_pos_bias, input,
                                         out, causal, rsqrt_head_size, mask_dimension, max_sequence_length,
                                         use_persistent_softmax, mask_filter_value);
  } else {
    const int blockSize = 256;
    const int sh_bytes = sizeof(float) * all_sequence_length;
    SoftmaxWithRawMaskLargeKernel<T, blockSize>
        <<<grid, blockSize, sh_bytes, stream>>>(
            all_sequence_length, sequence_length,
            attention_mask, key_padding_mask, rel_pos_bias, broadcast_rel_pos_bias, input,
            out, causal, rsqrt_head_size, mask_dimension, max_sequence_length,
            use_persistent_softmax, mask_filter_value);
  }

  if (use_persistent_softmax) {
    return onnxruntime::cuda::dispatch_warpwise_softmax_forward<T, T, float, false>(
        ort_stream,
        output,
        persistent_softmax_workspace,
        all_sequence_length,
        all_sequence_length,
        batch_size * num_heads * sequence_length);
  }

  return CUDA_CALL(cudaGetLastError());
}

// Template Instantiation
template Status ComputeSoftmax<float>(
    cudaStream_t stream, const int all_sequence_length, const int sequence_length,
    const int batch_size, const int num_heads, const float* rel_pos_bias,
    const bool broadcast_rel_pos_bias, float* input, float* output, bool causal);

template Status ComputeSoftmax<half>(
    cudaStream_t stream, const int all_sequence_length, const int sequence_length,
    const int batch_size, const int num_heads, const half* rel_pos_bias,
    const bool broadcast_rel_pos_bias, half* input, half* output, bool causal);

template Status ComputeSoftmaxWithCumSeqLength<float>(
    const float* input,
    const float* rel_pos_bias,
    const bool broadcast_rel_pos_bias,
    const int32_t* cum_seq_length,
    const int batch_size,
    const int sequence_length,
    const int num_heads,
    float* output, cudaStream_t stream);

template Status ComputeSoftmaxWithCumSeqLength<half>(
    const half* input,
    const half* rel_pos_bias,
    const bool broadcast_rel_pos_bias,
    const int32_t* cum_seq_length,
    const int batch_size,
    const int sequence_length,
    const int num_heads,
    half* output, cudaStream_t stream);

template Status ComputeSoftmaxWithMask1D<float>(cudaStream_t stream,
                                                const int all_sequence_length,
                                                const int sequence_length,
                                                const int batch_size,
                                                const int num_heads,
                                                const int* mask_index,
                                                const int* mask_start,
                                                const float* rel_pos_bias,
                                                const bool broadcast_rel_pos_bias,
                                                const float* input,
                                                float* output,
                                                const bool causal);

template Status ComputeSoftmaxWithMask1D<half>(cudaStream_t stream,
                                               const int all_sequence_length,
                                               const int sequence_length,
                                               const int batch_size,
                                               const int num_heads,
                                               const int* mask_index,
                                               const int* mask_start,
                                               const half* rel_pos_bias,
                                               const bool broadcast_rel_pos_bias,
                                               const half* input,
                                               half* output,
                                               const bool causal);

template Status ComputeSoftmaxWithRawMask<float>(Stream* ort_stream,
                                                 const int all_sequence_length,
                                                 const int sequence_length,
                                                 const int batch_size,
                                                 const int num_heads,
                                                 const int* attention_mask,
                                                 const bool* key_padding_mask,
                                                 const float* rel_pos_bias,
                                                 const bool broadcast_rel_pos_bias,
                                                 const float* input,
                                                 float* output,
                                                 const bool causal,
                                                 const float rsqrt_head_size,
                                                 const int mask_dimension,
                                                 const int max_sequence_length,
                                                 const bool use_persistent_softmax,
                                                 float* persistent_softmax_workspace,
                                                 const float mask_filter_value);

template Status ComputeSoftmaxWithRawMask<half>(Stream* ort_stream,
                                                const int all_sequence_length,
                                                const int sequence_length,
                                                const int batch_size,
                                                const int num_heads,
                                                const int* attention_mask,
                                                const bool* key_padding_mask,
                                                const half* rel_pos_bias,
                                                const bool broadcast_rel_pos_bias,
                                                const half* input,
                                                half* output,
                                                const bool causal,
                                                const float rsqrt_head_size,
                                                const int mask_dimension,
                                                const int max_sequence_length,
                                                const bool use_persistent_softmax,
                                                half* persistent_softmax_workspace,
                                                const float mask_filter_value);

}  // namespace attention_softmax_cuda
}  // namespace contrib
}  // namespace onnxruntime
