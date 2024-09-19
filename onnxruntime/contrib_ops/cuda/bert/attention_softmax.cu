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

#define DISPATCH_BIAS(attn_bias, HAS_BIAS, ...)                 \
  [&] {                                                         \
    const dim3 grid(num_heads* sequence_length, batch_size, 1); \
    if (attn_bias != nullptr) {                                 \
      constexpr static bool HAS_BIAS = true;                    \
      return __VA_ARGS__();                                     \
    } else {                                                    \
      constexpr static bool HAS_BIAS = false;                   \
      return __VA_ARGS__();                                     \
    }                                                           \
  }()

// Macro to declare variables:
//   offset: offset in input/output
//   bias_offset: offset in attn_bias
//   b: batch index
//   s: sequence index
// grid size is (num_heads * sequence_length, batch_size, 1)
// input and output shape is (batch_size, num_heads, sequence_length, total_sequence_length)
// bias shape is (batch_size or 1, num_heads or 1, sequence_length, total_sequence_length)
#define DECLARE_SOFTMAX_VARS()                                                                                      \
  [[maybe_unused]] const int s = blockIdx.x % sequence_length;                                                      \
  const int b = blockIdx.y;                                                                                         \
  int64_t offset = static_cast<int64_t>(b * gridDim.x + blockIdx.x) * static_cast<int64_t>(total_sequence_length);  \
  [[maybe_unused]] int64_t bias_offset = 0;                                                                         \
  if constexpr (HAS_BIAS) {                                                                                         \
    const int j = (broadcast_attn_bias_dim_0 ? 0 : (b * gridDim.x)) + (broadcast_attn_bias_dim_1 ? s : blockIdx.x); \
    bias_offset = static_cast<int64_t>(j) * static_cast<int64_t>(total_sequence_length);                            \
  }

// This kernel is for non causal, attention mask 1D or None, and total_sequence_length > 1024.
template <typename T, unsigned TPB, bool HAS_BIAS>
__device__ inline void Softmax(const int total_sequence_length,
                               const int sequence_length,
                               const int valid_end,
                               const int valid_start,
                               const T* attn_bias,
                               const bool broadcast_attn_bias_dim_0,
                               const bool broadcast_attn_bias_dim_1,
                               const T* input,
                               T* output) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  float thread_data_max(-CUDART_INF_F);

  DECLARE_SOFTMAX_VARS();

  // e^x is represented as infinity if x is large enough, like 100.f.
  // Infinity divided by Infinity is a NAN. Thus, softmax gets a NAN if one or more item are large enough.
  // a math transform as below is leveraged to get a stable softmax:
  // e^xi/(e^x1 + ...e^xn) = e^(xi - max) / (e^(x1 - max) + ... + e^(xn - max))
  for (int i = threadIdx.x; i < valid_end; i += TPB) {
    if (i >= valid_start) {
      float input_data = HAS_BIAS
                             ? float(input[offset + i]) + float(attn_bias[bias_offset + i])
                             : float(input[offset + i]);
      if (thread_data_max < input_data) {
        thread_data_max = input_data;
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
      float input_data = HAS_BIAS
                             ? float(input[offset + i]) + float(attn_bias[bias_offset + i])
                             : float(input[offset + i]);

      thread_data_sum += expf(input_data - max_block);
    }
  }

  const auto sum = BlockReduce(tmp_storage).Reduce(thread_data_sum, cub::Sum());
  if (threadIdx.x == 0) {
    sum_reverse_block = 1.f / sum;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < total_sequence_length; i += TPB) {
    const int index = offset + i;
    float input_data = HAS_BIAS
                           ? float(input[index]) + float(attn_bias[bias_offset + i])
                           : float(input[index]);
    const float val = (i >= valid_start && i < valid_end) ? expf(input_data - max_block) * sum_reverse_block : 0.f;
    output[index] = T(val);
  }
}

// This kernel is for non causal, attention mask 1D or None, and total_sequence_length <= 1024.
template <typename T, unsigned TPB, bool HAS_BIAS>
__device__ inline void SoftmaxSmall(const int total_sequence_length,
                                    const int sequence_length,
                                    const int valid_end,
                                    const int valid_start,
                                    const T* attn_bias,
                                    const bool broadcast_attn_bias_dim_0,
                                    const bool broadcast_attn_bias_dim_1,
                                    const T* input,
                                    T* output,
                                    bool causal) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  DECLARE_SOFTMAX_VARS();

  const int index = offset + threadIdx.x;

  // Update end position for causal.
  int end = valid_end;
  if (causal) {
    const int end_causal = total_sequence_length - sequence_length + s + 1;
    if (end_causal < end) {
      end = end_causal;
    }
  }

  const bool is_valid = (threadIdx.x >= valid_start && threadIdx.x < end);
  float input_data = is_valid ? (HAS_BIAS
                                     ? float(input[index]) + float(attn_bias[bias_offset + threadIdx.x])
                                     : float(input[index]))
                              : float(-CUDART_INF_F);

  // e^x is represented as infinity if x is large enough, like 100.f.
  // Infinity divided by Infinity is a NAN. Thus, softmax gets a NAN if one or more item are large enough.
  // a math transform as below is leveraged to get a stable softmax:
  // e^xi/(e^x1 + ...e^xn) = e^(xi - max) / (e^(x1 - max) + ... + e^(xn - max))
  const auto max = BlockReduce(tmp_storage).Reduce(input_data, cub::Max(), end);

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

  // threadIdx.x might be larger than total_sequence_length due to alignment to 32x.
  if (threadIdx.x < total_sequence_length) {
    output[index] = is_valid ? T(thread_data_exp * sum_reverse_block) : T(0.f);
  }
}

// This kernel is for causal or not, attention mask 1D or None, and total_sequence_length <= 1024.
template <typename T, unsigned TPB, bool HAS_BIAS>
__global__ void SoftmaxLargeKernel(const int total_sequence_length,
                                   const int sequence_length,
                                   const int valid_end,
                                   const int valid_start,
                                   const T* attn_bias,
                                   const bool broadcast_attn_bias_dim_0,
                                   const bool broadcast_attn_bias_dim_1,
                                   const T* input,
                                   T* output,
                                   bool causal) {
  extern __shared__ float cached_data[];  // float[total_sequence_length]

  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  DECLARE_SOFTMAX_VARS();

  // Update end position for causal.
  int end = valid_end;
  if (causal) {
    int end_causal = total_sequence_length - sequence_length + s + 1;
    if (end_causal < end) {
      end = end_causal;
    }
  }

  float thread_data_max = -CUDART_INF_F;
  for (int i = threadIdx.x; i < total_sequence_length; i += TPB) {
    const int index = offset + i;
    const bool is_valid = (i >= valid_start && i < end);
    float input_data = is_valid ? (HAS_BIAS
                                       ? float(input[index]) + float(attn_bias[bias_offset + i])
                                       : float(input[index]))
                                : float(-CUDART_INF_F);
    cached_data[i] = input_data;
    thread_data_max = max(thread_data_max, input_data);
  }
  const auto max = BlockReduce(tmp_storage).Reduce(thread_data_max, cub::Max(), end);

  // Store max value
  if (threadIdx.x == 0) {
    max_block = max;
  }
  __syncthreads();

  float thread_data_exp(0.f);
  for (int i = threadIdx.x; i < total_sequence_length; i += TPB) {
    const bool is_valid = (i >= valid_start && i < end);
    cached_data[i] = is_valid ? expf(cached_data[i] - max_block) : 0.0f;
    thread_data_exp += cached_data[i];
  }
  const auto sum = BlockReduce(tmp_storage).Reduce(thread_data_exp, cub::Sum(), end);

  // Store value of 1.0/sum.
  if (threadIdx.x == 0) {
    sum_reverse_block = (1.f) / sum;
  }
  __syncthreads();

  // threadIdx.x might be larger than total_sequence_length due to alignment to 32x.
  for (int i = threadIdx.x; i < total_sequence_length; i += TPB) {
    const bool is_valid = (i >= valid_start && i < end);
    output[offset + i] = is_valid ? T(cached_data[i] * sum_reverse_block) : T(0.f);
  }
}

// This kernel is for causal or not, raw attention mask (2D, 3D or 4D) and total_sequence_length > 1024.
template <typename T, int TPB, bool HAS_BIAS>
__global__ void SoftmaxWithRawMaskLargeKernel(const int total_sequence_length,
                                              const int sequence_length,
                                              const int* attention_mask,  // 2D, 3D or 4D attention mask
                                              const bool* key_padding_mask,
                                              const T* attn_bias,
                                              const bool broadcast_attn_bias_dim_0,
                                              const bool broadcast_attn_bias_dim_1,
                                              const T* input,
                                              T* output,
                                              const bool causal,
                                              const float rsqrt_head_size,
                                              const int mask_dimension,
                                              const int max_sequence_length,
                                              const bool skip_softmax,
                                              const float mask_filter_value) {
  extern __shared__ float cached_data[];  // float[total_sequence_length]

  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  float max_thread_data = -CUDART_INF_F;

  DECLARE_SOFTMAX_VARS();

  for (int i = threadIdx.x; i < total_sequence_length; i += TPB) {
    int index = offset + i;
    float input_data = HAS_BIAS
                           ? float(input[index]) + float(attn_bias[bias_offset + i])
                           : float(input[index]);
    float thread_data = input_data * rsqrt_head_size;
    if (causal) {
      int from_index = total_sequence_length - sequence_length + s;  // offset in total sequence length.
      if (i > from_index) {
        thread_data = -CUDART_INF_F;
      }
    }

    int mask_offset = 0;
    if (mask_dimension == 2) {
      mask_offset = b * total_sequence_length + i;
    } else if (mask_dimension == 3) {
      mask_offset = (b * sequence_length + s) * total_sequence_length + i;
    } else if (mask_dimension == 4) {
      int from_index = total_sequence_length - sequence_length + s;
      mask_offset = (b * max_sequence_length + from_index) * max_sequence_length + i;
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
    cached_data[i] = thread_data;
    max_thread_data = max(max_thread_data, thread_data);
  }

  if (skip_softmax) {
    return;
  }

  const float max = BlockReduce(tmp_storage).Reduce(max_thread_data, cub::Max(), total_sequence_length);

  // Store max value
  if (threadIdx.x == 0) {
    max_block = max;
  }
  __syncthreads();

  float sum_thread_data_exp = 0.0f;
  for (int i = threadIdx.x; i < total_sequence_length; i += TPB) {
    auto ev = expf(cached_data[i] - max_block);
    cached_data[i] = ev;
    sum_thread_data_exp += ev;
  }
  const auto sum = BlockReduce(tmp_storage).Reduce(sum_thread_data_exp, cub::Sum(), TPB);

  // Store value of 1.0/sum
  if (threadIdx.x == 0) {
    sum_reverse_block = (1.f) / sum;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < total_sequence_length; i += TPB) {
    output[offset + i] = T(cached_data[i] * sum_reverse_block);
  }
}

// This kernel is for causal or not, raw attention mask (2D, 3D or 4D), and total_sequence_length <= 1024.
template <typename T, unsigned TPB, bool HAS_BIAS>
__device__ inline void SoftmaxWithRawMaskSmall(const int total_sequence_length,
                                               const int sequence_length,
                                               const int* attention_mask,  // 2D, 3D or 4D attention mask
                                               const bool* key_padding_mask,
                                               const T* attn_bias,
                                               const bool broadcast_attn_bias_dim_0,
                                               const bool broadcast_attn_bias_dim_1,
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

  DECLARE_SOFTMAX_VARS();

  int64_t index = offset + threadIdx.x;

  float thread_data = -CUDART_INF_F;
  if (threadIdx.x < total_sequence_length) {
    thread_data = float(input[index]) * rsqrt_head_size;

    if (causal) {
      int from_index = total_sequence_length - sequence_length + s;  // offset in total sequence length.
      if (threadIdx.x > from_index) {
        thread_data = -CUDART_INF_F;
      }
    }

    int mask_offset = 0;
    if (mask_dimension == 2) {
      mask_offset = b * total_sequence_length + threadIdx.x;
    } else if (mask_dimension == 3) {
      mask_offset = (b * sequence_length + s) * total_sequence_length + threadIdx.x;
    } else if (mask_dimension == 4) {
      int from_index = total_sequence_length - sequence_length + s;
      mask_offset = (b * max_sequence_length + from_index) * max_sequence_length + threadIdx.x;
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

    if (HAS_BIAS) {
      thread_data += float(attn_bias[bias_offset + threadIdx.x]);
    }
  }

  if (skip_softmax) {
    if (threadIdx.x < total_sequence_length) {
      output[index] = T(thread_data);
    }
    return;
  }

  const float max = BlockReduce(tmp_storage).Reduce(thread_data, cub::Max(), total_sequence_length);

  // Store max value
  if (threadIdx.x == 0) {
    max_block = max;
  }
  __syncthreads();

  float thread_data_exp = threadIdx.x < total_sequence_length ? expf(thread_data - max_block) : 0.0f;
  const auto sum = BlockReduce(tmp_storage).Reduce(thread_data_exp, cub::Sum(), total_sequence_length);

  // Store value of 1.0/sum
  if (threadIdx.x == 0) {
    sum_reverse_block = (1.f) / sum;
  }
  __syncthreads();

  if (threadIdx.x < total_sequence_length) {
    output[index] = T(thread_data_exp * sum_reverse_block);
  }
}

template <typename T, unsigned TPB, bool HAS_BIAS>
__global__ void SoftmaxKernelSmall(const int total_sequence_length,
                                   const int sequence_length,
                                   const T* attn_bias,
                                   const bool broadcast_attn_bias_dim_0,
                                   const bool broadcast_attn_bias_dim_1,
                                   const T* input,
                                   T* output,
                                   bool causal) {
  SoftmaxSmall<T, TPB, HAS_BIAS>(total_sequence_length, sequence_length, total_sequence_length, 0,
                                 attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1, input, output, causal);
}

template <typename T, unsigned TPB, bool HAS_BIAS>
__global__ void SoftmaxKernel(const int total_sequence_length,
                              const int sequence_length,
                              const T* attn_bias,
                              const bool broadcast_attn_bias_dim_0,
                              const bool broadcast_attn_bias_dim_1,
                              const T* input,
                              T* output) {
  Softmax<T, TPB, HAS_BIAS>(total_sequence_length, sequence_length, total_sequence_length, 0,
                            attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1, input, output);
}

template <typename T>
Status ComputeSoftmax(cudaStream_t stream, const int total_sequence_length, const int sequence_length,
                      const int batch_size, const int num_heads, const T* attn_bias,
                      const bool broadcast_attn_bias_dim_0, const bool broadcast_attn_bias_dim_1,
                      T* input, T* output, bool causal) {
  DISPATCH_BIAS(attn_bias, HAS_BIAS, [&] {
    if (total_sequence_length <= 32) {
      const int blockSize = 32;
      SoftmaxKernelSmall<T, blockSize, HAS_BIAS><<<grid, blockSize, 0, stream>>>(
          total_sequence_length, sequence_length,
          attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1, input, output, causal);
    } else if (total_sequence_length <= 64) {
      const int blockSize = 64;
      SoftmaxKernelSmall<T, blockSize, HAS_BIAS><<<grid, blockSize, 0, stream>>>(
          total_sequence_length, sequence_length,
          attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1, input, output, causal);
    } else if (total_sequence_length <= 128) {
      const int blockSize = 128;
      SoftmaxKernelSmall<T, blockSize, HAS_BIAS><<<grid, blockSize, 0, stream>>>(
          total_sequence_length, sequence_length,
          attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1, input, output, causal);
    } else if (total_sequence_length <= 256) {
      const int blockSize = 256;
      SoftmaxKernelSmall<T, blockSize, HAS_BIAS><<<grid, blockSize, 0, stream>>>(
          total_sequence_length, sequence_length,
          attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1, input, output, causal);
    } else if (total_sequence_length <= 512) {
      const int blockSize = 512;
      SoftmaxKernelSmall<T, blockSize, HAS_BIAS><<<grid, blockSize, 0, stream>>>(
          total_sequence_length, sequence_length,
          attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1, input, output, causal);
    } else if (total_sequence_length <= 1024) {
      const int blockSize = 1024;
      SoftmaxKernelSmall<T, blockSize, HAS_BIAS><<<grid, blockSize, 0, stream>>>(
          total_sequence_length, sequence_length,
          attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1, input, output, causal);
    } else if (!causal) {
      const int blockSize = 1024;
      SoftmaxKernel<T, blockSize, HAS_BIAS><<<grid, blockSize, 0, stream>>>(
          total_sequence_length, sequence_length,
          attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1, input, output);
    } else {
      const int blockSize = 256;
      const int sh_bytes = sizeof(float) * total_sequence_length;
      SoftmaxLargeKernel<T, blockSize, HAS_BIAS><<<grid, blockSize, sh_bytes, stream>>>(
          total_sequence_length, sequence_length, total_sequence_length, 0, attn_bias,
          broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1,
          input, output, true);
    }
  });
  return CUDA_CALL(cudaGetLastError());
}

template <typename T, unsigned TPB, bool HAS_BIAS>
__global__ void MaskedSoftmaxKernelSmall(const int total_sequence_length,
                                         const int sequence_length,
                                         const int* mask_end,
                                         const int* mask_start,
                                         const T* attn_bias,
                                         const bool broadcast_attn_bias_dim_0,
                                         const bool broadcast_attn_bias_dim_1,
                                         const T* input,
                                         T* output,
                                         bool causal) {
  __shared__ int start_position;
  __shared__ int end_position;

  if (threadIdx.x == 0) {
    const int batch = blockIdx.y;
    start_position = mask_start != nullptr ? max(0, mask_start[batch]) : 0;
    end_position = min(total_sequence_length, mask_end[batch]);

    // Attend to no word has same effect as attend to all words. This is added to get parity with CPU result.
    if (start_position >= end_position) {
      start_position = 0;
      end_position = total_sequence_length;
    }
  }
  __syncthreads();

  SoftmaxSmall<T, TPB, HAS_BIAS>(total_sequence_length, sequence_length, end_position, start_position,
                                 attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1, input, output, causal);
}

template <typename T, unsigned TPB, bool HAS_BIAS>
__device__ inline void SoftmaxSmallPacked(const int total_sequence_length,
                                          const int sequence_length,
                                          const int end,
                                          const T* attn_bias,
                                          const bool broadcast_attn_bias_dim_0,
                                          const bool broadcast_attn_bias_dim_1,
                                          const T* input,
                                          T* output) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  DECLARE_SOFTMAX_VARS();

  int64_t index = offset + threadIdx.x;

  bool is_valid = threadIdx.x < end;

  float input_data = HAS_BIAS ? float(input[index]) + float(attn_bias[bias_offset + threadIdx.x]) : float(input[index]);

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

  // threadIdx.x might be larger than total_sequence_length due to alignment to 32x.
  if (threadIdx.x < sequence_length) {
    output[index] = T(thread_data_exp * sum_reverse_block);
  }
}

template <typename T, unsigned TPB, bool HAS_BIAS>
__global__ void SoftmaxKernelSmallWithCumSeqLen(const T* input,
                                                const T* attn_bias,
                                                const bool broadcast_attn_bias_dim_0,
                                                const bool broadcast_attn_bias_dim_1,
                                                const int* cum_seq_length,
                                                const int total_sequence_length,
                                                const int sequence_length,
                                                T* output) {
  __shared__ int end_position;

  if (threadIdx.x == 0) {
    const int batch = blockIdx.y;
    end_position = cum_seq_length[batch + 1] - cum_seq_length[batch];
  }
  __syncthreads();

  SoftmaxSmallPacked<T, TPB, HAS_BIAS>(total_sequence_length, sequence_length, end_position,
                                       attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1, input, output);
}

template <typename T, unsigned TPB, bool HAS_BIAS>
__global__ void SoftmaxKernelWithCumSeqLen(const T* input,
                                           const T* attn_bias,
                                           const bool broadcast_attn_bias_dim_0,
                                           const bool broadcast_attn_bias_dim_1,
                                           const int* cum_seq_length,
                                           const int total_sequence_length,
                                           const int sequence_length,
                                           T* output) {
  __shared__ int end_position;

  if (threadIdx.x == 0) {
    const int batch = blockIdx.y;
    end_position = cum_seq_length[batch + 1] - cum_seq_length[batch];
  }
  __syncthreads();

  constexpr int start_position = 0;
  Softmax<T, TPB, HAS_BIAS>(total_sequence_length, sequence_length, end_position, start_position,
                            attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1, input, output);
}

template <typename T, unsigned TPB, bool HAS_BIAS>
__global__ void MaskedSoftmaxKernel(const int total_sequence_length,
                                    const int sequence_length,
                                    const int* mask_end,
                                    const int* mask_start,
                                    const T* attn_bias,
                                    const bool broadcast_attn_bias_dim_0,
                                    const bool broadcast_attn_bias_dim_1,
                                    const T* input, T* output) {
  __shared__ int start_position;
  __shared__ int end_position;

  if (threadIdx.x == 0) {
    const int batch = blockIdx.y;
    start_position = mask_start != nullptr ? max(0, mask_start[batch]) : 0;
    end_position = min(total_sequence_length, mask_end[batch]);

    // Attend to no word has same effect as attend to all words. This is added to get parity with CPU result.
    if (start_position >= end_position) {
      start_position = 0;
      end_position = total_sequence_length;
    }
  }
  __syncthreads();

  Softmax<T, TPB, HAS_BIAS>(total_sequence_length, sequence_length, end_position, start_position,
                            attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1, input, output);
}

template <typename T, unsigned TPB, bool HAS_BIAS>
__global__ void SoftmaxWithRawMaskSmallKernel(const int total_sequence_length,
                                              const int sequence_length,
                                              const int* attention_mask,
                                              const bool* key_padding_mask,
                                              const T* attn_bias,
                                              const bool broadcast_attn_bias_dim_0,
                                              const bool broadcast_attn_bias_dim_1,
                                              const T* input,
                                              T* output,
                                              const bool causal,
                                              const float rsqrt_head_size,
                                              const int mask_dimension,
                                              const int max_sequence_length,
                                              const bool skip_softmax,
                                              const float mask_filter_value) {
  SoftmaxWithRawMaskSmall<T, TPB, HAS_BIAS>(
      total_sequence_length, sequence_length, attention_mask, key_padding_mask,
      attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1, input, output,
      causal, rsqrt_head_size, mask_dimension, max_sequence_length,
      skip_softmax, mask_filter_value);
}

template <typename T>
Status ComputeSoftmaxWithCumSeqLength(
    const T* input,
    const T* attn_bias,
    const bool broadcast_attn_bias_dim_0,
    const bool broadcast_attn_bias_dim_1,
    const int32_t* cum_seq_length,
    const int batch_size,
    const int sequence_length,
    const int total_sequence_length,
    const int num_heads,
    T* output, cudaStream_t stream) {
  DISPATCH_BIAS(attn_bias, HAS_BIAS, [&] {
    if (sequence_length <= 32) {
      const int blockSize = 32;
      SoftmaxKernelSmallWithCumSeqLen<T, blockSize, HAS_BIAS>
          <<<grid, blockSize, 0, stream>>>(input, attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1,
                                           cum_seq_length, total_sequence_length, sequence_length, output);
    } else if (sequence_length <= 64) {
      const int blockSize = 64;
      SoftmaxKernelSmallWithCumSeqLen<T, blockSize, HAS_BIAS>
          <<<grid, blockSize, 0, stream>>>(input, attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1,
                                           cum_seq_length, total_sequence_length, sequence_length, output);
    } else if (sequence_length <= 128) {
      const int blockSize = 128;
      SoftmaxKernelSmallWithCumSeqLen<T, blockSize, HAS_BIAS>
          <<<grid, blockSize, 0, stream>>>(input, attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1,
                                           cum_seq_length, total_sequence_length, sequence_length, output);
    } else if (sequence_length <= 256) {
      const int blockSize = 256;
      SoftmaxKernelSmallWithCumSeqLen<T, blockSize, HAS_BIAS>
          <<<grid, blockSize, 0, stream>>>(input, attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1,
                                           cum_seq_length, total_sequence_length, sequence_length, output);
    } else if (sequence_length <= 512) {
      const int blockSize = 512;
      SoftmaxKernelSmallWithCumSeqLen<T, blockSize, HAS_BIAS>
          <<<grid, blockSize, 0, stream>>>(input, attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1,
                                           cum_seq_length, total_sequence_length, sequence_length, output);
    } else if (sequence_length <= 1024) {
      const int blockSize = 1024;
      SoftmaxKernelSmallWithCumSeqLen<T, blockSize, HAS_BIAS>
          <<<grid, blockSize, 0, stream>>>(input, attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1,
                                           cum_seq_length, total_sequence_length, sequence_length, output);
    } else {
      const int blockSize = 1024;
      SoftmaxKernelWithCumSeqLen<T, blockSize, HAS_BIAS>
          <<<grid, 1024, 0, stream>>>(input, attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1,
                                      cum_seq_length, total_sequence_length, sequence_length, output);
    }
  });

  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status ComputeSoftmaxWithMask1D(cudaStream_t stream,
                                const int total_sequence_length,
                                const int sequence_length,
                                const int batch_size,
                                const int num_heads,
                                const int* mask_index,
                                const int* mask_start,
                                const T* attn_bias,
                                const bool broadcast_attn_bias_dim_0,
                                const bool broadcast_attn_bias_dim_1,
                                const T* input,
                                T* output,
                                const bool causal) {
  DISPATCH_BIAS(attn_bias, HAS_BIAS, [&] {
    if (total_sequence_length <= 32) {
      const int blockSize = 32;
      MaskedSoftmaxKernelSmall<T, blockSize, HAS_BIAS>
          <<<grid, blockSize, 0, stream>>>(total_sequence_length, sequence_length, mask_index, mask_start,
                                           attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1,
                                           input, output, causal);
    } else if (total_sequence_length <= 64) {
      const int blockSize = 64;
      MaskedSoftmaxKernelSmall<T, blockSize, HAS_BIAS>
          <<<grid, blockSize, 0, stream>>>(total_sequence_length, sequence_length, mask_index, mask_start,
                                           attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1,
                                           input, output, causal);
    } else if (total_sequence_length <= 128) {
      const int blockSize = 128;
      MaskedSoftmaxKernelSmall<T, blockSize, HAS_BIAS>
          <<<grid, blockSize, 0, stream>>>(total_sequence_length, sequence_length, mask_index, mask_start,
                                           attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1,
                                           input, output, causal);
    } else if (total_sequence_length <= 256) {
      const int blockSize = 256;
      MaskedSoftmaxKernelSmall<T, blockSize, HAS_BIAS>
          <<<grid, blockSize, 0, stream>>>(total_sequence_length, sequence_length, mask_index, mask_start,
                                           attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1,
                                           input, output, causal);
    } else if (total_sequence_length <= 512) {
      const int blockSize = 512;
      MaskedSoftmaxKernelSmall<T, blockSize, HAS_BIAS>
          <<<grid, blockSize, 0, stream>>>(total_sequence_length, sequence_length, mask_index, mask_start,
                                           attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1,
                                           input, output, causal);
    } else if (total_sequence_length <= 1024) {
      const int blockSize = 1024;
      MaskedSoftmaxKernelSmall<T, blockSize, HAS_BIAS>
          <<<grid, blockSize, 0, stream>>>(total_sequence_length, sequence_length, mask_index, mask_start,
                                           attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1,
                                           input, output, causal);
    } else if (!causal) {
      const int blockSize = 1024;
      MaskedSoftmaxKernel<T, blockSize, HAS_BIAS>
          <<<grid, blockSize, 0, stream>>>(total_sequence_length, sequence_length, mask_index, mask_start,
                                           attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1,
                                           input, output);
    }
  });

  if (total_sequence_length > 1024 && causal) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ComputeSoftmaxWithMask1D does not support causal with total sequence length > 1024.");
  }

  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status ComputeSoftmaxWithRawMask(Stream* ort_stream,
                                 const int total_sequence_length,
                                 const int sequence_length,
                                 const int batch_size,
                                 const int num_heads,
                                 const int* attention_mask,
                                 const bool* key_padding_mask,
                                 const T* attn_bias,
                                 const bool broadcast_attn_bias_dim_0,
                                 const bool broadcast_attn_bias_dim_1,
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
  T* out = use_persistent_softmax ? persistent_softmax_workspace : output;

  DISPATCH_BIAS(attn_bias, HAS_BIAS, [&] {
    if (total_sequence_length <= 32) {
      const int blockSize = 32;
      SoftmaxWithRawMaskSmallKernel<T, blockSize, HAS_BIAS>
          <<<grid, blockSize, 0, stream>>>(total_sequence_length, sequence_length, attention_mask, key_padding_mask,
                                           attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1, input,
                                           out, causal, rsqrt_head_size, mask_dimension, max_sequence_length,
                                           use_persistent_softmax, mask_filter_value);
    } else if (total_sequence_length <= 64) {
      const int blockSize = 64;
      SoftmaxWithRawMaskSmallKernel<T, blockSize, HAS_BIAS>
          <<<grid, blockSize, 0, stream>>>(total_sequence_length, sequence_length, attention_mask, key_padding_mask,
                                           attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1, input,
                                           out, causal, rsqrt_head_size, mask_dimension, max_sequence_length,
                                           use_persistent_softmax, mask_filter_value);
    } else if (total_sequence_length <= 128) {
      const int blockSize = 128;
      SoftmaxWithRawMaskSmallKernel<T, blockSize, HAS_BIAS>
          <<<grid, blockSize, 0, stream>>>(total_sequence_length, sequence_length, attention_mask, key_padding_mask,
                                           attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1, input,
                                           out, causal, rsqrt_head_size, mask_dimension, max_sequence_length,
                                           use_persistent_softmax, mask_filter_value);
    } else if (total_sequence_length <= 256) {
      const int blockSize = 256;
      SoftmaxWithRawMaskSmallKernel<T, blockSize, HAS_BIAS>
          <<<grid, blockSize, 0, stream>>>(total_sequence_length, sequence_length, attention_mask, key_padding_mask,
                                           attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1, input,
                                           out, causal, rsqrt_head_size, mask_dimension, max_sequence_length,
                                           use_persistent_softmax, mask_filter_value);
    } else if (total_sequence_length <= 512) {
      const int blockSize = 512;
      SoftmaxWithRawMaskSmallKernel<T, blockSize, HAS_BIAS>
          <<<grid, blockSize, 0, stream>>>(total_sequence_length, sequence_length, attention_mask, key_padding_mask,
                                           attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1, input,
                                           out, causal, rsqrt_head_size, mask_dimension, max_sequence_length,
                                           use_persistent_softmax, mask_filter_value);
    } else if (total_sequence_length <= 1024) {
      const int blockSize = 1024;
      SoftmaxWithRawMaskSmallKernel<T, blockSize, HAS_BIAS>
          <<<grid, blockSize, 0, stream>>>(total_sequence_length, sequence_length, attention_mask, key_padding_mask,
                                           attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1, input,
                                           out, causal, rsqrt_head_size, mask_dimension, max_sequence_length,
                                           use_persistent_softmax, mask_filter_value);
    } else {
      const int blockSize = 256;
      const int sh_bytes = sizeof(float) * total_sequence_length;
      SoftmaxWithRawMaskLargeKernel<T, blockSize, HAS_BIAS>
          <<<grid, blockSize, sh_bytes, stream>>>(
              total_sequence_length, sequence_length, attention_mask, key_padding_mask,
              attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1, input,
              out, causal, rsqrt_head_size, mask_dimension, max_sequence_length,
              use_persistent_softmax, mask_filter_value);
    }
  });

  if (use_persistent_softmax) {
    return onnxruntime::cuda::dispatch_warpwise_softmax_forward<T, T, float, false>(
        ort_stream,
        output,
        persistent_softmax_workspace,
        total_sequence_length,
        total_sequence_length,
        batch_size * num_heads * sequence_length);
  }

  return CUDA_CALL(cudaGetLastError());
}

// Template Instantiation
template Status ComputeSoftmax<float>(
    cudaStream_t stream, const int total_sequence_length, const int sequence_length,
    const int batch_size, const int num_heads, const float* attn_bias,
    const bool broadcast_attn_bias_dim_0, const bool broadcast_attn_bias_dim_1,
    float* input, float* output, bool causal);

template Status ComputeSoftmax<half>(
    cudaStream_t stream, const int total_sequence_length, const int sequence_length,
    const int batch_size, const int num_heads, const half* attn_bias,
    const bool broadcast_attn_bias_dim_0, const bool broadcast_attn_bias_dim_1,
    half* input, half* output, bool causal);

template Status ComputeSoftmaxWithCumSeqLength<float>(
    const float* input,
    const float* attn_bias,
    const bool broadcast_attn_bias_dim_0,
    const bool broadcast_attn_bias_dim_1,
    const int32_t* cum_seq_length,
    const int batch_size,
    const int sequence_length,
    const int total_sequence_length,
    const int num_heads,
    float* output, cudaStream_t stream);

template Status ComputeSoftmaxWithCumSeqLength<half>(
    const half* input,
    const half* attn_bias,
    const bool broadcast_attn_bias_dim_0,
    const bool broadcast_attn_bias_dim_1,
    const int32_t* cum_seq_length,
    const int batch_size,
    const int sequence_length,
    const int total_sequence_length,
    const int num_heads,
    half* output, cudaStream_t stream);

template Status ComputeSoftmaxWithMask1D<float>(cudaStream_t stream,
                                                const int total_sequence_length,
                                                const int sequence_length,
                                                const int batch_size,
                                                const int num_heads,
                                                const int* mask_index,
                                                const int* mask_start,
                                                const float* attn_bias,
                                                const bool broadcast_attn_bias_dim_0,
                                                const bool broadcast_attn_bias_dim_1,
                                                const float* input,
                                                float* output,
                                                const bool causal);

template Status ComputeSoftmaxWithMask1D<half>(cudaStream_t stream,
                                               const int total_sequence_length,
                                               const int sequence_length,
                                               const int batch_size,
                                               const int num_heads,
                                               const int* mask_index,
                                               const int* mask_start,
                                               const half* attn_bias,
                                               const bool broadcast_attn_bias_dim_0,
                                               const bool broadcast_attn_bias_dim_1,
                                               const half* input,
                                               half* output,
                                               const bool causal);

template Status ComputeSoftmaxWithRawMask<float>(Stream* ort_stream,
                                                 const int total_sequence_length,
                                                 const int sequence_length,
                                                 const int batch_size,
                                                 const int num_heads,
                                                 const int* attention_mask,
                                                 const bool* key_padding_mask,
                                                 const float* attn_bias,
                                                 const bool broadcast_attn_bias_dim_0,
                                                 const bool broadcast_attn_bias_dim_1,
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
                                                const int total_sequence_length,
                                                const int sequence_length,
                                                const int batch_size,
                                                const int num_heads,
                                                const int* attention_mask,
                                                const bool* key_padding_mask,
                                                const half* attn_bias,
                                                const bool broadcast_attn_bias_dim_0,
                                                const bool broadcast_attn_bias_dim_1,
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
