// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// TrtSequenceOffset kernels are modified from FasterTransformer
/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/bert/bert_padding.h"
#include <cub/cub.cuh>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int32_t kMAX_THREADS_PER_BLOCK = 256;

// -----------------------------------
// Get indices of non-padding tokens and padding tokens. Here we assume that padding is on the right side of sequence.
// sequence_token_count is number of non-padding tokens per sequence, and it has shape [batch_size].
// For example, we have 3 sequences with 1, 2, 4 non-padding tokens and positions like the following (* means padding):
//   Sequence_0:   0,  1*, 2*,  3*
//   Sequence_1:   4,  5,  6*,  7*
//   Sequence_2:   8,  9,  10,  11
// token_offset: 0, 4, 5, 8, 9, 10, 11,  1*, 2*, 3*, 6*, 7*
// token_count_buffer has two numbers for non-padding tokens:
//   total_token_count: 1 + 2 + 4 = 7
//   max_token_count: 4
// cumulated_token_count: 0, 1, 1+2, 1+2+4
__global__ void getTokenOffset(int* token_count_buffer,
                               int* token_offset,
                               int* cumulated_token_count,
                               const int* sequence_token_count,
                               const int batch_size,
                               const int sequence_length) {
  // Find offset of non-padding tokens, and max sequence length among all batches
  // TODO(tianleiwu): Use cub::DevicePartition::Flagged like BuildGlobalIndex in longformer_global_impl.cu
  //                  to build token_offset when sequence length is large.
  int total_tokens = 0;
  int max_tokens = 0;
  int index = 0;
  cumulated_token_count[0] = 0;
  for (int i = 0; i < batch_size; i++) {
    const int count = sequence_token_count[i];
    if (count > max_tokens) {
      max_tokens = count;
    }
    cumulated_token_count[i + 1] = cumulated_token_count[i] + count;

    for (int j = 0; j < count; j++) {
      token_offset[index] = i * sequence_length + j;
      index++;
    }
    total_tokens += count;
  }

  //  Offset of paddings
  for (int i = 0; i < batch_size; i++) {
    const int count = sequence_token_count[i];
    for (int j = 0; j < sequence_length - count; j++) {
      token_offset[index] = i * sequence_length + count + j;
      index++;
    }
  }

  token_count_buffer[0] = total_tokens;
  token_count_buffer[1] = max_tokens;
}

void LaunchGetTokenOffset(int* token_count_buffer,
                          int* token_offset,
                          int* cumulated_token_count,
                          const int* sequence_token_count,
                          const int batch_size,
                          const int sequence_length,
                          cudaStream_t stream) {
  getTokenOffset<<<1, 1, 0, stream>>>(
      token_count_buffer, token_offset, cumulated_token_count, sequence_token_count, batch_size, sequence_length);
}

// -----------------------------------
// Remove paddings
template <typename T>
__global__ void __launch_bounds__(kMAX_THREADS_PER_BLOCK)
    removePadding(T* target, const T* source, const int* token_offset, const int width) {
  const int tid = threadIdx.x;
  const int token_index = blockIdx.x;
  const int source_offset = token_offset[token_index];
  const int target_offset = token_index;

  for (int i = tid; i < width; i += blockDim.x) {
    target[target_offset * width + i] = source[source_offset * width + i];
  }
}

template <>
void LaunchRemovePadding(
    half* output, const half* input, const int* token_offset, const int token_count, const int hidden_size,
    cudaStream_t stream) {
  // input: [batch_size, sequence_length, hidden_size]
  // output: [token_count, hidden_size]

  // Make sure memory is aligned to 128 bit
  ORT_ENFORCE(!(reinterpret_cast<size_t>(input) & 0xF) && !(reinterpret_cast<size_t>(output) & 0xF), "alignment");

  if (hidden_size % 8 == 0) {
    const int width = hidden_size / 8;
    const int4* input2 = reinterpret_cast<const int4*>(input);
    int4* output2 = reinterpret_cast<int4*>(output);
    removePadding<int4><<<token_count, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
        output2, input2, token_offset, width);
  } else if (hidden_size % 4 == 0) {
    const int width = hidden_size / 4;
    const int64_t* input2 = reinterpret_cast<const int64_t*>(input);
    int64_t* output2 = reinterpret_cast<int64_t*>(output);
    removePadding<int64_t><<<token_count, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
        output2, input2, token_offset, width);
  } else if (hidden_size % 2 == 0) {
    const int width = hidden_size / 2;
    const int32_t* input2 = reinterpret_cast<const int32_t*>(input);
    int32_t* output2 = reinterpret_cast<int32_t*>(output);
    removePadding<int32_t><<<token_count, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
        output2, input2, token_offset, width);
  } else {
    const int width = hidden_size;
    const int16_t* input2 = reinterpret_cast<const int16_t*>(input);
    int16_t* output2 = reinterpret_cast<int16_t*>(output);
    removePadding<int16_t><<<token_count, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
        output2, input2, token_offset, width);
  }
}

template <>
void LaunchRemovePadding(
    float* output, const float* input, const int* token_offset, const int token_count, const int hidden_size,
    cudaStream_t stream) {
  ORT_ENFORCE(!(reinterpret_cast<size_t>(input) & 0xF) && !(reinterpret_cast<size_t>(output) & 0xF), "alignment");

  if (hidden_size % 4 == 0) {
    const int width = hidden_size / 4;
    const int4* input2 = reinterpret_cast<const int4*>(input);
    int4* output2 = reinterpret_cast<int4*>(output);
    removePadding<int4><<<token_count, kMAX_THREADS_PER_BLOCK, 0, stream>>>(output2, input2, token_offset, width);
  } else if (hidden_size % 2 == 0) {
    const int width = hidden_size / 2;
    const int64_t* input2 = reinterpret_cast<const int64_t*>(input);
    int64_t* output2 = reinterpret_cast<int64_t*>(output);
    removePadding<int64_t><<<token_count, kMAX_THREADS_PER_BLOCK, 0, stream>>>(output2, input2, token_offset, width);
  } else {
    const int width = hidden_size;
    const int32_t* input2 = reinterpret_cast<const int32_t*>(input);
    int32_t* output2 = reinterpret_cast<int32_t*>(output);
    removePadding<int32_t><<<token_count, kMAX_THREADS_PER_BLOCK, 0, stream>>>(output2, input2, token_offset, width);
  }
}

// -----------------------------------
// Recover padding.
template <typename T>
__global__ void __launch_bounds__(kMAX_THREADS_PER_BLOCK)
    restorePadding(T* target, const T* source, const int* token_offset, const int width, const int token_count) {
  const int tid = threadIdx.x;
  const int token_index = blockIdx.x;
  const int target_seq_id = token_offset[token_index];
  const int source_seq_id = token_index;
  constexpr T padding_zero = 0;

  if (token_index < token_count) {
    for (int i = tid; i < width; i += blockDim.x) {
      target[target_seq_id * width + i] = source[source_seq_id * width + i];
    }
  } else {
    // It is padding: fill with zeros
    for (int i = tid; i < width; i += blockDim.x) {
      target[target_seq_id * width + i] = padding_zero;
    }
  }
}

template <>
__global__ void __launch_bounds__(kMAX_THREADS_PER_BLOCK)
    restorePadding(int4* target, const int4* source, const int* token_offset, const int width, const int token_count) {
  const int tid = threadIdx.x;
  const int token_index = blockIdx.x;
  const int target_seq_id = token_offset[token_index];
  const int source_seq_id = token_index;
  int4 padding_zero{0, 0, 0, 0};

  if (token_index < token_count) {
    for (int i = tid; i < width; i += blockDim.x) {
      target[target_seq_id * width + i] = source[source_seq_id * width + i];
    }
  } else {
    // It is padding: fill with zeros
    for (int i = tid; i < width; i += blockDim.x) {
      target[target_seq_id * width + i] = padding_zero;
    }
  }
}

template <>
void LaunchRestorePadding(
    float* output, const float* input, const int* token_offset, const int token_count, const int hidden_size,
    const int batch_size, const int sequence_length,
    cudaStream_t stream) {
  ORT_ENFORCE(!(reinterpret_cast<size_t>(input) & 0xF) && !(reinterpret_cast<size_t>(output) & 0xF), "alignment");

  int grid_size = batch_size * sequence_length;
  if (hidden_size % 4 == 0) {
    const int width = hidden_size / 4;
    const int4* input2 = reinterpret_cast<const int4*>(input);
    int4* output2 = reinterpret_cast<int4*>(output);
    restorePadding<int4><<<grid_size, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
        output2, input2, token_offset, width, token_count);
  } else if (hidden_size % 2 == 0) {
    const int width = hidden_size / 2;
    const int64_t* input2 = reinterpret_cast<const int64_t*>(input);
    int64_t* output2 = reinterpret_cast<int64_t*>(output);
    restorePadding<int64_t><<<grid_size, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
        output2, input2, token_offset, width, token_count);
  } else {
    const int width = hidden_size;
    const int32_t* input2 = reinterpret_cast<const int32_t*>(input);
    int32_t* output2 = reinterpret_cast<int32_t*>(output);
    restorePadding<int32_t><<<grid_size, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
        output2, input2, token_offset, width, token_count);
  }
}

template <>
void LaunchRestorePadding(
    half* output, const half* input, const int* token_offset, const int token_count, const int hidden_size,
    const int batch_size, const int sequence_length,
    cudaStream_t stream) {
  // input: [token_count, hidden_size]
  // output: [batch_size, sequence_length, hidden_size]

  ORT_ENFORCE(!(reinterpret_cast<size_t>(input) & 0xF) && !(reinterpret_cast<size_t>(output) & 0xF), "alignment");

  int grid_size = batch_size * sequence_length;
  if (hidden_size % 8 == 0) {
    const int width = hidden_size / 8;
    const int4* input2 = reinterpret_cast<const int4*>(input);
    int4* output2 = reinterpret_cast<int4*>(output);
    restorePadding<int4><<<grid_size, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
        output2, input2, token_offset, width, token_count);
  } else if (hidden_size % 4 == 0) {
    const int width = hidden_size / 4;
    const int64_t* input2 = reinterpret_cast<const int64_t*>(input);
    int64_t* output2 = reinterpret_cast<int64_t*>(output);
    restorePadding<int64_t><<<grid_size, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
        output2, input2, token_offset, width, token_count);
  } else if (hidden_size % 2 == 0) {
    const int width = hidden_size / 2;
    const int32_t* input2 = reinterpret_cast<const int32_t*>(input);
    int32_t* output2 = reinterpret_cast<int32_t*>(output);
    restorePadding<int32_t><<<grid_size, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
        output2, input2, token_offset, width, token_count);
  } else {
    const int width = hidden_size;
    const int16_t* input2 = reinterpret_cast<const int16_t*>(input);
    int16_t* output2 = reinterpret_cast<int16_t*>(output);
    restorePadding<int16_t><<<grid_size, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
        output2, input2, token_offset, width, token_count);
  }
}

__global__ void __launch_bounds__(kMAX_THREADS_PER_BLOCK)
    getTrtSequenceOffset(int* trt_mha_padding_offset,
                         const int* sequence_token_count,
                         const int batch_size) {
  extern __shared__ int tmp_offset[];
  if (threadIdx.x == 0) {
    tmp_offset[0] = 0;
    for (int i = 0; i < batch_size; i++) {
      tmp_offset[i + 1] = tmp_offset[i] + sequence_token_count[i];
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < batch_size + 1; i += blockDim.x) {
    trt_mha_padding_offset[i] = tmp_offset[i];
  }
}

// Get sequence offset for TensorRT fused attention when there is no padding (or padding is removed)
void LaunchTrtSequenceOffset(int* trt_mha_padding_offset,
                             const int* sequence_token_count,
                             const int batch_size,
                             cudaStream_t stream) {
  getTrtSequenceOffset<<<1, kMAX_THREADS_PER_BLOCK, sizeof(int) * (batch_size + 1), stream>>>(
      trt_mha_padding_offset, sequence_token_count, batch_size);
}

__global__ void __launch_bounds__(kMAX_THREADS_PER_BLOCK)
    getTrtSequenceOffset(int* trt_mha_padding_offset,
                         const int* sequence_token_count,
                         const int batch_size,
                         const int sequence_length) {
  extern __shared__ int tmp_offset[];
  if (threadIdx.x == 0) {
    tmp_offset[0] = 0;
    // B for fused attention is 2 * batch_size
    for (int i = 0; i < batch_size; i++) {
      tmp_offset[i * 2 + 1] = tmp_offset[i * 2] + sequence_token_count[i];
      tmp_offset[i * 2 + 2] = sequence_length * (i + 1);
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < 2 * batch_size + 1; i += blockDim.x) {
    trt_mha_padding_offset[i] = tmp_offset[i];
  }
}

// When there is no attention mask, the sequence offset is like
// 0, sequence_length, 2 * sequence_length, 3 * sequence_length, .... ,batch_size * sequence_length
__global__ void __launch_bounds__(kMAX_THREADS_PER_BLOCK)
    getTrtSequenceOffsetNoMask(int* trt_mha_padding_offset,
                               const int batch_size,
                               const int sequence_length) {
  extern __shared__ int tmp_offset[];
  if (threadIdx.x == 0) {
    tmp_offset[0] = 0;
    for (int i = 0; i < batch_size; i++) {
      tmp_offset[i + 1] = sequence_length * (i + 1);
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < batch_size + 1; i += blockDim.x) {
    trt_mha_padding_offset[i] = tmp_offset[i];
  }
}

// Get sequence offset for TensorRT fused attention when we keep the padding
void LaunchTrtSequenceOffset(int* trt_mha_padding_offset,
                             const int* sequence_token_count,
                             const int batch_size,
                             const int sequence_length,
                             cudaStream_t stream) {
  if (nullptr == sequence_token_count) {
    getTrtSequenceOffsetNoMask<<<1, kMAX_THREADS_PER_BLOCK, sizeof(int) * (batch_size + 1), stream>>>(
        trt_mha_padding_offset, batch_size, sequence_length);
  } else {
    getTrtSequenceOffset<<<1, kMAX_THREADS_PER_BLOCK, sizeof(int) * (2 * batch_size + 1), stream>>>(
        trt_mha_padding_offset, sequence_token_count, batch_size, sequence_length);
  }
}

__global__ void __launch_bounds__(kMAX_THREADS_PER_BLOCK)
    getTrtSequenceOffset2d(int* trt_mha_padding_offset,
                           const int* attention_masks,
                           const int batch_size,
                           const int sequence_length) {
    typedef cub::BlockReduce<int, kMAX_THREADS_PER_BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int batch_id = blockIdx.x;
    const int* batch_mask = attention_masks + (batch_id * sequence_length);
    const bool leftmost_non_zero = (batch_mask[0] != 0);
    int biggest_position = 0;

    for (int i = threadIdx.x; i < sequence_length; i += blockDim.x) {
      if (leftmost_non_zero == (batch_mask[i] != 0)) {
        biggest_position = i;
      } else {
        break;
      }
    }

    int last_leading_position = BlockReduce(temp_storage).Reduce(biggest_position, cub::Max(), blockDim.x);

    if (threadIdx.x == 0) {
      int batch_offset = batch_id * sequence_length;
      trt_mha_padding_offset[2 * batch_id] = batch_offset;
      trt_mha_padding_offset[2 * batch_id + 1] = batch_offset + last_leading_position + 1;
      if (batch_id == gridDim.x - 1) {
        trt_mha_padding_offset[2 * batch_id + 2] = batch_offset + sequence_length;
      }
    }
}

// only support simple left padding with mask 0s on leading left,
//           or simple right padding with mask 1s on leading left.
void LaunchTrtSequenceOffset2d(int* trt_mha_padding_offset,
                               const int* attention_masks,
                               const int batch_size,
                               const int sequence_length,
                               cudaStream_t stream) {
  getTrtSequenceOffset2d<<<batch_size, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
      trt_mha_padding_offset, attention_masks, batch_size, sequence_length);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
