/*
 The implementation of this file is based on embLayerNorm plugin in TensorRT demo:
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

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "layer_norm.cuh"
#include "embed_layer_norm_impl.h"
#include <cuda_fp16.h>

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <unsigned TPB>
__global__ void MaskIndexKernelSmall(int sequence_length, const int* mask, int* mask_index) {
  using BlockReduce = cub::BlockReduce<int, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // blockIdx.x is b
  const int offset = blockIdx.x * sequence_length;  // batch strides of sequence_length

  cub::Min min;
  int thread_data(sequence_length);

  const int idx = offset + threadIdx.x;
  if (threadIdx.x < sequence_length) {
    const int val = mask[idx];
    if (val == 0)  // masked position: report thread idx
    {
      thread_data = threadIdx.x;
    }
  }

  const auto min_index = BlockReduce(temp_storage).Reduce(thread_data, min);

  if (threadIdx.x == 0) {
    mask_index[blockIdx.x] = min_index;
  }
}

template <unsigned TPB>
__global__ void MaskIndexKernel(int sequence_length, const int* mask, int* mask_index) {
  using BlockReduce = cub::BlockReduce<int, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // blockIdx.x is b
  const int offset = blockIdx.x * sequence_length;  // batch strides of sequence_length

  cub::Min min;
  int thread_data(sequence_length);

  for (int i = threadIdx.x; i < sequence_length; i += TPB) {
    const int idx = offset + i;
    const int val = mask[idx];
    if (val == 0)  // masked position: report thread idx
    {
      thread_data = min(thread_data, i);
    }
  }

  const auto min_index = BlockReduce(temp_storage).Reduce(thread_data, min);

  if (threadIdx.x == 0) {
    mask_index[blockIdx.x] = min_index;
  }
}

inline Status ComputeMaskIndex(cudaStream_t stream,
                             const int sequence_length,
                             const int batch_size,
                             const int* mask,
                             int* mask_index) {
  // Mask idx is of length batch_size and assumes the valid region is contiguous starting
  // from the beginning of the sequence

  // Assume n = batch_size x sequence_length
  if (sequence_length <= 32) {
    MaskIndexKernelSmall<32><<<batch_size, 32, 0, stream>>>(sequence_length, mask, mask_index);
  } else if (sequence_length <= 128) {
    MaskIndexKernelSmall<128><<<batch_size, 128, 0, stream>>>(sequence_length, mask, mask_index);
  } else if (sequence_length == 384) {
    MaskIndexKernelSmall<384><<<batch_size, 384, 0, stream>>>(sequence_length, mask, mask_index);
  } else {
    MaskIndexKernel<256><<<batch_size, 256, 0, stream>>>(sequence_length, mask, mask_index);
  }

  return CUDA_CALL(cudaGetLastError());
}

template <typename T, unsigned TPB>
__global__ void EmbedLayerNormKernel(
    int hidden_size, const int* input_ids, const int* segment_ids, const T* beta, const T* gamma,
    const T* word_embedding, const T* position_embedding, const T* segment_embedding,
    const T epsilon, T* output, T* embedding_sum, const int* position_ids, const bool broadcast_position_ids) {
  KeyValuePairSum pair_sum;
  // 1. lookup word and segment of the block
  // blockIdx.x = position in the sequence
  // blockIdx.y = batch
  // gridDim.x = sequence_length
  // gridDim.y = batch_size
  __shared__ int word_id;
  __shared__ int segment_id;
  __shared__ int position_id;

  const T rld = T(1.f / hidden_size);
  const int sequence_position = blockIdx.y * gridDim.x + blockIdx.x;
  if (threadIdx.x == 0) {
    word_id = input_ids[sequence_position];
    if (nullptr == segment_ids) {
      segment_id = 0;
    } else {
      segment_id = segment_ids[sequence_position];
    }
    if (nullptr == position_ids) {
      position_id = blockIdx.x;
    } else if (broadcast_position_ids){
      position_id = position_ids[sequence_position % gridDim.x];
    } else {
      position_id = position_ids[sequence_position];
    }
  }
  __syncthreads();

  // 2. load pos/segment/word embeddings and add them toghether
  // offset into embeddings is given by word_id * hidden_size
  const int position_offset = position_id * hidden_size;

  const int word_offset = word_id * hidden_size;
  const int segment_offset = segment_id * hidden_size;
  // the output offset is given by b * (sequence_length * hidden_size) + s * hidden_size
  const int output_offset = sequence_position * hidden_size;

  cub::KeyValuePair<T, T> thread_data(0, 0);

  for (int it = threadIdx.x; it < hidden_size; it += TPB) {
    const T w(word_embedding[word_offset + it]);
    T t(0);
    if (nullptr != segment_embedding)
      t = segment_embedding[segment_offset + it];
    const T p(position_embedding[position_offset + it]);
    const T val = w + t + p;

    output[output_offset + it] = val;
    if (embedding_sum != nullptr) {
      embedding_sum[output_offset + it] = val;
    }

    const T rldval = rld * val;
    thread_data = pair_sum(thread_data, cub::KeyValuePair<T, T>(rldval, rldval * val));
  }

  // 3. layer norm on the sum
  LayerNorm<T, TPB>(thread_data, hidden_size, output_offset, beta, gamma, epsilon, output);
}

template <typename T>
Status EmbedSkipLayerNorm(
    cudaStream_t stream, int hidden_size, int batch_size, int sequence_length,
    const int* input_ids, const int* segment_ids, const T* beta, const T* gamma,
    const T* word_embedding, const T* position_embedding, const T* segment_embedding,
    const T epsilon, T* output, T* embedding_sum, const int* position_ids,
    const bool broadcast_position_ids) {
  constexpr int tpb = 256;
  const dim3 grid(sequence_length, batch_size, 1);
  const dim3 block(tpb, 1, 1);

  EmbedLayerNormKernel<T, tpb>
      <<<grid, block, 0, stream>>>(hidden_size, input_ids, segment_ids, beta, gamma,
                                   word_embedding, position_embedding, segment_embedding,
                                   epsilon, output, embedding_sum, position_ids, broadcast_position_ids);

  return CUDA_CALL(cudaGetLastError());
}

Status LaunchEmbedLayerNormKernel(
    cudaStream_t stream,
    void* output,
    void* mask_index,
    const int* input_ids,
    const int* segment_ids,
    const int* input_mask,
    const void* gamma,
    const void* beta,
    const void* word_embedding,
    const void* position_embedding,
    const void* segment_embedding,
    float epsilon,
    const int hidden_size,
    int batch_size,
    int sequence_length,
    const size_t element_size,
    void* embedding_sum,
    const int* position_ids,
    const bool broadcast_position_ids) {

  if (mask_index != nullptr) {
    if (nullptr == input_mask) {
      CUDA_RETURN_IF_ERROR(cudaMemsetAsync(mask_index, 0, sizeof(int) * batch_size, stream));
    } else {
      ORT_RETURN_IF_ERROR(
        ComputeMaskIndex(stream, sequence_length, batch_size, input_mask, static_cast<int*>(mask_index)));
    }
  }

  if (element_size == 2) {
    return EmbedSkipLayerNorm<half>(
        stream, hidden_size, batch_size, sequence_length, input_ids, segment_ids,
        reinterpret_cast<const half*>(beta), reinterpret_cast<const half*>(gamma),
        reinterpret_cast<const half*>(word_embedding), reinterpret_cast<const half*>(position_embedding),
        reinterpret_cast<const half*>(segment_embedding), __float2half_rn(epsilon),
        reinterpret_cast<half*>(output), reinterpret_cast<half*>(embedding_sum), position_ids,
        broadcast_position_ids);
  } else {
    return EmbedSkipLayerNorm<float>(
        stream, hidden_size, batch_size, sequence_length, input_ids, segment_ids,
        reinterpret_cast<const float*>(beta), reinterpret_cast<const float*>(gamma),
        reinterpret_cast<const float*>(word_embedding), reinterpret_cast<const float*>(position_embedding),
        reinterpret_cast<const float*>(segment_embedding), epsilon,
        reinterpret_cast<float*>(output), reinterpret_cast<float*>(embedding_sum), position_ids,
        broadcast_position_ids);
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
