/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/

/*
Kernel implementation for rotary embeddings.
*/

#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/bert/rotary_embedding_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__global__ void RotaryEmbeddingBSNH(T* output,                   // BxSxNxH
                                    const T* input,              // BxSxNxH
                                    const T* cos_cache,          // Mx(H/2)
                                    const T* sin_cache,          // Mx(H/2)
                                    const int64_t* position_ids, // (1) or BxS
                                    const int sequence_length,
                                    const int num_heads,
                                    const int head_size,
                                    const int position_ids_format,
                                    const bool interleaved) {
  // B = batch size, S = sequence length, N = num heads, H = head size, M = max sequence length
  // Use .x in innermost loop to access global memory efficiently
  
  const int b = blockIdx.z;
  const int s = blockIdx.y;
  const int n = blockIdx.x;

  const int i = threadIdx.x;

  const int block_offset = b * sequence_length * num_heads + s * num_heads + n;
  const int data_offset = block_offset * head_size;

  const T* input_data = input + data_offset;
  T* output_data = output + data_offset;

  // Cache is (M, H/2)
  const int half_head_size = head_size / 2;
  const int position_id = (position_ids_format == 0) ? \
                          static_cast<int>(position_ids[0]) + s \
                          : static_cast<int>(position_ids[b * sequence_length + s]);
  const int cache_offset = position_id * half_head_size;
  const T* cos_data = cos_cache + cache_offset;
  const T* sin_data = sin_cache + cache_offset;

  int cache_idx = 0;
  T sign = 0;
  int j = 0;
  if (interleaved) {
    cache_idx = (i / 2) % half_head_size;
    sign = (i % 2 == 0) ? -1 : 1;
    j = (i % 2 == 0) ? i+1 : i-1;  // i - sign
  } else {
    cache_idx = i % half_head_size;
    sign = (i < half_head_size) ? -1 : 1;
    j = (i + half_head_size) % head_size;
  }
  output_data[i] = input_data[i] * cos_data[cache_idx] + sign * input_data[j] * sin_data[cache_idx];
}


template <typename T>
Status LaunchRotaryEmbeddingKernel(
    cudaStream_t stream,
    T* output,
    const T* input,
    const int64_t* position_ids,
    const T* cos_cache,
    const T* sin_cache,
    const int batch_size,
    const int sequence_length,
    const int num_heads,
    const int head_size,
    const int max_sequence_length,
    const int position_ids_format,
    const bool interleaved,
    const int max_threads_per_block) {

  constexpr int smem_size = 0;
  const dim3 grid(num_heads, sequence_length, batch_size);
  const dim3 block(head_size, 1, 1);

  // Note: Current implementation assumes head_size <= max_threads_per_block
  // because head_size is currently large for LLaMA-2. For smaller head_size
  // and num_heads values, we can create a block as `block(num_heads, head_size, 1)`
  // instead. This will require kernel changes to support.

  assert(head_size <= max_threads_per_block);
  RotaryEmbeddingBSNH<<<grid, block, smem_size, stream>>>(
    output, input, cos_cache, sin_cache, position_ids,
    sequence_length, num_heads, head_size, position_ids_format, interleaved
  );

  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchRotaryEmbeddingKernel<float>(
    cudaStream_t stream,
    float* output,
    const float* input,
    const int64_t* position_ids,
    const float* cos_cache,
    const float* sin_cache,
    const int batch_size,
    const int sequence_length,
    const int num_heads,
    const int head_size,
    const int max_sequence_length,
    const int position_ids_format,
    const bool interleaved,
    const int max_threads_per_block);

template Status LaunchRotaryEmbeddingKernel<half>(
    cudaStream_t stream,
    half* output,
    const half* input,
    const int64_t* position_ids,
    const half* cos_cache,
    const half* sin_cache,
    const int batch_size,
    const int sequence_length,
    const int num_heads,
    const int head_size,
    const int max_sequence_length,
    const int position_ids_format,
    const bool interleaved,
    const int max_threads_per_block);


}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
