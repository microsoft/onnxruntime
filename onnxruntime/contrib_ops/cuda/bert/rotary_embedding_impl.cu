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

// Kernel for LLaMA Microsoft model
template <typename T>
__global__ void RotaryEmbeddingBSNH(T* output,                  // BxSxNxH
                                    const T* input,             // BxSxNxH
                                    const T* cos_cache,         // Mx(H/2)
                                    const T* sin_cache,         // Mx(H/2)
                                    const int position_id,
                                    const int sequence_length,
                                    const int num_heads,
                                    const int head_size) {
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
  const int cache_offset = (position_id + s) * half_head_size;
  const T* cos_data = cos_cache + cache_offset;
  const T* sin_data = sin_cache + cache_offset;

  const int cache_idx = (i / 2) % half_head_size;
  const T sign = (i % 2 == 0) ? -1 : 1;
  const int j = (i % 2 == 0) ? i+1 : i-1;  // i - sign

  output_data[i] = input_data[i] * cos_data[cache_idx] + sign * input_data[j] * sin_data[cache_idx];
}

// Kernel for LLaMA Hugging Face model
template <typename T>
__global__ void RotaryEmbeddingBNSH(
		const T* input,
		const int64_t* pos,
		const T* cos_ptr,
		const T* sin_ptr,
		const int64_t num_heads,
		const int64_t seqlen,
		const int64_t head_dim,
    const int64_t seqlen_with_past,
		T* output) {
  // each block handle one head_dim of input
  // block size is same as head_dim, one thread handle one element.
  // 1. get cos and sin from buffer with index from pos
  // 2. in0 = input[i]
  //    in1 = -input[(i + half) % dim] if i < half else input[(i+half) % dim]
  // 3. output = in0 * cos + in1 * sin
  const int batch_offset = blockIdx.z;
  const int head_offset = blockIdx.y;
  const int seqlen_offset = blockIdx.x;
  const int i = threadIdx.x;

  const int block_offset = batch_offset * num_heads * seqlen + head_offset * seqlen + seqlen_offset;
  const auto* in_offset = input + head_dim * block_offset;
  auto* out_offset = output + head_dim * block_offset;

  int64_t pos_id = pos[batch_offset * seqlen + seqlen_offset];
  if (pos_id >= seqlen_with_past) {
    // TODO: may be need to assert pos_id < seqlen, this depends the input position_ids
    // safe guard: when pos is invalid, use seqlen_with_past - 1
    int64_t start_pos = seqlen_with_past - seqlen;
    pos_id = start_pos + seqlen_offset;
  }

  // cos and sin with shape[seqlen, head_dim]
  // cos = cos_ptr[pos_id][i]
  auto cos = cos_ptr[pos_id * head_dim + i];
  auto sin = sin_ptr[pos_id * head_dim + i];
  auto in0 = in_offset[i];
  const int half_dim = head_dim / 2;
  const int i1 = (i + half_dim) % head_dim;
  auto in1 = (i1 >= half_dim) ? -in_offset[i1] : in_offset[i1];

  out_offset[i] = in0 * cos + in1 * sin;
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
    const int model_format,
    const int max_threads_per_block) {

  const int smem_size = 0;

  if (model_format == 0) {
    const dim3 grid(num_heads, sequence_length, batch_size);
    const dim3 block(head_size, 1, 1);
    RotaryEmbeddingBSNH<<<grid, block, smem_size, stream>>>(
      output, input, cos_cache, sin_cache, *position_ids, sequence_length, num_heads, head_size
    );
    // if (head_size * num_heads <= max_threads_per_block) {
    //   const dim3 block(head_size, 1, 1);
    //   RotaryEmbeddingBSNH<<<grid, block, smem_size, stream>>>(
    //     output, input, cos_cache, sin_cache, *position_id, head_size
    //   );
    // } else {
    //   const dim3 block(max_threads_per_block / head_size, head_size, 1);
    //   RotaryEmbeddingBSNH<<<grid, block, smem_size, stream>>>(
    //     output, input, cos_cache, sin_cache, *position_id, head_size
    //   );
    // }
  } else if (model_format == 1) {
    const dim3 grid(sequence_length, num_heads, batch_size);
    const dim3 block(head_size, 1, 1);
    RotaryEmbeddingBNSH<<<grid, block, smem_size, stream>>>(
      input, position_ids, cos_cache, sin_cache,
      num_heads, sequence_length, head_size, sequence_length,
      output
    );
  }

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
    const int model_format,
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
    const int model_format,
    const int max_threads_per_block);


}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
