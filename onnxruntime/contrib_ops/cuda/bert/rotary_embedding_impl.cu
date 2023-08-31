// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/rotary_embedding_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__global__ void RotaryEmbeddingKernel(
		const T* input,
		const int64_t* pos,
		const T* cos_ptr,
		const T* sin_ptr,
		int64_t batch_size,
		int64_t num_heads,
		int64_t seqlen,
		int64_t head_dim,
                int64_t seqlen_with_past,
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
		Stream* stream,
		const T* input,
		int64_t batch_size,
		int64_t num_heads,
		int64_t seqlen,
		int64_t head_dim,
		int64_t seqlen_with_past,
                const int64_t* pos,
		const T* cos_buffer,
		const T* sin_buffer,
		T* output) {
  // TODO: check head_dim should less than kMaxThreadsPerBlock
  const int blockSize = head_dim;
  const dim3 gridSize(seqlen, num_heads, batch_size);
  cudaStream_t s = static_cast<cudaStream_t>(stream->GetHandle());
  RotaryEmbeddingKernel<T><<<gridSize, blockSize, 0, s>>>(
		  input,
		  pos,
		  cos_buffer,
		  sin_buffer,
		  batch_size,
		  num_heads,
		  seqlen,
		  head_dim,
		  seqlen_with_past,
		  output
		  );

  return CUDA_CALL(cudaGetLastError());
}

// instantiation
template
Status LaunchRotaryEmbeddingKernel<float>(
		Stream* stream,
		const float* input,
		int64_t batch_size,
		int64_t num_heads,
		int64_t seqlen,
		int64_t head_dim,
		int64_t seqlen_with_past,
                const int64_t* pos,
		const float* cos_buffer,
		const float* sin_buffer,
		float* output);

template
Status LaunchRotaryEmbeddingKernel<half>(
		Stream* stream,
		const half* input,
		int64_t batch_size,
		int64_t num_heads,
		int64_t seqlen,
		int64_t head_dim,
		int64_t seqlen_with_past,
                const int64_t* pos,
		const half* cos_buffer,
		const half* sin_buffer,
	        half* output);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
