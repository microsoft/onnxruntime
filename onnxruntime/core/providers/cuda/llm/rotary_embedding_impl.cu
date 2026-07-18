/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/

/*
Kernel implementation for rotary embeddings.
*/

#include "core/providers/cuda/llm/rotary_embedding_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include <cuda_fp16.h>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void RotaryEmbeddingBSNH(T* output,                    // BxSxNxH
                                    const T* input,               // BxSxNxH
                                    const T* cos_cache,           // BxSx(H/2) or Mx(H/2)
                                    const T* sin_cache,           // BxSx(H/2) or Mx(H/2)
                                    const int64_t* position_ids,  // (0) or BxS
                                    const int sequence_length, const int num_heads, const int head_size,
                                    const int rotary_embedding_dim, const int max_sequence_length,
                                    const int position_ids_format,
                                    const bool interleaved,
                                    int4 in_strides, int4 out_strides  // strides in bnsh coord, h is always contiguous
) {
  // B = batch size, S = sequence length, N = num heads, H = head size, M = max sequence length
  // Use .x in innermost loop to access global memory efficiently

  const int b = blockIdx.y;
  const int s = blockIdx.x;
  const int n = blockIdx.z;

  const int i = threadIdx.x;

  if (i >= head_size) {
    return;
  }

  const T* input_data = input + b * in_strides.x + s * in_strides.z + n * in_strides.y;
  T* output_data = output + b * out_strides.x + s * out_strides.z + n * out_strides.y;

  if (i >= rotary_embedding_dim) {
    output_data[i] = input_data[i];
    return;
  }

  // Cache is (M, H/2)
  const int half_rotary_embedding_dim = rotary_embedding_dim / 2;
  int cache_offset;

  // position_ids_format == 0 means position_ids is nullptr; cache is (B*S, H/2) and index is always valid.
  // position_ids_format == 1 means position_ids is a 2D array of size (batch_size, sequence_length)
  int b_s_index = b * sequence_length + s;
  if (position_ids_format != 0) {
    int64_t pos = position_ids[b_s_index];
#if !defined(NDEBUG)
    if (i == 0) {
      CUDA_KERNEL_ASSERT(pos >= 0 && pos < static_cast<int64_t>(max_sequence_length));
    }
#endif
    if (pos < 0 || pos >= static_cast<int64_t>(max_sequence_length)) {
      // OOB position id — can't propagate error from GPU, so pass through input unchanged.
      output_data[i] = input_data[i];
      return;
    }
    b_s_index = static_cast<int>(pos);
  }
  cache_offset = b_s_index * half_rotary_embedding_dim;
  const T* cos_data = cos_cache + cache_offset;
  const T* sin_data = sin_cache + cache_offset;

  int cache_idx = 0;
  T sign = 0;
  int j = 0;
  if (interleaved) {
    cache_idx = (i / 2) % half_rotary_embedding_dim;
    sign = (i % 2 == 0) ? -1 : 1;
    j = (i % 2 == 0) ? i + 1 : i - 1;  // i - sign
  } else {
    cache_idx = i % half_rotary_embedding_dim;
    sign = (i < half_rotary_embedding_dim) ? -1 : 1;
    j = (i + half_rotary_embedding_dim) % rotary_embedding_dim;
  }
  output_data[i] = input_data[i] * cos_data[cache_idx] + sign * input_data[j] * sin_data[cache_idx];
}

template <typename T>
Status LaunchRotaryEmbeddingKernel(cudaStream_t stream, T* output, const T* input, const int64_t* position_ids,
                                   const T* cos_cache, const T* sin_cache, const int batch_size,
                                   const int sequence_length, const int num_heads, const int head_size,
                                   const int rotary_embedding_dim, const int max_sequence_length,
                                   const int position_ids_format, const bool interleaved,
                                   const int max_threads_per_block, const bool is_input_bnsh_format) {
  int4 in_strides;
  int4 out_strides;
  if (is_input_bnsh_format) {
    // Semantic meaning of the strides:
    // int in_head_stride = sequence_length * head_size;
    // int out_head_stride = sequence_length * head_size;
    // in_strides = int4{num_heads * in_head_stride, in_head_stride, in_head_stride / sequence_length, 1};
    // out_strides = int4{num_heads * out_head_stride, out_head_stride, out_head_stride / sequence_length, 1};
    // Simplify to:
    in_strides = int4{num_heads * sequence_length * head_size, sequence_length * head_size, head_size, 1};
    out_strides = int4{num_heads * sequence_length * head_size, sequence_length * head_size, head_size, 1};
  } else {
    // input is in bshn format
    // int in_head_stride = head_size;
    // int out_head_stride = head_size;
    // Simplify to:
    in_strides = int4{num_heads * sequence_length * head_size, head_size, num_heads * head_size, 1};
    out_strides = int4{num_heads * sequence_length * head_size, head_size, num_heads * head_size, 1};
  }
  return LaunchRotaryEmbeddingKernel<T>(
      stream, output, input, position_ids,
      cos_cache, sin_cache, batch_size,
      sequence_length, num_heads, head_size,
      rotary_embedding_dim, max_sequence_length,
      position_ids_format, interleaved,
      max_threads_per_block,
      in_strides, out_strides);
}

template <typename T>
Status LaunchRotaryEmbeddingKernel(cudaStream_t stream, T* output, const T* input, const int64_t* position_ids,
                                   const T* cos_cache, const T* sin_cache, const int batch_size,
                                   const int sequence_length, const int num_heads, const int head_size,
                                   const int rotary_embedding_dim, const int max_sequence_length,
                                   const int position_ids_format, const bool interleaved,
                                   const int max_threads_per_block,
                                   int4 in_strides, int4 out_strides  // strides in bnsh coord
) {
  // Note: Requires head_size <= max_threads_per_block (1024). Each thread processes one element
  // of head_size, so the entire head must fit in a single thread block.
  ORT_ENFORCE(head_size <= max_threads_per_block, "Rotary embedding dim must be <= max_threads_per_block");
  // strides in canonical bnsh coord, h is always contiguous (dim_stride == 1)
  ORT_ENFORCE(in_strides.w == 1 && out_strides.w == 1, "head dim must be contiguous");

  int tpb = (head_size + 31) / 32 * 32;

  const dim3 block(tpb);
  const dim3 grid(sequence_length, batch_size, num_heads);

  RotaryEmbeddingBSNH<<<grid, block, 0, stream>>>(output, input, cos_cache, sin_cache, position_ids, sequence_length,
                                                  num_heads, head_size, rotary_embedding_dim, max_sequence_length,
                                                  position_ids_format, interleaved, in_strides, out_strides);
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchRotaryEmbeddingKernel<float>(cudaStream_t stream, float* output, const float* input,
                                                   const int64_t* position_ids, const float* cos_cache,
                                                   const float* sin_cache, const int batch_size,
                                                   const int sequence_length, const int num_heads, const int head_size,
                                                   const int rotary_embedding_dim, const int max_sequence_length,
                                                   const int position_ids_format, const bool interleaved,
                                                   const int max_threads_per_block, const bool is_input_bnsh_format);

template Status LaunchRotaryEmbeddingKernel<half>(cudaStream_t stream, half* output, const half* input,
                                                  const int64_t* position_ids, const half* cos_cache,
                                                  const half* sin_cache, const int batch_size,
                                                  const int sequence_length, const int num_heads, const int head_size,
                                                  const int rotary_embedding_dim, const int max_sequence_length,
                                                  const int position_ids_format, const bool interleaved,
                                                  const int max_threads_per_block, const bool is_input_bnsh_format);

// Native CUDA type instantiation: OrtToCudaType<BFloat16>::type = __nv_bfloat16.
// Used when rotary_embedding.cc dispatches via OrtToCudaType for native HW arithmetic on SM80+.
template Status LaunchRotaryEmbeddingKernel<__nv_bfloat16>(
    cudaStream_t stream, __nv_bfloat16* output, const __nv_bfloat16* input, const int64_t* position_ids,
    const __nv_bfloat16* cos_cache, const __nv_bfloat16* sin_cache, const int batch_size, const int sequence_length,
    const int num_heads, const int head_size, const int rotary_embedding_dim, const int max_sequence_length,
    const int position_ids_format, const bool interleaved, const int max_threads_per_block,
    const bool is_input_bnsh_format);

}  // namespace cuda
}  // namespace onnxruntime
