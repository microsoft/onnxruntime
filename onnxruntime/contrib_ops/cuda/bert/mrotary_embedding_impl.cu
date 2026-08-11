/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/

/*
Kernel implementation for fused Multimodal RoPE (M-RoPE) as used by the Qwen family of
vision-language models (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE, Qwen3.5, Qwen3.5-MoE).
*/

#include "contrib_ops/cuda/bert/mrotary_embedding_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include <cuda_fp16.h>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

struct MRotaryEmbeddingStrides {
  int64_t batch;
  int64_t head;
  int64_t sequence;
};

__host__ __device__ constexpr int64_t ComputeOffset(
    int batch, int sequence, int head, MRotaryEmbeddingStrides strides) {
  return static_cast<int64_t>(batch) * strides.batch +
         static_cast<int64_t>(sequence) * strides.sequence +
         static_cast<int64_t>(head) * strides.head;
}

static_assert(ComputeOffset(2, 0, 0, {1073741824, 524288, 2}) == 2147483648LL);

template <typename T, bool use_smem>
__global__ void MRotaryEmbeddingBSNH(
    T* output,                    // BxSxNxH
    const T* input,               // BxSxNxH
    const T* cos_cache,           // Mx(H/2)
    const T* sin_cache,           // Mx(H/2)
    const int64_t* position_ids,  // 3xBxS (T, H, W streams)
    const int sequence_length, const int num_heads, const int head_size,
    const int rotary_embedding_dim, const int max_sequence_length,
    const bool interleaved,
    const int3 mrope_section,
    const int mrope_layout,
    const float scale,
    MRotaryEmbeddingStrides in_strides, MRotaryEmbeddingStrides out_strides) {
  // B = batch size, S = sequence length, N = num heads, H = head size, M = max sequence length
  const int b = blockIdx.y;
  const int s = blockIdx.x;
  const int n = blockIdx.z;
  const int batch_size = gridDim.y;

  const int i = threadIdx.x;

  const T* input_data = input + ComputeOffset(b, s, n, in_strides);
  T* output_data = output + ComputeOffset(b, s, n, out_strides);

  [[maybe_unused]] extern __shared__ char smem_[];
  [[maybe_unused]] T* smem = reinterpret_cast<T*>(smem_);

  if constexpr (use_smem) {
    if (i < head_size) {
      smem[i] = input_data[i];
    }
    __syncthreads();
  }

  if (i >= head_size) {
    return;
  }

  if (i >= rotary_embedding_dim) {
    if constexpr (use_smem) {
      output_data[i] = smem[i];
    } else {
      output_data[i] = input_data[i];
    }
    return;
  }

  const int half_rotary_embedding_dim = rotary_embedding_dim / 2;

  int cache_idx = 0;
  T sign = 0;
  int j = 0;
  if (interleaved) {
    cache_idx = (i / 2) % half_rotary_embedding_dim;
    sign = (i % 2 == 0) ? -1 : 1;
    j = (i % 2 == 0) ? i + 1 : i - 1;
  } else {
    cache_idx = i % half_rotary_embedding_dim;
    sign = (i < half_rotary_embedding_dim) ? -1 : 1;
    j = (i + half_rotary_embedding_dim) % rotary_embedding_dim;
  }

  // Determine which position stream (0=T, 1=H, 2=W) owns this cache column, per mrope_layout.
  int stream = 0;  // default: Temporal
  if (mrope_layout == 0) {
    // Sectioned/Chunked: contiguous [T]*section.x + [H]*section.y + [W]*section.z
    if (cache_idx < mrope_section.x) {
      stream = 0;
    } else if (cache_idx < mrope_section.x + mrope_section.y) {
      stream = 1;
    } else {
      stream = 2;
    }
  } else {
    // Interleaved: T everywhere by default; H/W punch in every 3rd column.
    const int64_t h_length = static_cast<int64_t>(mrope_section.y) * 3;
    const int64_t w_length = static_cast<int64_t>(mrope_section.z) * 3;
    if (cache_idx % 3 == 1 && cache_idx < h_length) {
      stream = 1;
    } else if (cache_idx % 3 == 2 && cache_idx < w_length) {
      stream = 2;
    } else {
      stream = 0;
    }
  }

  const int64_t position_id = position_ids[(static_cast<int64_t>(stream) * batch_size + b) * sequence_length + s];
#if !defined(NDEBUG)
  if (i == 0) {
    CUDA_KERNEL_ASSERT(position_id >= 0 && position_id < static_cast<int64_t>(max_sequence_length));
  }
#endif
  if (position_id < 0 || position_id >= static_cast<int64_t>(max_sequence_length)) {
    output_data[i] = use_smem ? smem[i] : input_data[i];
    return;
  }
  const int64_t cache_offset = position_id * half_rotary_embedding_dim;
  const T cos_value = static_cast<T>(static_cast<float>(cos_cache[cache_offset + cache_idx]) * scale);
  const T sin_value = static_cast<T>(static_cast<float>(sin_cache[cache_offset + cache_idx]) * scale);

  if constexpr (use_smem) {
    output_data[i] = smem[i] * cos_value + sign * smem[j] * sin_value;
  } else {
    output_data[i] = input_data[i] * cos_value + sign * input_data[j] * sin_value;
  }
}

template <typename T>
Status LaunchMRotaryEmbeddingKernel(cudaStream_t stream, T* output, const T* input, const int64_t* position_ids,
                                    const T* cos_cache, const T* sin_cache, const int batch_size,
                                    const int sequence_length, const int num_heads, const int head_size,
                                    const int rotary_embedding_dim, const int max_sequence_length,
                                    const bool interleaved, const int3 mrope_section, const int mrope_layout,
                                    const float scale,
                                    const int max_threads_per_block, const bool is_input_bnsh_format) {
  ORT_ENFORCE(head_size <= max_threads_per_block, "head_size must be <= max_threads_per_block");

  MRotaryEmbeddingStrides in_strides;
  MRotaryEmbeddingStrides out_strides;
  if (is_input_bnsh_format) {
    const int64_t head_stride = static_cast<int64_t>(sequence_length) * head_size;
    in_strides = {static_cast<int64_t>(num_heads) * head_stride, head_stride, head_size};
    out_strides = in_strides;
  } else {
    const int64_t sequence_stride = static_cast<int64_t>(num_heads) * head_size;
    in_strides = {static_cast<int64_t>(sequence_length) * sequence_stride, head_size, sequence_stride};
    out_strides = in_strides;
  }

  int tpb = (head_size + 31) / 32 * 32;

  const dim3 block(tpb);
  const dim3 grid(sequence_length, batch_size, num_heads);

  if (output == input) {
    size_t smem_size = head_size * sizeof(T);
    MRotaryEmbeddingBSNH<T, true><<<grid, block, smem_size, stream>>>(
        output, input, cos_cache, sin_cache, position_ids, sequence_length,
        num_heads, head_size, rotary_embedding_dim, max_sequence_length,
        interleaved, mrope_section, mrope_layout, scale, in_strides, out_strides);
  } else {
    MRotaryEmbeddingBSNH<T, false><<<grid, block, 0, stream>>>(
        output, input, cos_cache, sin_cache, position_ids, sequence_length,
        num_heads, head_size, rotary_embedding_dim, max_sequence_length,
        interleaved, mrope_section, mrope_layout, scale, in_strides, out_strides);
  }

  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchMRotaryEmbeddingKernel<float>(
    cudaStream_t stream, float* output, const float* input, const int64_t* position_ids, const float* cos_cache,
    const float* sin_cache, const int batch_size, const int sequence_length, const int num_heads,
    const int head_size, const int rotary_embedding_dim, const int max_sequence_length, const bool interleaved,
    const int3 mrope_section, const int mrope_layout, const float scale, const int max_threads_per_block,
    const bool is_input_bnsh_format);

template Status LaunchMRotaryEmbeddingKernel<half>(
    cudaStream_t stream, half* output, const half* input, const int64_t* position_ids, const half* cos_cache,
    const half* sin_cache, const int batch_size, const int sequence_length, const int num_heads,
    const int head_size, const int rotary_embedding_dim, const int max_sequence_length, const bool interleaved,
    const int3 mrope_section, const int mrope_layout, const float scale, const int max_threads_per_block,
    const bool is_input_bnsh_format);

template Status LaunchMRotaryEmbeddingKernel<BFloat16>(
    cudaStream_t stream, BFloat16* output, const BFloat16* input, const int64_t* position_ids,
    const BFloat16* cos_cache, const BFloat16* sin_cache, const int batch_size, const int sequence_length,
    const int num_heads, const int head_size, const int rotary_embedding_dim, const int max_sequence_length,
    const bool interleaved, const int3 mrope_section, const int mrope_layout, const float scale,
    const int max_threads_per_block,
    const bool is_input_bnsh_format);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
