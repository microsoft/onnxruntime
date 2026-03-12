// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/llm/attention_mask_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

// CUDA kernel to convert boolean attention mask to sequence lengths.
// Also validates that the mask follows right-padding convention via CUDA_KERNEL_ASSERT.
//
// The kernel processes one batch per thread.
// For each batch, it finds the first False in the mask row, which indicates
// where padding starts. The sequence length is the index of first False.
//
// Validation (via CUDA_KERNEL_ASSERT, reported asynchronously):
// - All-false masks are valid (represents fully masked / zero-length sequence)
// - After the first False, all remaining elements must be False (contiguous padding)
//
// Handle broadcasting:
// - 2D mask (q_seq_len, total_seq_len): broadcasts over batch; uses first query position (row 0)
// - 3D mask (num_heads, q_seq_len, total_seq_len): broadcasts to [1, num_heads, q_seq, total_seq]
//   No per-batch variation; uses first head, first q position for all batches
// - 4D mask (B, H, q_seq_len, total_seq_len): we look at first head, first q position
__global__ void ConvertMaskToSeqlensKernel(
    const bool* __restrict__ attn_mask,
    int* __restrict__ seqlens_k,
    const int batch_size,
    const int total_seq_len,
    const int mask_dims,
    const int64_t mask_dim0,
    const int64_t mask_dim1,
    const int64_t mask_dim2,
    const int seqlen_offset) {
  int batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (batch_idx >= batch_size) {
    return;
  }

  // Calculate the starting offset for this batch's mask row
  // We need to figure out which row of the mask to use based on broadcasting rules
  const bool* mask_row = nullptr;

  if (mask_dims == 2) {
    // Shape: (q_seq_len, total_seq_len) per ONNX spec. Broadcasts over batch.
    // Use first query position (row 0) for sequence length determination.
    // For 2D masks [q_seq, total_seq], only used in decode path where q_seq=1,
    // so row 0 is always correct. Flash excludes 2D bool masks for prompt.
    mask_row = attn_mask;
  } else if (mask_dims == 3) {
    // Shape: (num_heads, q_seq_len, total_seq_len)
    // This broadcasts to [1, num_heads, q_seq, total_seq] - same mask for all batches
    // We look at first head (h_idx = 0) and first q position (q_idx = 0)
    int h_idx = 0;  // First head
    int q_idx = 0;  // First query position
    // Stride: q_seq_len * total_seq_len per head
    int64_t head_stride = mask_dim1 * total_seq_len;  // mask_dim1 = q_seq_len
    int64_t q_stride = total_seq_len;
    // Same mask row for all batches since 3D has no batch dimension
    mask_row = attn_mask + h_idx * head_stride + q_idx * q_stride;
  } else {
    // 4D: Shape (B, H, q_seq_len, total_seq_len)
    // B could be batch_size or 1 (broadcast)
    // H could be num_heads or 1 (broadcast)
    // We look at first head (h_idx = 0) and first q position (q_idx = 0)
    int effective_batch = (mask_dim0 == 1) ? 0 : batch_idx;
    int h_idx = 0;  // First head
    int q_idx = 0;  // First query position
    // Strides
    int64_t batch_stride = mask_dim1 * mask_dim2 * total_seq_len;
    int64_t head_stride = mask_dim2 * total_seq_len;
    int64_t q_stride = total_seq_len;
    mask_row = attn_mask + effective_batch * batch_stride + h_idx * head_stride + q_idx * q_stride;
  }

  // Find the first False (where padding starts)
  // All elements before this should be True, all after should be False
  int seq_len;
  if (!mask_row[0]) {
    // Entire row is padding (all-false mask)
    seq_len = 0;
  } else {
    seq_len = total_seq_len;  // Default: all True (no padding)
    bool found_first_false = false;

    for (int i = 1; i < total_seq_len; ++i) {
      bool current = mask_row[i];

      if (!found_first_false && !current) {
        // Found first False - this is where padding starts
        seq_len = i;
        found_first_false = true;
      } else if (found_first_false && current) {
        // Found True after False - mask is not contiguous (invalid)
        CUDA_KERNEL_ASSERT(false);  // mask must be contiguous (no True after False)
      }
    }
  }

  // seqlens_k output: seq_len + seqlen_offset
  // Decode with past (seqlen_offset=-kv_seq_len): pre-append cache count
  // Prompt/MEA (seqlen_offset=0): actual token count
  // Clamp to 0: all-false mask (seq_len=0) with negative decode offset
  // would produce negative seqlens_k, which is undefined in Flash kernels.
  seqlens_k[batch_idx] = max(0, seq_len + seqlen_offset);
}

// Convert boolean mask to sequence lengths with a configurable offset.
// seqlens_k[b] = num_true_tokens + seqlen_offset
Status LaunchConvertMaskToFlashSeqlensK(
    const bool* attn_mask_bool,
    int* seqlens_k,
    int batch_size,
    int total_seq_len,
    int mask_dims,
    int64_t mask_dim0,
    int64_t mask_dim1,
    int64_t mask_dim2,
    cudaStream_t stream,
    int max_threads_per_block,
    int seqlen_offset) {
  if (batch_size == 0 || total_seq_len == 0) {
    return Status::OK();
  }

  int threads = std::min(batch_size, max_threads_per_block);
  int blocks = (batch_size + threads - 1) / threads;

  ConvertMaskToSeqlensKernel<<<blocks, threads, 0, stream>>>(
      attn_mask_bool,
      seqlens_k,
      batch_size,
      total_seq_len,
      mask_dims,
      mask_dim0,
      mask_dim1,
      mask_dim2,
      seqlen_offset);

  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
__global__ void ConvertBoolMaskToAttentionBiasKernel(
    const bool* __restrict__ attn_mask,
    T* __restrict__ attention_bias,
    const int64_t num_elements,
    const float mask_filter_value) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < num_elements;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    attention_bias[idx] = attn_mask[idx] ? T(0.0f) : T(mask_filter_value);
  }
}

template <typename T>
Status LaunchConvertBoolMaskToAttentionBias(
    const bool* attn_mask_bool,
    T* attention_bias,
    int64_t num_elements,
    float mask_filter_value,
    cudaStream_t stream,
    int max_threads_per_block) {
  if (num_elements == 0) {
    return Status::OK();
  }

  int threads = static_cast<int>(std::min(static_cast<int64_t>(max_threads_per_block), num_elements));
  int64_t blocks = (num_elements + threads - 1) / threads;
  // Cap grid size to avoid exceeding CUDA gridDim.x limit (2^31 - 1).
  // The grid-stride loop in the kernel handles the overflow.
  constexpr int64_t kMaxGridDimX = 65535;
  unsigned int grid_size = static_cast<unsigned int>(std::min(blocks, kMaxGridDimX));

  ConvertBoolMaskToAttentionBiasKernel<T><<<grid_size, threads, 0, stream>>>(
      attn_mask_bool, attention_bias, num_elements, mask_filter_value);

  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchConvertBoolMaskToAttentionBias<float>(
    const bool*, float*, int64_t, float, cudaStream_t, int);
template Status LaunchConvertBoolMaskToAttentionBias<__half>(
    const bool*, __half*, int64_t, float, cudaStream_t, int);
template Status LaunchConvertBoolMaskToAttentionBias<__nv_bfloat16>(
    const bool*, __nv_bfloat16*, int64_t, float, cudaStream_t, int);

// Convert nonpad_kv_seqlen (int64) to seqlens_k (int32) as actual token count.
// Flash attention's mha_fwd_kvcache expects seqlens_k_ = number of valid tokens.
__global__ void ConvertNonpadKvSeqlenToFlashSeqlensKKernel(
    const int64_t* __restrict__ nonpad_kv_seqlen,
    int* __restrict__ seqlens_k,
    const int batch_size,
    const int total_sequence_length) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < batch_size) {
    int64_t val = nonpad_kv_seqlen[idx];
    CUDA_KERNEL_ASSERT(val >= 0);
    CUDA_KERNEL_ASSERT(val <= static_cast<int64_t>(total_sequence_length));
    val = max(static_cast<int64_t>(0), min(val, static_cast<int64_t>(total_sequence_length)));
    seqlens_k[idx] = static_cast<int>(val);  // count, not index
  }
}

Status LaunchConvertNonpadKvSeqlenToFlashSeqlensK(
    const int64_t* nonpad_kv_seqlen,
    int* seqlens_k,
    int batch_size,
    int total_sequence_length,
    cudaStream_t stream,
    int max_threads_per_block) {
  if (batch_size == 0) {
    return Status::OK();
  }

  int threads = std::min(batch_size, max_threads_per_block);
  int blocks = (batch_size + threads - 1) / threads;

  ConvertNonpadKvSeqlenToFlashSeqlensKKernel<<<blocks, threads, 0, stream>>>(
      nonpad_kv_seqlen, seqlens_k, batch_size, total_sequence_length);

  return CUDA_CALL(cudaGetLastError());
}

// CUDA kernel to convert nonpad_kv_seqlen to an additive attention bias.
// Generates (batch_size, q_seq_len, total_seq_len) output where:
//   position t < nonpad_kv_seqlen[b] → 0.0 (attend)
//   position t >= nonpad_kv_seqlen[b] → mask_filter_value (mask out)
template <typename T>
__global__ void ConvertNonpadKvSeqlenToAttentionBiasKernel(
    const int64_t* __restrict__ nonpad_kv_seqlen,
    T* __restrict__ attention_bias,
    const int batch_size,
    const int q_seq_len,
    const int total_seq_len,
    const float mask_filter_value) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(batch_size) * q_seq_len * total_seq_len;
  for (; idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int b = static_cast<int>(idx / (static_cast<int64_t>(q_seq_len) * total_seq_len));
    int t = static_cast<int>(idx % total_seq_len);
    int64_t valid_len = nonpad_kv_seqlen[b];
    CUDA_KERNEL_ASSERT(valid_len >= 0 && valid_len <= static_cast<int64_t>(total_seq_len));
    valid_len = max(static_cast<int64_t>(0), min(valid_len, static_cast<int64_t>(total_seq_len)));
    attention_bias[idx] = (t < static_cast<int>(valid_len)) ? T(0.0f) : T(mask_filter_value);
  }
}

template <typename T>
Status LaunchConvertNonpadKvSeqlenToAttentionBias(
    const int64_t* nonpad_kv_seqlen,
    T* attention_bias,
    int batch_size,
    int q_seq_len,
    int total_seq_len,
    float mask_filter_value,
    cudaStream_t stream,
    int max_threads_per_block) {
  int64_t total = static_cast<int64_t>(batch_size) * q_seq_len * total_seq_len;
  if (total == 0) {
    return Status::OK();
  }

  int threads = static_cast<int>(std::min(static_cast<int64_t>(max_threads_per_block), total));
  int64_t blocks = (total + threads - 1) / threads;
  constexpr int64_t kMaxGridDimX = 65535;
  unsigned int grid_size = static_cast<unsigned int>(std::min(blocks, kMaxGridDimX));

  ConvertNonpadKvSeqlenToAttentionBiasKernel<T><<<grid_size, threads, 0, stream>>>(
      nonpad_kv_seqlen, attention_bias, batch_size, q_seq_len, total_seq_len, mask_filter_value);

  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchConvertNonpadKvSeqlenToAttentionBias<float>(
    const int64_t*, float*, int, int, int, float, cudaStream_t, int);
template Status LaunchConvertNonpadKvSeqlenToAttentionBias<__half>(
    const int64_t*, __half*, int, int, int, float, cudaStream_t, int);
template Status LaunchConvertNonpadKvSeqlenToAttentionBias<__nv_bfloat16>(
    const int64_t*, __nv_bfloat16*, int, int, int, float, cudaStream_t, int);

// Add an addend bias into an existing bias buffer using cyclic broadcasting.
// Used to compose nonpad_kv_seqlen bias [B, q, t] with an attn_mask bias that
// may be smaller (e.g. 2D [q, t] broadcasts over batch).
template <typename T>
__global__ void AddBiasInPlaceKernel(
    T* __restrict__ bias,
    const T* __restrict__ addend,
    int64_t total_elements,
    int64_t addend_elements) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total_elements;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    float sum = static_cast<float>(bias[idx]) + static_cast<float>(addend[idx % addend_elements]);
    bias[idx] = T(sum);
  }
}

template <typename T>
Status LaunchAddBiasInPlace(
    T* bias,
    const T* addend,
    int64_t total_elements,
    int64_t addend_elements,
    cudaStream_t stream,
    int max_threads_per_block) {
  if (total_elements == 0 || addend_elements == 0) {
    return Status::OK();
  }

  int threads = static_cast<int>(std::min(static_cast<int64_t>(max_threads_per_block), total_elements));
  int64_t blocks = (total_elements + threads - 1) / threads;
  constexpr int64_t kMaxGridDimX = 65535;
  unsigned int grid_size = static_cast<unsigned int>(std::min(blocks, kMaxGridDimX));

  AddBiasInPlaceKernel<T><<<grid_size, threads, 0, stream>>>(
      bias, addend, total_elements, addend_elements);

  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchAddBiasInPlace<float>(float*, const float*, int64_t, int64_t, cudaStream_t, int);
template Status LaunchAddBiasInPlace<__half>(__half*, const __half*, int64_t, int64_t, cudaStream_t, int);
template Status LaunchAddBiasInPlace<__nv_bfloat16>(__nv_bfloat16*, const __nv_bfloat16*, int64_t, int64_t, cudaStream_t, int);

// Simple kernel to fill an int32 buffer with a constant value on device.
// Used for CUDA-graph-capturable seqlens_k initialization (no host memory).
__global__ void FillInt32Kernel(int* __restrict__ output, const int value, const int count) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < count) {
    output[idx] = value;
  }
}

Status LaunchFillInt32(int* output, int value, int count, cudaStream_t stream, int max_threads_per_block) {
  if (count == 0) {
    return Status::OK();
  }

  int threads = std::min(count, max_threads_per_block);
  int blocks = (count + threads - 1) / threads;

  FillInt32Kernel<<<blocks, threads, 0, stream>>>(output, value, count);

  return CUDA_CALL(cudaGetLastError());
}

}  // namespace cuda
}  // namespace onnxruntime
