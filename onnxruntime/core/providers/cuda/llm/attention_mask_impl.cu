// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/llm/attention_mask_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

// Validation error codes (stored in validation_result buffer)
constexpr int kValidationOK = 0;
constexpr int kValidationErrorNotStartWithTrue = 1;
constexpr int kValidationErrorNotContiguous = 2;

// CUDA kernel to convert boolean attention mask to sequence lengths.
// Also validates that the mask follows right-padding convention.
//
// The kernel processes one batch per thread.
// For each batch, it finds the first False in the mask row, which indicates
// where padding starts. The sequence length is the index of first False.
//
// Validation:
// - The mask must start with True (first element must be True)
// - After the first False, all remaining elements must be False (contiguous padding)
//
// Handle broadcasting:
// - 2D mask (batch_size, total_seq_len): stride = total_seq_len, batch_idx = threadIdx
// - 3D mask (num_heads, q_seq_len, total_seq_len): broadcasts to [1, num_heads, q_seq, total_seq]
//   No per-batch variation; uses first head, first q position for all batches
// - 4D mask (B, H, q_seq_len, total_seq_len): we look at first head, first q position
__global__ void ConvertMaskToSeqlensKernel(
    const bool* __restrict__ attn_mask,
    int* __restrict__ seqlens_k,
    int* __restrict__ validation_result,
    const int batch_size,
    const int total_seq_len,
    const int mask_dims,
    const int64_t mask_dim0,
    const int64_t mask_dim1,
    const int64_t mask_dim2) {
  int batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (batch_idx >= batch_size) {
    return;
  }

  // Calculate the starting offset for this batch's mask row
  // We need to figure out which row of the mask to use based on broadcasting rules
  const bool* mask_row = nullptr;

  if (mask_dims == 2) {
    // Shape: (batch_size or 1, total_seq_len)
    // If mask_dim0 == 1, broadcast across all batches
    int effective_batch = (mask_dim0 == 1) ? 0 : batch_idx;
    mask_row = attn_mask + effective_batch * total_seq_len;
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

  // Initialize validation result for this batch
  validation_result[batch_idx] = kValidationOK;

  // Check that mask starts with True
  if (!mask_row[0]) {
    validation_result[batch_idx] = kValidationErrorNotStartWithTrue;
    seqlens_k[batch_idx] = -1;  // Invalid
    return;
  }

  // Find the first False (where padding starts)
  // All elements before this should be True, all after should be False
  int seq_len = total_seq_len;  // Default: all True (no padding)
  bool found_first_false = false;

  for (int i = 1; i < total_seq_len; ++i) {
    bool current = mask_row[i];

    if (!found_first_false && !current) {
      // Found first False - this is where padding starts
      seq_len = i;
      found_first_false = true;
    } else if (found_first_false && current) {
      // Found True after False - this is invalid (not contiguous)
      validation_result[batch_idx] = kValidationErrorNotContiguous;
      seqlens_k[batch_idx] = -1;  // Invalid
      return;
    }
  }

  // seqlens_k is total_sequence_length - 1 for GQA convention
  seqlens_k[batch_idx] = seq_len - 1;
}

Status LaunchConvertMaskToSeqlensK(
    const bool* attn_mask_bool,
    int* seqlens_k,
    int* validation_result,
    int batch_size,
    int total_seq_len,
    int mask_dims,
    int64_t mask_dim0,
    int64_t mask_dim1,
    int64_t mask_dim2,
    cudaStream_t stream,
    int max_threads_per_block) {
  if (batch_size == 0 || total_seq_len == 0) {
    return Status::OK();
  }

  int threads = std::min(batch_size, max_threads_per_block);
  int blocks = (batch_size + threads - 1) / threads;

  ConvertMaskToSeqlensKernel<<<blocks, threads, 0, stream>>>(
      attn_mask_bool,
      seqlens_k,
      validation_result,
      batch_size,
      total_seq_len,
      mask_dims,
      mask_dim0,
      mask_dim1,
      mask_dim2);

  return CUDA_CALL(cudaGetLastError());
}

}  // namespace cuda
}  // namespace onnxruntime
