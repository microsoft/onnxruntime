// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/common/status.h"

namespace onnxruntime {
namespace cuda {

// Convert a boolean attention mask to sequence lengths for use with GQA kernels.
//
// The mask is expected to have the following properties:
// 1. It represents right-padding only (valid tokens first, padding at the end)
// 2. Each batch's mask should start with True (valid) values
// 3. True values should be contiguous, followed by contiguous False (padding) values
// 4. The mask must be broadcastable to (batch_size, num_heads, q_seq_len, total_seq_len)
//
// For 2D mask (batch_size, total_seq_len): uses the mask directly per batch
// For 3D mask (num_heads, q_seq_len, total_seq_len): broadcasts across batches, uses first head/q
// For 4D mask (B, H, q_seq_len, total_seq_len): uses first head, first q position
//
// Parameters:
//   attn_mask_bool: Input boolean mask on GPU (True = valid, False = padding)
//   seqlens_k: Output buffer for sequence lengths (seqlen - 1 for GQA convention)
//   batch_size: Number of batches
//   total_seq_len: Total sequence length (last dimension of mask)
//   mask_dims: Number of dimensions in the mask (2, 3, or 4)
//   mask_dim0: First dimension of mask (batch_size for 2D, num_heads for 3D, batch_size for 4D)
//   mask_dim1: Second dimension (0 for 2D, q_seq_len for 3D, num_heads for 4D)
//   mask_dim2: Third dimension (0 for 2D/3D, q_seq_len for 4D)
//   stream: CUDA stream
//   max_threads_per_block: Maximum threads per block
//
// Returns:
//   Status::OK() on success
//
// Note: Mask validity (right-padding convention, starts with True, contiguous True/False)
//   is checked asynchronously via CUDA_KERNEL_ASSERT inside the kernel. Invalid masks will
//   trigger a device-side assertion failure.
Status LaunchConvertMaskToSeqlensK(
    const bool* attn_mask_bool,
    int* seqlens_k,
    int batch_size,
    int total_seq_len,
    int mask_dims,
    int64_t mask_dim0,
    int64_t mask_dim1,
    int64_t mask_dim2,
    cudaStream_t stream,
    int max_threads_per_block);

// Convert a boolean attention mask to an additive attention bias for the MHA path.
// Maps true -> 0.0 (attend) and false -> mask_filter_value (mask out).
// The output has the same shape as the input mask.
template <typename T>
Status LaunchConvertBoolMaskToAttentionBias(
    const bool* attn_mask_bool,
    T* attention_bias,
    int64_t num_elements,
    float mask_filter_value,
    cudaStream_t stream,
    int max_threads_per_block);

}  // namespace cuda
}  // namespace onnxruntime
