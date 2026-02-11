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
//   Error status if mask is invalid (not right-padding, doesn't start with True, etc.)
//
// Note: This function validates the mask on GPU and will return an error if:
//   - The mask doesn't start with True for any batch
//   - The True/False values are not contiguous (e.g., True, False, True pattern)
Status LaunchConvertMaskToSeqlensK(
    const bool* attn_mask_bool,
    int* seqlens_k,
    int* validation_result,  // GPU buffer for validation, size = batch_size
    int batch_size,
    int total_seq_len,
    int mask_dims,
    int64_t mask_dim0,
    int64_t mask_dim1,
    int64_t mask_dim2,
    cudaStream_t stream,
    int max_threads_per_block);

}  // namespace cuda
}  // namespace onnxruntime
