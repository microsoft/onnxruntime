// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/common/status.h"

namespace onnxruntime {
namespace cuda {

// Convert a boolean attention mask to sequence lengths with a configurable offset.
//
// The mask is expected to have the following properties:
// 1. It represents right-padding only (valid tokens first, padding at the end)
// 2. All-false masks (zero-length sequence) are valid; otherwise mask should start with True
// 3. True values should be contiguous, followed by contiguous False (padding) values
// 4. The mask must be broadcastable to (batch_size, num_heads, q_seq_len, total_seq_len)
//
// For 2D mask (q_seq_len, total_seq_len): broadcasts over batch; uses first query position (row 0)
// For 3D mask (num_heads, q_seq_len, total_seq_len): broadcasts across batches, uses first head/q
// For 4D mask (B, H, q_seq_len, total_seq_len): uses first head, first q position
//
// seqlen_offset adjusts the raw token count:
//   seqlens_k[b] = num_true_tokens + seqlen_offset
//
// Common offsets:
//   0: total valid token count (for decode Step 4 where mha_fwd_kvcache reads from
//      pre-populated cache with k_new=nullptr, and for MEA custom right padding)
//  -N: subtract N from count (for decode with mha_fwd_kvcache where N=kv_sequence_length,
//      giving the number of tokens already in cache BEFORE appending new ones)
//
// Note: Mask validity (right-padding convention, contiguous True/False)
//   is checked via CUDA_KERNEL_ASSERT inside the kernel (debug builds only).
//   In release builds, non-contiguous masks produce memory-safe but semantically incorrect output:
//   seqlens_k is computed as the count of leading True values (up to the first False),
//   ignoring any True values that appear after the first False.
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
    int seqlen_offset = 0);

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

// Convert nonpad_kv_seqlen (int64, per-batch valid KV lengths) to seqlens_k (int32)
// as actual token count. Flash attention's mha_fwd_kvcache expects seqlens_k_ = number
// of valid tokens.
Status LaunchConvertNonpadKvSeqlenToFlashSeqlensK(
    const int64_t* nonpad_kv_seqlen,
    int* seqlens_k,
    int batch_size,
    int total_sequence_length,
    cudaStream_t stream,
    int max_threads_per_block);

// Convert nonpad_kv_seqlen to an additive attention bias for the MHA unfused path.
// Generates a (batch_size, q_seq_len, total_seq_len) tensor where:
//   position t < nonpad_kv_seqlen[b] → 0.0 (attend)
//   position t >= nonpad_kv_seqlen[b] → mask_filter_value (mask out)
template <typename T>
Status LaunchConvertNonpadKvSeqlenToAttentionBias(
    const int64_t* nonpad_kv_seqlen,
    T* attention_bias,
    int batch_size,
    int q_seq_len,
    int total_seq_len,
    float mask_filter_value,
    cudaStream_t stream,
    int max_threads_per_block);

// Additively compose an addend bias into an existing bias buffer in-place.
// Supports cyclic broadcasting: addend of size [q, t] is repeated over batch
// to compose with a bias of size [B, q, t]. When both have the same number
// of elements (e.g. 4D mask [B, 1, q, t]), it performs a direct element-wise add.
template <typename T>
Status LaunchAddBiasInPlace(
    T* bias,
    const T* addend,
    int64_t total_elements,
    int64_t addend_elements,
    cudaStream_t stream,
    int max_threads_per_block);

// Fill an int32 buffer with a constant value entirely on device.
// CUDA-graph-capturable alternative to host vector + cudaMemcpyAsync.
Status LaunchFillInt32(int* output, int value, int count, cudaStream_t stream, int max_threads_per_block);

}  // namespace cuda
}  // namespace onnxruntime
