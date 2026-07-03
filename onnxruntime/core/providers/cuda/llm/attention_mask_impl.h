// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/common/status.h"

namespace onnxruntime {
namespace cuda {

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

// Zero output elements for batches where seqlens_k == 0 (fully masked).
// Used in the MEA path only: CUTLASS epilogue computes 1/s_prime where s_prime=0,
// producing NaN for fully-masked batches. This kernel overwrites those NaN outputs
// with zeros. The unfused path produces valid finite output (mean-of-V via uniform
// softmax) and does not need this. Flash handles it natively with an early-exit.
template <typename T>
Status LaunchZeroOutputForFullyMaskedBatches(
    T* output,
    const int* seqlens_k,
    int batch_size,
    int64_t elements_per_batch,
    cudaStream_t stream,
    int max_threads_per_block);

// Zero output rows that are fully masked by the intersection of the causal frontier
// and an explicit attention bias (composed is_causal + attn_mask), per onnx/onnx#8068
// "fully-masked-row -> 0" (Bug-2). The MEA/CUTLASS path applies a finite mask sentinel
// (kCutlassSafeMaskFilterValue) rather than -inf, so a query row with no allowed key
// softmaxes to a uniform mean-of-V instead of zero. For each (batch, head, query) this
// kernel reconstructs the per-batch bottom-right/top-left causal frontier and checks the
// additive bias over the causally-allowed keys; if every such key is masked
// (bias <= masked_bias_value) the corresponding output row is overwritten with zeros
// (select-not-multiply). Output is BSNH: [batch_size, q_sequence_length, num_heads, v_head_size].
//
// seqlens_k: per-batch valid key count (external cache); pass nullptr to use
//   total_sequence_length for every batch.
// is_causal / causal_from_top_left: causal frontier selection matching the MEA params.
// broadcast_bias_dim_0 / broadcast_bias_dim_1: attn_bias broadcast over batch / head.
// masked_bias_value: the additive-bias sentinel used for masked keys (a key counts as
//   masked when bias <= this value); use the same value passed to the mask conversion.
template <typename T>
Status LaunchZeroFullyMaskedRows(
    T* output,
    const T* attn_bias,
    const int* seqlens_k,
    int batch_size,
    int num_heads,
    int q_sequence_length,
    int total_sequence_length,
    int v_head_size,
    bool is_causal,
    bool causal_from_top_left,
    bool broadcast_bias_dim_0,
    bool broadcast_bias_dim_1,
    float masked_bias_value,
    cudaStream_t stream,
    int max_threads_per_block);

// Fill an int32 buffer with a constant value entirely on device.
// CUDA-graph-capturable alternative to host vector + cudaMemcpyAsync.
Status LaunchFillInt32(int* output, int value, int count, cudaStream_t stream, int max_threads_per_block);

}  // namespace cuda
}  // namespace onnxruntime
