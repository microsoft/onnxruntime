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

// Fill an int32 buffer with a constant value entirely on device.
// CUDA-graph-capturable alternative to host vector + cudaMemcpyAsync.
Status LaunchFillInt32(int* output, int value, int count, cudaStream_t stream, int max_threads_per_block);

}  // namespace cuda
}  // namespace onnxruntime
