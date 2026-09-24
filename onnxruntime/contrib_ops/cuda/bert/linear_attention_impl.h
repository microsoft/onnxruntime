// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Fused recurrent kernel for gated_delta update rule (decode path, T tokens sequentially).
// Processes all heads in parallel on GPU; each (batch, kv_head) gets one thread block.
// State is kept in shared memory for the entire token loop to avoid global memory round-trips.
template <typename T>
Status LaunchLinearAttentionKernel(
    cudaStream_t stream,
    const T* query,       // [B, T, H_q * d_k]
    const T* key,         // [B, T, n_k * d_k]
    const T* value,       // [B, T, H_kv * d_v]
    const T* decay,       // [B, T, H_kv] or [B, T, H_kv * d_k] or nullptr
    const T* beta,        // [B, T, H_kv] or [B, T, 1] or nullptr
    T* output,            // [B, T, max(H_q, H_kv) * d_v]
    const T* past_state,  // [W, B, H_kv, d_k, d_v] -- may alias present_state
    T* present_state,     // [W, B, H_kv, d_k, d_v]
    int batch_size,
    int seq_len,
    int q_num_heads,
    int kv_num_heads,
    int n_k_heads,
    int d_k,
    int d_v,
    float scale,
    bool needs_decay,
    bool decay_per_key_dim,
    bool needs_beta,
    bool beta_per_head,
    bool needs_retrieval,
    int decode_seq_threshold,
    int row_split,
    int multiprocessor_count,
    int max_threads_per_block,
    size_t max_shared_memory_per_block,
    // Axis-0 extent W of past_state / present_state (>= 1). The window axis leads the batch axis
    // so that a slot is one contiguous [B, H_kv, d_k, d_v] block. Right-aligned: token t writes
    // slot t + W - seq_len and slots with a negative index are skipped, so slot W-1 always holds
    // the state after the last token and is the slot past_state is read from. Pass 1 for a plain
    // single-state tensor with no window axis.
    int state_window = 1);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
