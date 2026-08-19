// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

#include <cstdint>
#include <string>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class VarlenLinearAttention final : public onnxruntime::cuda::CudaKernel {
 public:
  VarlenLinearAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int q_num_heads_;
  int kv_num_heads_;
  std::string update_rule_;
  float scale_;
  // Leading (axis-0) extent of past_state / present_state; 0 means no window axis (single state).
  int state_window_;
};

// Launches the packed varlen linear-attention recurrence.
//
// query/key/value/decay/beta/output hold every sequence's tokens back to back along axis 0.
// cu_seqlens is a device int32 tensor of length (batch_size + 1): sequence i occupies the
// half-open token range [cu_seqlens[i], cu_seqlens[i + 1]). Offset *values* are a trusted
// producer precondition -- the caller only validates tensor shapes/types on the host, never
// synchronizes to read offsets back, and this kernel indexes tokens by the offsets as-is.
//
// all_ones is a host-known precondition -- total_tokens == batch_size established from tensor
// shapes -- that selects a dedicated one-token-per-sequence fast path without reading cu_seqlens
// at all (token i belongs to sequence i directly).
template <typename T>
Status LaunchVarlenLinearAttentionKernel(
    cudaStream_t stream,
    const T* query,             // [total_tokens, q_num_heads * d_k]
    const T* key,               // [total_tokens, n_k_heads * d_k]
    const T* value,             // [total_tokens, kv_num_heads * d_v]
    const T* decay,             // [total_tokens, kv_num_heads] or [total_tokens, kv_num_heads * d_k] or nullptr
    const T* beta,              // [total_tokens, kv_num_heads] or [total_tokens, 1] or nullptr
    T* output,                  // [total_tokens, max(q_num_heads, kv_num_heads) * d_v]
    const T* past_state,        // [state_window, batch_size, kv_num_heads, d_k, d_v] -- may alias present_state
    T* present_state,           // [state_window, batch_size, kv_num_heads, d_k, d_v]
    const int32_t* cu_seqlens,  // [batch_size + 1], device-resident
    int batch_size,
    bool all_ones,
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
    int multiprocessor_count,
    int max_threads_per_block,
    size_t max_shared_memory_per_block,
    // Axis-0 extent W of past_state / present_state (>= 1). Right-aligned per sequence: for a
    // sequence of length L, local token t writes slot t + W - L and negative slots are skipped,
    // so slot W-1 always holds the state after that sequence's last token. Pass 1 for a plain
    // single-state tensor with no window axis.
    int state_window = 1);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
