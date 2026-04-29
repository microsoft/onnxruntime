// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "core/common/status.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// ============================================================================
// GQA Unfused Attention (CUDA fallback for large head_size / fp16 overflow)
// ============================================================================
//
// Purpose:
//   A numerically-stable unfused attention kernel that handles:
//     - Group-Query Attention natively (num_heads != kv_num_heads) via a
//       reshape-Q trick (no K/V head-replication) — works with MHA too.
//     - head_size > 256 in fp16/bf16 (writes QK scores into a FP32 scratch so
//       raw Q*K^T cannot overflow fp16 even when scale=1.0 — see issue #28195).
//     - Different Q and K sequence lengths (prompt with/without past).
//     - Causal mask, optional sliding-window mask, optional softcap, optional
//       additive attention bias, per-batch variable k-sequence lengths.
//
// Input layout contract:
//   Q       : [B, N_q, S_q, H]    BNSH, contiguous. N_q must be a multiple of
//             N_kv; heads within a KV group must be contiguous
//             (i.e. [B, N_kv, group_size, S_q, H]).
//   K cache : [B, N_kv, max_S_kv, H]  BNSH. Valid data is [..., 0:total_kv, :].
//   V cache : [B, N_kv, max_S_kv, H_v] BNSH. Valid data is [..., 0:total_kv, :].
//   Output  : [B, N_q, S_q, H_v]  BNSH, contiguous.
//
// Mask/softcap/scale semantics:
//   - scale is applied to raw QK (before softcap / bias).
//   - softcap (> 0) is applied after scale:  x = softcap * tanh(x / softcap).
//   - attn_bias (if non-null) is added after softcap (additive mask).
//   - causal: k > (past + q) is -inf where past = total_kv - S_q.
//   - local_window_size (>= 0): k < (past + q) - local_window_size is -inf.
//     local_window_size == -1 disables the sliding-window mask.
//
// The new kernel is suitable only as a fallback when Flash / MEA are ineligible
// (head_size > 256, past_key present with mask, GQA with MHA-only unfused, etc).
// The QK GEMM runs with CUBLAS_COMPUTE_32F and writes a FP32 scratch to avoid
// fp16 overflow.
//
// ============================================================================

struct GqaUnfusedAttentionParams {
  int batch_size = 0;
  int num_heads = 0;     // N_q
  int kv_num_heads = 0;  // N_kv (num_heads % kv_num_heads == 0)
  int head_size = 0;     // H
  int v_head_size = 0;   // H_v (usually == H)

  int q_sequence_length = 0;  // S_q
  int total_kv_length = 0;    // total valid K/V positions (past + new)
  int max_kv_length = 0;      // K/V buffer allocated length for stride (>= total_kv_length)

  // attn_bias (optional): shape [B or 1, N_q or 1, S_q, total_kv_length] (row-major).
  // When broadcast_dim_0 is true, batch axis is broadcast (shape[0]==1).
  // When broadcast_dim_1 is true, head axis is broadcast (shape[1]==1).
  bool broadcast_attn_bias_dim_0 = false;
  bool broadcast_attn_bias_dim_1 = false;

  bool is_causal = false;
  int local_window_size = -1;  // -1 disables sliding window
  float scale = 1.0f;
  float softcap = 0.0f;  // 0 disables

  // Per-batch K lengths (optional). When non-null, positions k >= seqlens_k[b]
  // are masked out (useful for right-padded packed batches).
  const int* seqlens_k = nullptr;
};

// Returns required scratch size in bytes. Caller must allocate
// GetGqaUnfusedAttentionWorkspaceSize(...) bytes and pass as workspace.
size_t GetGqaUnfusedAttentionWorkspaceSize(int batch_size,
                                           int num_heads,
                                           int q_sequence_length,
                                           int total_kv_length);

// Compute: Y = softmax(scale * Q * K^T [softcap, causal, window, bias, seqlens_k]) * V.
// All pointers are on device. Q/K/V/output are in type T (fp16/bf16/float).
// attn_bias (if present) is in type T.
template <typename T>
common::Status LaunchGqaUnfusedAttention(
    const cudaDeviceProp& device_prop,
    cublasHandle_t cublas,
    cudaStream_t stream,
    const GqaUnfusedAttentionParams& params,
    const T* query,
    const T* key,
    const T* value,
    const T* attn_bias,
    T* output,
    void* workspace);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
