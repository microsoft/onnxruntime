// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cpu/bert/attention_common.h"

using onnxruntime::Stream;
using onnxruntime::contrib::AttentionQkvFormat;

namespace onnxruntime::cudnn_sdpa {

bool is_stable();

bool is_supported(const cudaDeviceProp& dprops,
                  int num_heads_q,
                  int num_heads_kv,
                  int head_size_qk,
                  int head_size_v,
                  int sequence_length_q,
                  int sequence_length_kv,
                  bool is_causal);

void run(
    void* output,
    void* q,
    void* k,
    void* v,
    void* bias,                     // (optional) attention bias with shape [b or 1, h_q or 1, s_q, s_kv].
    int* mask_sequence_lengths_q,   // (optional) sequence lengths of q for padding mask. Shape: [batch_size]
    int* mask_sequence_lengths_kv,  // (optional) sequence lengths of k or v for padding mask. Shape: [batch_size]
    int batch_size,
    int num_heads_q,
    int num_heads_kv,
    int head_size_qk,
    int head_size_v,
    int sequence_length_q,
    int sequence_length_kv,
    float scale,
    bool is_causal,
    bool is_bf16,                    // True if bfloat16, otherwise float16
    bool broadcast_attn_bias_dim_0,  // broadcast attention bias dimension 0
    bool broadcast_attn_bias_dim_1,  // broadcast attention bias dimension 1
    int sliding_window,              // sliding window length. 0 means no sliding window.
    AttentionQkvFormat qkv_format,   // Q_K_V_BNSH, Q_K_V_BSNH, Q_K_V_BSNH_BNSH_BNSH are supported
    cudnnHandle_t handle,
    Stream* stream,
    AllocatorPtr allocator);

// Paged-KV variant of the SDPA above. The K/V "cache" tensors point into a shared block pool with
// physical layout [cache_num_blocks, block_size, num_heads_kv, head_size]; a per-batch page_table
// selects which blocks belong to which sequence. Same masking surface as run() with these first-cut
// restrictions (matching cuDNN 9.5+ paged SDPA support):
//   * s_q == 1 (decode, one query token per batch entry). Q is BNSH_1: [batch_size, num_heads_q, 1,
//     head_size_qk].
//   * No attention bias, no softcap, no sliding window, no head sink.
//   * Causal masking is a no-op for s_q == 1 (the token attends to all keys up to its own
//     position, which the padding mask bounds), so this entry point never asks for it.
//   * mask_sequence_lengths_kv is required: [batch_size] int32 device buffer of per-sequence KV
//     lengths.
// Shape gate only; is_supported_paged does not probe the cuDNN planner. The paged planner is
// validated on H100+ (sm>=90), so this shape gate refuses sm_8x uniformly -- callers that opt in
// on sm_8x fall through to another PagedAttention backend rather than reaching a graph->build()
// planner-rejection path with no in-op fallback.
bool is_supported_paged(const cudaDeviceProp& dprops,
                        int num_heads_q,
                        int num_heads_kv,
                        int head_size_qk,
                        int head_size_v,
                        int sequence_length_q,       // must be 1
                        int max_sequence_length_kv,  // upper bound for graph build
                        int block_size);

// Pre-dispatch buildability probe. Returns true iff a cuDNN paged graph for these shape/type
// parameters is present in the thread-local cache OR was just built successfully. On planner
// rejection it returns false without throwing, so the PagedAttention cascade can clear
// use_cudnn_paged for the current Run and fall back to FlashAttention / MemoryEfficientAttention.
// The lookup key is byte-identical to the one run_paged uses, so the caller must invoke this
// every Run rather than caching the answer in a node-scalar: buildability is a per-(thread,
// shape) property (the cache is thread_local and keyed on the full PagedGraphParams).
//
// Capture-safe: on a cache miss with `stream` capturing the returned graph, this returns false
// rather than attempting a non-capturable build. A well-behaved producer therefore warms the
// cache with at least one non-capturing Compute per (thread, shape) before capture.
//
// The graph is compiled at max_seq_len_kv == max_num_blocks_per_seq * block_size (the natural
// page-table capacity) because cuDNN 9.12's planner rejects any smaller value; per-sequence
// lengths in mask_sequence_lengths_kv still bound actual attention range, so this coarsening is
// semantically safe.
bool try_build_paged_graph(
    int batch_size,
    int num_heads_q,
    int num_heads_kv,
    int head_size_qk,
    int head_size_v,
    int cache_num_blocks,
    int block_size,
    int max_num_blocks_per_seq,
    float scale,
    bool is_bf16,
    cudnnHandle_t handle,
    Stream* stream);

// Executes the paged SDPA. Returns true on success, false if the cuDNN graph is unavailable
// (planner rejection cached from a prior try_build_paged_graph, cache miss on a capturing stream,
// or a build attempted here that the planner rejected). On false the output is not written, so the
// caller can propagate a Status without leaving partial state.
bool run_paged(
    void* output,                   // [batch_size, num_heads_q, 1, head_size_v]
    void* q,                        // [batch_size, num_heads_q, 1, head_size_qk]
    void* k_cache,                  // [cache_num_blocks, block_size, num_heads_kv, head_size_qk]
    void* v_cache,                  // [cache_num_blocks, block_size, num_heads_kv, head_size_v]
    int* block_table,               // [batch_size, max_num_blocks_per_seq] int32
    int* mask_sequence_lengths_kv,  // [batch_size] int32 per-sequence KV length
    int batch_size,
    int num_heads_q,
    int num_heads_kv,
    int head_size_qk,
    int head_size_v,
    int cache_num_blocks,  // leading dimension of the block pool
    int block_size,
    int max_num_blocks_per_seq,  // stride of block_table's leading dim
    float scale,
    bool is_bf16,  // True if bfloat16, otherwise float16
    cudnnHandle_t handle,
    Stream* stream,
    AllocatorPtr allocator);

}  // namespace onnxruntime::cudnn_sdpa
