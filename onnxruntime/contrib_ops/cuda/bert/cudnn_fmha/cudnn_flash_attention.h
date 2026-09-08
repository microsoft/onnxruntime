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
bool is_supported_paged(const cudaDeviceProp& dprops,
                        int num_heads_q,
                        int num_heads_kv,
                        int head_size_qk,
                        int head_size_v,
                        int sequence_length_q,       // must be 1
                        int max_sequence_length_kv,  // upper bound for graph build
                        int block_size);

void run_paged(
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
    int max_sequence_length_kv,  // graph build bound; must satisfy real kv_len[i] <= this
    int cache_num_blocks,        // leading dimension of the block pool
    int block_size,
    int max_num_blocks_per_seq,  // stride of block_table's leading dim
    float scale,
    bool is_bf16,  // True if bfloat16, otherwise float16
    cudnnHandle_t handle,
    Stream* stream,
    AllocatorPtr allocator);

}  // namespace onnxruntime::cudnn_sdpa
