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

}  // namespace onnxruntime::cudnn_sdpa
