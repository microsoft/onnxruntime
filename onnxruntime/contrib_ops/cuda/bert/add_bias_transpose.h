// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Fused kernel of Add bias (optional, can be None) and Transpose.
// Shape of inputs and outputs:
//     biases:  (num_matrices, num_heads * head_size) or None
// format 0: (requires sequence_length = kv_sequence_length and qk_head_size = v_head_size when num_matrices == 3)
//     input:   (num_matrices, batch_size, sequence_length, num_heads, head_size)
//     output:  (num_matrices, batch_size, num_heads, sequence_length, head_size)
// format 1:
//     input :  (batch_size, sequence_length, num_matrices, num_heads, head_size)
//     output:  (num_matrices, batch_size, num_heads, sequence_length, head_size)
//     qkv_add_bias: (batch_size, sequence_length, num_matrices, num_heads, head_size) optional
// format 2:
//     input :  (batch_size, sequence_length, num_matrices, num_heads, head_size)
//     output:  (batch_size, sequence_length, num_heads, num_matrices, head_size)
// format 3: (requires sequence_length = kv_sequence_length and qk_head_size = v_head_size when num_matrices == 3)
//     input:   (batch_size, sequence_length, num_matrices, num_heads, head_size)
//     output:  (num_matrices, batch_size, sequence_length, num_heads, head_size)
// format 4: (requires qk_head_size == v_head_size)
//     input:   (batch_size, sequence_length, num_heads, num_matrices, head_size)
//     output:  (num_matrices, batch_size, sequence_length, num_heads, head_size)
// format 5: (requires qk_head_size == v_head_size)
//     input:   (batch_size, sequence_length, num_heads, num_matrices, head_size)
//     output:  (num_matrices, batch_size, num_heads, sequence_length, head_size)

template <typename T>
void LaunchAddBiasTranspose(
    cudaStream_t stream, const int num_matrices, const int format, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int num_heads, const int qk_head_size,
    const T* input, const T* biases, T* output, bool enable_half4, const int v_head_size, T* qkv_add_bias = nullptr,
    int total_matrix_count = -1, bool do_rotary = false, int rotary_embedding = 0, int past_sequence_length = 0);

// Add bias (optional, can be None) and Transpose for separated inputs of Q, K and V, and output Trt format.
// For self attention:
//   output:  (batch_size, sequence_length, num_heads, 3, head_size)
//   It assumes sequence_length == kv_sequence_length and head_size == v_head_size.
// For cross attention, output has Q and packed KV like the following:
//        Q:  (batch_size, sequence_length, num_heads, head_size)
//       KV:  (batch_size, kv_sequence_length, num_heads, 2, head_size)
template <typename T>
void LaunchAddBiasTransposeTrt(
    cudaStream_t stream, const int max_threads_per_block,
    const int batch_size, const int sequence_length,
    const int num_heads, const int head_size,
    const T* biases, const T* query, const T* key, const T* value, T* output,
    bool is_cross_attention, int kv_sequence_length = -1);

// Add bias (required) for separated inputs of Q, K and V.
//    Q:  (batch_size, sequence_length, num_heads, head_size)
//    K:  (batch_size, kv_sequence_length, num_heads, head_size)
//    V:  (batch_size, kv_sequence_length, num_heads, v_head_size)
template <typename T>
void LaunchAddBias(
    cudaStream_t stream, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int kv_sequence_length,
    const int num_heads, const int head_size, const int v_head_size,
    const T* biases, const T* query, const T* key, const T* value, T* q, T* k, T* v);

// Add bias (required) for Q:  (batch_size, sequence_length, num_heads, head_size)
template <typename T>
void LaunchAddBias(
    cudaStream_t stream, const int max_threads_per_block,
    const int batch_size, const int sequence_length,
    const int num_heads, const int head_size,
    const T* biases, const T* query, T* q);

// Add bias (optional, can be None) transpose kernel defined in packed_multihead_attention_impl.cu.
// Support the following format transforms (for float and half only).
// source_format  => target_format:
//   Q_K_V_TNH    => Q_K_V_BNSH (requires token_offset)
//   Q_K_V_TNH    => Q_K_V_TNH
//   Q_K_V_TNH    => QKV_TN3H
//   QKV_TN3H     => Q_K_V_BNSH (requires token_offset)
//   QKV_TN3H     => Q_K_V_TNH
//   QKV_TN3H     => QKV_TN3H
template <typename T>
void AddBiasTransposePacked(
    const T* query, const T* key, const T* value, const T* bias, T* output,
    const int batch_size, const int sequence_length,
    const int num_heads, const int qk_head_size, const int v_head_size,
    AttentionQkvFormat source_format, AttentionQkvFormat target_format,
    const int32_t* token_offset, int32_t token_count,
    cudaStream_t stream);

// Add bias (required) transpose kernel defined in packed_attention_impl.cu.
// Support the following format transforms (for float and half only):
//     format          transform
//     Q_K_V_BNSH:     Tx3xNxH => 3xBxNxSxH (requires token_offset)
//     Q_K_V_BSNH:     Tx3xNxH => 3xTxNxH
//     QKV_BSN3H:      Tx3xNxH => TxNx3xH
template <typename T>
void AddBiasTransposePacked(
    const T* input, const T* biases, T* output,
    const int batch_size, const int sequence_length,
    const int num_heads, const int qk_head_size, const int v_head_size,
    AttentionQkvFormat format, const int32_t* token_offset, int32_t token_count,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
