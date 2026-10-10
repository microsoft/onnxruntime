// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"
#include "contrib_ops/cuda/bert/attention_data.h"
#include "contrib_ops/cuda/bert/attention_kv_cache.h"
#include "core/framework/allocator.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, typename U>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T, U>& data);

template <typename T, bool output_bnsh>
Status LaunchUnpackQKV(const T* packed_qkv, T* unpacked_q, T* unpacked_k, T* unpacked_v, const int num_heads,
                       const int kv_num_heads, const int head_size, const int sequence_length, const int batch_size,
                       cudaStream_t stream, const int max_threads_per_block);

template <typename T>
Status LaunchConvertHeadSinkToFloat(
    const T* input,
    float* output,
    int count,
    cudaStream_t stream,
    int max_threads_per_block);

template <typename T>
// Also used by ONNX Attention (core/providers/cuda/llm/attention.cc) for GQA head expansion in MEA path.
Status LaunchUngroup(const GroupQueryAttentionParameters& parameters,
                     float2* k_buff, float2* v_buff,
                     const float2* k_og, const float2* v_og,
                     const int buff_seqlen, const int og_seqlen,
                     const bool is_bsnh,
                     cudaStream_t stream,
                     const int max_threads_per_block);

Status LaunchGetSequenceLengths(
    const int* total_seq_lens_minus_one,
    int* past_seq_lens,
    int* total_seq_lens,
    int* padded_seq_lens,
    int* cache_past_seq_lens,
    int* cache_total_seq_lens,
    int* evict_counts,
    const int batch_size,
    const int sequence_length,
    const bool is_first_prompt,
    const int max_total_sequence_length,
    const int kv_cache_capacity,
    const int kv_cache_real_capacity,
    cudaStream_t stream,
    const int max_threads_per_block);

// Evicts the oldest evict_counts[b] entries of a windowed (sliding_window_cache) KV cache by
// left-shifting the retained entries to the front, so the cache keeps holding the most recent
// tokens contiguously at indices [0, L). Must run before the new tokens are appended.
Status LaunchCompactKvCache(void* k_cache,
                            void* v_cache,
                            void* scratch,
                            const int* evict_counts,
                            const int* cache_past_seq_lens,
                            const int batch_size,
                            const int kv_num_heads,
                            const int capacity,
                            const int row_bytes,
                            cudaStream_t stream);

// Copies `rows` entries between two BNSH KV caches, reading from row src_offsets[b] + s (or from
// row s when src_offsets is null) and writing to row s. Used to seed the staging cache that a
// multi-token step of a windowed (sliding_window_cache) KV cache runs against, and to copy the
// surviving window back into the real cache afterwards.
Status LaunchCopyKvCacheWindow(void* dst_k,
                               void* dst_v,
                               const void* src_k,
                               const void* src_v,
                               const int batch_size,
                               const int kv_num_heads,
                               const int src_capacity,
                               const int dst_capacity,
                               const int rows,
                               const int* src_offsets,
                               const int row_bytes,
                               cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
