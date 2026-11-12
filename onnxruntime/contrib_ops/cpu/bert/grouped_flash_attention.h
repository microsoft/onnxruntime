// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>

#include "core/common/status.h"
#include "core/framework/allocator.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

// Attention of query heads that share key/value heads (grouped-query attention), over a key/value cache in fp32 or
// fp16. Strides are in elements. The kernel only reads the cache: appending new rows is the caller's job.
template <typename T>
struct GroupedFlashAttentionArgs {
  int batch_size = 0;
  int sequence_length = 0;  // query tokens per batch entry
  int num_heads = 0;        // query heads, a multiple of kv_num_heads
  int kv_num_heads = 0;
  int head_size = 0;    // width of Q and K
  int v_head_size = 0;  // width of V and of the output

  // Query (b, head h, token s) starts at query + b * q_batch_stride + h * q_head_stride + s * q_seq_stride.
  const T* query = nullptr;
  size_t q_batch_stride = 0;
  size_t q_head_stride = 0;
  size_t q_seq_stride = 0;

  // Cached row t of (b, kv head k) starts at key + b * k_batch_stride + k * k_head_stride + t * k_row_stride, and
  // likewise for value. Query head h reads kv head h / (num_heads / kv_num_heads).
  const T* key = nullptr;
  size_t k_batch_stride = 0;
  size_t k_head_stride = 0;
  size_t k_row_stride = 0;
  const T* value = nullptr;
  size_t v_batch_stride = 0;
  size_t v_head_stride = 0;
  size_t v_row_stride = 0;

  // Output (b, token s, head h) starts at output + b * o_batch_stride + s * o_seq_stride + h * o_head_stride.
  T* output = nullptr;
  size_t o_batch_stride = 0;
  size_t o_seq_stride = 0;
  size_t o_head_stride = 0;

  // Per batch entry. Query s attends to cached rows [0, min(causal_offsets[b] + s + 1, total_lengths[b])) when causal,
  // else to [0, total_lengths[b]). Only the first query_lengths[b] tokens are computed when query_lengths is given;
  // the output rows of the others are left untouched.
  const int* total_lengths = nullptr;
  const int* causal_offsets = nullptr;
  const int* query_lengths = nullptr;
  bool causal = true;
  // When positive, a query sees only the last local_window_size of the rows it would otherwise see.
  int local_window_size = -1;

  float scale = 1.0f;
  float softcap = 0.0f;  // when positive, scores become softcap * tanh(scores / softcap) before the bias

  // Optional additive bias for (b, h, s, cached row t) at
  // attention_bias + b * bias_batch_stride + h * bias_head_stride + s * bias_seq_stride + bias_offsets[b] + t.
  const T* attention_bias = nullptr;
  size_t bias_batch_stride = 0;
  size_t bias_head_stride = 0;
  size_t bias_seq_stride = 0;
  const int* bias_offsets = nullptr;

  // Smooth softmax: one extra logit per head (head_sink[h], or 0 without head_sink) joins the softmax denominator.
  bool use_sink = false;
  const T* head_sink = nullptr;

  const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* mlas_config = nullptr;
};

// Runs the attention with all query heads of a kv head stacked as the rows of one GEMM against each cache tile, so a
// tile is read (and, for fp16, converted to fp32) once per group instead of once per query head. The softmax runs
// online in fp32 tile by tile, and when there are fewer row blocks than threads (decode), the cache is also split
// into chunks whose partial softmax states are merged with a log-sum-exp reduction. Steps for which
// GroupedFlashAttentionReadsHalfCache holds read fp16 tiles in place instead of converting them.
template <typename T>
Status RunGroupedFlashAttention(const GroupedFlashAttentionArgs<T>& args, AllocatorPtr allocator,
                                concurrency::ThreadPool* thread_pool);

// Whether RunGroupedFlashAttention reads an fp16 cache in place with MLAS's fp32-accumulating fp16 kernels
// (MlasHalfAttnQK and MlasHalfAttnPV) for steps of sequence_length query tokens and group_size query heads per kv
// head. It does where MLAS accelerates those kernels (ARM64 with fp16 arithmetic), for steps of up to 16 query rows
// (sequence_length * group_size), i.e. decode; larger steps convert each tile to fp32 once for SGEMM, which is faster
// once enough rows share the tile.
bool GroupedFlashAttentionReadsHalfCache(int sequence_length, int group_size);

}  // namespace contrib
}  // namespace onnxruntime
