// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cute/tensor.hpp"

#include "contrib_ops/cuda/bert/paged/cuda_common.cuh"
#include "contrib_ops/cuda/bert/paged/paged_attention/paged_attention.cuh"
#include "contrib_ops/cuda/bert/paged/paged_attention/lbp_attention_task_selector.cuh"

namespace onnxruntime::contrib::paged {

using namespace cute;

struct NaiveWorker : public IWorker<NaiveWorker> {
  int num_task_chunks;
  int i = 0;

  __forceinline__ __device__ void
  take_work(int& chunk_start, int& chunk_end) {
    if (i < num_task_chunks) {
      int curr_task_chunk = i++;
      chunk_start = curr_task_chunk;
      chunk_end = curr_task_chunk + 1;
    }
  }

  __forceinline__ __device__ void broadcast_work(void*, int&, int&) { /* do nothing */ };
};

#if !defined(PAGED_ATTENTION_MAXNREG)
#define PAGED_ATTENTION_MAXNREG 128
#endif

template <int NumThreads, int HeadSize, int PageSize, int ChunkSize, int NumQueriesPerCta, typename TIO, typename TKV, typename TSB>
__global__ void
PAGED_LAUNCH_BOUNDS(NumThreads, PAGED_ATTENTION_MAXNREG) paged_attention_kernel(
    TIO* __restrict__ out,                   // [num_seqs, num_heads, head_size]
    const TIO* __restrict__ q,               // [num_seqs, num_heads, head_size]
    const TKV* __restrict__ k_cache,         // [num_pages, num_kv_heads, head_size/x, page_size, x]
    const TKV* __restrict__ v_cache,         // [num_pages, num_kv_heads, head_size, page_size]
    const TSB* __restrict__ scalebias,       // [num_pages, 2, num_kv_heads, 2, head_size/chunk_size, page_size]
    const int* __restrict__ page_table,      // [num_seqs, max_num_pages_per_seq]
    const int* __restrict__ context_lens,    // [num_seqs]
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const float scale,
    const int num_seqs,
    const int num_heads,
    const int num_kv_heads,
    const int max_num_pages_per_seq,
    const int q_stride,
    const int max_context_len
)
#if !defined(PAGED_SPLIT_COMPILATION) || defined(PAGED_ATTENTION_KERNEL_IMPL)
{
  __shared__ char broadcast_buffer[BROADCAST0_BUFFER_SIZE_IN_BYTES];

  using TI = TIO;
  using NaiveConfig = TaskConfig<
      /*TaskChunkSeqLen*/ 512,
      /*InplaceFlashAcc*/ false,
      /*UseHeadSeq*/ false,
      /*SingleChunk*/ false,
      NumQueriesPerCta>;

  NaiveWorker w{};
  int seq_head_idx = blockIdx.x;
  int num_queries_per_kv = num_heads / num_kv_heads;

  int seq_idx = seq_head_idx / (num_heads / NaiveConfig::NumQueriesPerCta);
  int head_idx = (seq_head_idx % (num_heads / NaiveConfig::NumQueriesPerCta)) * NaiveConfig::NumQueriesPerCta;
  int kv_head_idx = head_idx / num_queries_per_kv;

  constexpr int TaskChunkSeqLen = NaiveConfig::TaskChunkSeqLen;
  w.num_task_chunks = ceil_div(context_lens[seq_idx], TaskChunkSeqLen);

  using Task = typename PagedAttentionTaskSelector<
      NumThreads, HeadSize, PageSize, TIO, TIO, TKV, NaiveWorker, NaiveConfig, KVConfig<TSB, ChunkSize>>::Task;
  Task::attention(
      broadcast_buffer, &w, seq_idx, head_idx, kv_head_idx, out, nullptr,
      q, k_cache, v_cache, scalebias, page_table, context_lens, alibi_slopes,
      scale, num_seqs, num_heads, num_kv_heads, max_num_pages_per_seq, q_stride
  );
}
#else
    ;
#endif

}  // namespace onnxruntime::contrib::paged
