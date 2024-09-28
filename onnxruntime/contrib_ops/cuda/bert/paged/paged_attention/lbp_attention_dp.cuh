// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "contrib_ops/cuda/bert/paged/atomics.cuh"
#include "contrib_ops/cuda/bert/paged/paged_attention/lbp_attention_task_selector.cuh"

namespace onnxruntime::contrib::paged {

template <bool InplaceFlashAcc, bool UseHeadSeq, int NumQueriesPerCta = 1>
struct DataParallelConfig : public TaskConfig<256, InplaceFlashAcc, UseHeadSeq, true, NumQueriesPerCta> {
  inline static constexpr int MaxNumSeqs = 1024;
  inline static constexpr int MaxNumHeads = 256;
};

struct DataParallelWorker : public IWorker<DataParallelWorker> {
  bool has_work = true;

  __forceinline__ __device__ void
  take_work(int& chunk_start, int& chunk_end) {
    if (has_work) {
      has_work = false;
      chunk_start = blockIdx.y;
      chunk_end = blockIdx.y + 1;
    }
  }

  __forceinline__ __device__ void broadcast_work(void*, int&, int&) { /* do nothing */ };
};

template <typename Config>
struct DataParallelWorkspace;

template <int NumQueriesPerCta>
struct DataParallelWorkspace<DataParallelConfig<true, true, NumQueriesPerCta>> {
  using Config = DataParallelConfig<true, true, NumQueriesPerCta>;
  float2 max_sum_[Config::MaxNumSeqs * Config::MaxNumHeads];

  static void
  create(stream_t stream, void** workspace, size_t* size, int /*num_seqs*/, int /*num_heads*/, int /*head_size*/, int /*max_context_len*/) {
    size_t total_bytes = sizeof(DataParallelWorkspace);
    if (size != nullptr) {
      *size = total_bytes;
    }
    if (workspace != nullptr) {
      CUDA_CHECK(cudaMallocAsync(workspace, total_bytes, stream));
    }
  }

  static void
  init(stream_t stream, void* workspace, int /*num_seqs*/, int /*num_heads*/, int /*head_size*/, int /*max_context_len*/) {
    CUDA_CHECK(cudaMemsetAsync(workspace, 0, sizeof(DataParallelWorkspace), stream));
  }

  static void
  destroy(stream_t stream, void* workspace) {
    CUDA_CHECK(cudaFreeAsync(workspace, stream));
  }

  __forceinline__ __device__ static auto
  max_sum(void* workspace, int /*num_seqs*/, int /*num_heads*/, int /*max_context_len*/) {
    static_assert(Config::UseHeadSeq);
    return make_tensor(
        make_gmem_ptr<float2>(workspace),
        Layout<
            Shape<_1, Int<Config::MaxNumSeqs>, Int<Config::MaxNumHeads>>,
            Stride<_0, _1, Int<Config::MaxNumSeqs>>>{}
    );
  }
};

template <int NumQueriesPerCta>
struct DataParallelWorkspace<DataParallelConfig<false, false, NumQueriesPerCta>> {
  using Config = DataParallelConfig<false, false, NumQueriesPerCta>;

  static void create(stream_t stream, void** workspace, size_t* size, int num_seqs, int num_heads, int head_size, int max_context_len) {
    int num_task_chunks = ceil_div(max_context_len, Config::TaskChunkSeqLen);
    size_t maxsum_bytes = ceil_div(num_task_chunks * num_seqs * num_heads * sizeof(float2), 256) * 256;
    size_t out_bytes = num_task_chunks * num_seqs * num_heads * head_size * sizeof(float);
    size_t total_bytes = maxsum_bytes + out_bytes;
    if (size != nullptr) {
      *size = total_bytes;
    }
    if (workspace != nullptr) {
      CUDA_CHECK(cudaMallocAsync(workspace, total_bytes, stream));
    }
  }

  static void
  init(stream_t stream, void* workspace, int num_seqs, int num_heads, int head_size, int max_context_len) {
    // do nothing
  }

  static void
  destroy(stream_t stream, void* workspace) {
    CUDA_CHECK(cudaFreeAsync(workspace, stream));
  }

  __forceinline__ __device__ static auto
  max_sum(void* workspace, int num_seqs, int num_heads, int max_context_len) {
    static_assert(!Config::UseHeadSeq);
    constexpr int TaskChunkSeqLen = Config::TaskChunkSeqLen;
    int num_task_chunks = ceil_div(max_context_len, TaskChunkSeqLen);
    return make_tensor(
        make_gmem_ptr<float2>(workspace),
        make_layout(make_shape(num_task_chunks, num_seqs, num_heads), LayoutRight{})
    );
  }

  __forceinline__ __device__ static auto
  out(void* workspace, int num_seqs, int num_heads, int head_size, int max_context_len) {
    constexpr int TaskChunkSeqLen = Config::TaskChunkSeqLen;
    int num_task_chunks = ceil_div(max_context_len, TaskChunkSeqLen);
    size_t maxsum_bytes = ceil_div(num_task_chunks * num_seqs * num_heads * sizeof(float2), 256) * 256;
    auto* out_base = reinterpret_cast<float*>(static_cast<char*>(workspace) + maxsum_bytes);
    return make_tensor(
        make_gmem_ptr<float>(out_base),
        make_layout(make_shape(num_task_chunks, num_seqs, num_heads, head_size), LayoutRight{})
    );
  }
};

#if !defined(PAGED_ATTENTION_MAXNREG)
#define PAGED_ATTENTION_MAXNREG 128
#endif

template <int NumThreads, int HeadSize, int PageSize, typename TIO, typename TKV, typename TSB, typename Config>
__global__ void
PAGED_LAUNCH_BOUNDS(NumThreads, PAGED_ATTENTION_MAXNREG) lbp_attention_data_parallel_kernel(
    void* __restrict__ workspace,
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

  int seq_head_idx = blockIdx.x;
  int task_chunk_start = blockIdx.y;
  int num_queries_per_kv = num_heads / num_kv_heads;

  int seq_idx = seq_head_idx / (num_heads / Config::NumQueriesPerCta);
  int head_idx = (seq_head_idx % (num_heads / Config::NumQueriesPerCta)) * Config::NumQueriesPerCta;
  int kv_head_idx = head_idx / num_queries_per_kv;

  // NOTE: nvcc bug?
  //  error: identifier "onnxruntime::contrib::paged::TaskConfig<(int)512> ::TaskChunkSeqLen" is undefined in device code
  constexpr int TaskChunkSeqLen = Config::TaskChunkSeqLen;
  if (task_chunk_start >= ceil_div(context_lens[seq_idx], TaskChunkSeqLen)) {
    return;
  }

  using TI = TIO;
  using TO = std::conditional_t<Config::InplaceFlashAcc, TIO, float>;
  using Task = typename PagedAttentionTaskSelector<
      NumThreads, HeadSize, PageSize, TI, TO, TKV, DataParallelWorker, Config, KVConfig<TSB>>::Task;
  using WorkspaceT = DataParallelWorkspace<Config>;

  auto workspace_max_sum_tensor = WorkspaceT::max_sum(workspace, num_seqs, num_heads, max_context_len);
  float2* ws_max_sum;
  TO* maybe_ws_out;
  if constexpr (Config::InplaceFlashAcc) {
    ws_max_sum = &workspace_max_sum_tensor(_0{}, _0{}, _0{});
    maybe_ws_out = out;  // inplace flash acc to the final output
  } else {
    ws_max_sum = &workspace_max_sum_tensor(task_chunk_start, _0{}, _0{});
    auto ws_out_tensor = WorkspaceT::out(workspace, num_seqs, num_heads, HeadSize, max_context_len);
    maybe_ws_out = &ws_out_tensor(task_chunk_start, _0{}, _0{}, _0{});
  }

  DataParallelWorker w{};
  Task::attention(
      broadcast_buffer, &w, seq_idx, head_idx, kv_head_idx, maybe_ws_out, ws_max_sum,
      q, k_cache, v_cache, scalebias, page_table, context_lens, alibi_slopes,
      scale, num_seqs, num_heads, num_kv_heads, max_num_pages_per_seq, q_stride
  );
}
#else
    ;
#endif

}  // namespace onnxruntime::contrib::paged
