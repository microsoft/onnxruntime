// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "contrib_ops/cuda/bert/paged/atomics.cuh"
#include "contrib_ops/cuda/bert/paged/paged_attention/lbp_attention_task.cuh"
#include "contrib_ops/cuda/bert/paged/paged_attention/lbp_attention_gqa_task.cuh"

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

template <int NumThreads, int HeadSize, int PageSize, typename TIO, typename TKV, typename TSB, typename Config>
__global__ void
__launch_bounds__(NumThreads, NumThreads / constant::WarpSize) lbp_attention_data_parallel_kernel(
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
  using Task = std::conditional_t<
      Config::NumQueriesPerCta == 1,
      PagedAttentionTask<NumThreads, HeadSize, PageSize, TI, TO, TKV, DataParallelWorker, Config, KVConfig<TSB>>,
      PagedGroupQueryAttentionTask<NumThreads, HeadSize, PageSize, TI, TO, TKV, DataParallelWorker, Config, KVConfig<TSB>>>;
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

template <int NumThreads, typename Tensor>
__forceinline__ __device__ float2
flash_acc_precompute_global_stats(
    const Tensor& max_sum_view,
    int task_chunk_start, int task_chunk_end
) {
  constexpr int NumWarps = NumThreads / constant::WarpSize;

  __shared__ float reduction_buffer[NumWarps];
  __shared__ float2 smem_max_sum[1];

  float global_max = std::numeric_limits<float>::lowest();
  float global_sum = 0.0f;

  // 1. compute the global max of multiple TaskChunks
  for (int t = task_chunk_start + threadIdx.x; t < task_chunk_end; t += NumThreads) {
    float curr_max = max_sum_view(t).x;
    global_max = max(global_max, curr_max);
  }
  global_max = warp::reduce<constant::WarpSize>(global_max, [](float a, float b) { return max(a, b); });
  if (lane_id() == 0) {
    reduction_buffer[warp_id()] = global_max;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < NumWarps; i++) {
      global_max = max(reduction_buffer[i], global_max);
    }
    smem_max_sum->x = global_max;
  }
  __syncthreads();
  global_max = smem_max_sum->x;

  // 2. compute the global sum of multiple TaskChunks
  for (int t = task_chunk_start + threadIdx.x; t < task_chunk_end; t += NumThreads) {
    auto max_sum = max_sum_view(t);
    float curr_max = max_sum.x;
    float curr_sum = max_sum.y;
    float curr_factor = __expf(curr_max - global_max);
    global_sum += curr_factor * curr_sum;
  }
  global_sum = warp::reduce<constant::WarpSize>(global_sum, [](float a, float b) { return a + b; });
  if (lane_id() == 0) {
    reduction_buffer[warp_id()] = global_sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < NumWarps; i++) {
      global_sum += reduction_buffer[i];
    }
    smem_max_sum->y = global_sum;
  }
  __syncthreads();
  global_sum = smem_max_sum->y;

  return float2{global_max, global_sum};
}

template <int NumThreads, int HeadSize, typename Tensor1, typename Tensor2>
__forceinline__ __device__ void
flash_acc_with_global_stats(
    float* acc, float2 global_max_sum,               // accumulative part
    const Tensor1& inc, const Tensor2& inc_max_sum,  // incremental part, indexed as (chunk) and (chunk,dim)
    int task_chunk_start, int task_chunk_end
) {
  float global_max = global_max_sum.x;
  float global_sum = global_max_sum.y;
  float global_sum_inv = 1.0f / global_sum;

  CUTE_UNROLL
  for (int i = threadIdx.x; i < HeadSize; i += NumThreads) {
    float gloabl_acc = acc[i];
    for (int t = task_chunk_start; t < task_chunk_end; t++) {
      float curr_max = inc_max_sum(t).x;
      float curr_sum = inc_max_sum(t).y;
      float curr_factor = __expf(curr_max - global_max);
      gloabl_acc += curr_factor * (curr_sum * global_sum_inv) * inc(t, i);
    }
    acc[i] = gloabl_acc;
  }
}

template <int NumThreads, int HeadSize, int PageSize, typename TIO, typename TKV, typename Config>
__global__ void
__launch_bounds__(NumThreads) lbp_attention_reduction_kernel(
    void* __restrict__ workspace,
    TIO* __restrict__ out,                 // [num_seqs, num_heads, head_size]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int num_seqs,
    const int num_heads,
    const int max_context_len
)
#if !defined(PAGED_SPLIT_COMPILATION) || defined(PAGED_ATTENTION_KERNEL_IMPL)
{
  static_assert(!Config::InplaceFlashAcc);
  static_assert(!Config::UseHeadSeq);

  using TI = TIO;
  using TO = TIO;
  using Task = PagedAttentionTask<
      NumThreads, HeadSize, PageSize, TI, TO, TKV,
      DataParallelWorker, Config>;
  using WorkspaceT = DataParallelWorkspace<Config>;

  auto ws_max_sum = WorkspaceT::max_sum(workspace, num_seqs, num_heads, max_context_len);
  auto ws_out = WorkspaceT::out(workspace, num_seqs, num_heads, HeadSize, max_context_len);

  __shared__ float smem_out[HeadSize];
  // __shared__ float2 smem_max_sum[1];
  // smem_out will not be loaded if sum in smem_max_sum is 0.0f
  // if (threadIdx.x == 0) {
  //   smem_max_sum[0] = float2{std::numeric_limits<float>::lowest(), 0.0f};
  // }
  // __syncthreads();

  int seq_head_idx = blockIdx.x;
  int seq_idx = seq_head_idx / num_heads;
  int head_idx = seq_head_idx % num_heads;
  int task_chunk_start = 0;
  constexpr int TaskChunkSeqLen = Config::TaskChunkSeqLen;
  int task_chunk_end = ceil_div(context_lens[seq_idx], TaskChunkSeqLen);

  // implements a variant of flash_acc reduction over the workspace output
  // for (int t = task_chunk_start; t < task_chunk_end; t++) {
  //   Task::flash_acc(
  //       smem_out, smem_max_sum,
  //       &ws_out(t, seq_idx, head_idx, _0{}), &ws_max_sum(t, seq_idx, head_idx)
  //   );
  // }
  CUTE_UNROLL
  for (int i = threadIdx.x; i < HeadSize; i += NumThreads) {
    smem_out[i] = 0.0f;
  }
  // __syncthreads();
  float2 global_max_sum = flash_acc_precompute_global_stats<NumThreads>(ws_max_sum(_, seq_idx, head_idx), task_chunk_start, task_chunk_end);
  flash_acc_with_global_stats<NumThreads, HeadSize>(smem_out, global_max_sum, ws_out(_, seq_idx, head_idx, _), ws_max_sum(_, seq_idx, head_idx), task_chunk_start, task_chunk_end);

  auto gO = make_tensor(make_gmem_ptr(out), make_layout(make_shape(num_seqs, num_heads, Int<HeadSize>{}), LayoutRight{}));
  Task::write_flash_inc(
      &gO(seq_idx, head_idx, _0{}), nullptr,
      smem_out, nullptr
  );
}
#else
    ;
#endif

}  // namespace onnxruntime::contrib::paged
