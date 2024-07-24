// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "contrib_ops/cuda/bert/paged/atomics.cuh"
#include "contrib_ops/cuda/bert/paged/load_balance/bwd_weak.cuh"

#define TASK_DPRINTF_ENABLE 0
#define TASK_DPRINTF(fmt, ...)       \
  if constexpr (TASK_DPRINTF_ENABLE) \
    printf(fmt, ##__VA_ARGS__);
#define TASK_DPRINTF1(fmt, ...)      \
  if constexpr (TASK_DPRINTF_ENABLE) \
    if (threadIdx.x == 1)            \
      printf(fmt, ##__VA_ARGS__);

#define my_worker_id blockIdx.x

#include "contrib_ops/cuda/bert/paged/paged_attention/lbp_attention_task.cuh"

namespace onnxruntime::contrib::paged {

struct WorkStealingConfig : public TaskConfig<256> {
  inline static constexpr int MaxNumSeqs = 1024;
  inline static constexpr int MaxNumHeads = 256;
  inline static constexpr int MaxNumWorkers = 1536;
};

struct WorkStealingTaskItem {
  static constexpr uint64_t ChunkEndMask = /*  22 bits*/ 0xFFFF'FC00'0000'0000;
  static constexpr uint64_t ChunkStartMask = /*22 bits*/ 0x0000'03FF'FFF0'0000;
  static constexpr uint64_t SeqHeadMask = /*   20 bits*/ 0x0000'0000'000F'FFFF;
  static constexpr uint64_t ChunkEndOffset = 42;
  static constexpr uint64_t ChunkStartOffset = 20;
  static constexpr uint64_t SeqHeadOffset = 0;

  union {
    uint64_t packed_u64_;
    unsigned long long packed_ull_;
  };

  __forceinline__ __device__ static uint64_t
  task(int64_t seq_head, int64_t chunk_start, int64_t chunk_end) {
    return ((uint64_t(seq_head) << SeqHeadOffset) & SeqHeadMask) |
           ((uint64_t(chunk_start) << ChunkStartOffset) & ChunkStartMask) |
           ((uint64_t(chunk_end) << ChunkEndOffset) & ChunkEndMask);
  }

  __forceinline__ __device__ static uint64_t
  interval(int64_t chunk_start, int64_t chunk_end) {
    return ((uint64_t(chunk_start) << ChunkStartOffset) & ChunkStartMask) |
           ((uint64_t(chunk_end) << ChunkEndOffset) & ChunkEndMask);
  }

  __forceinline__ __device__ static constexpr uint64_t
  delta(int64_t chunk_start_delta, int64_t chunk_end_delta) {
    return ((uint64_t(chunk_start_delta) << ChunkStartOffset) & ChunkStartMask) |
           ((uint64_t(chunk_end_delta) << ChunkEndOffset) & ChunkEndMask);
  }

  __forceinline__ __device__ int
  chunk_end() const {
    return (packed_u64_ & ChunkEndMask) >> ChunkEndOffset;
  }

  __forceinline__ __device__ int
  chunk_start() const {
    return (packed_u64_ & ChunkStartMask) >> ChunkStartOffset;
  }

  __forceinline__ __device__ int
  seq_head() const {
    return (packed_u64_ & SeqHeadMask) >> SeqHeadOffset;
  }
};

struct WorkStealingWorker : public IWorker<WorkStealingWorker> {
  WorkStealingTaskItem item_;

  __forceinline__ __device__ void
  initialize_tok(int seq_head_idx, int tok_start, int tok_end, int task_chunk_size) {
    auto chunk_start = tok_start / task_chunk_size;
    auto chunk_end = ceil_div(tok_end, task_chunk_size);
    initialize_chunk(seq_head_idx, chunk_start, chunk_end);
  }

  __forceinline__ __device__ void
  initialize_chunk(int seq_head_idx, int chunk_start, int chunk_end) {
    atomicExch(&item_.packed_ull_, WorkStealingTaskItem::task(seq_head_idx, chunk_start, chunk_end));
  }

  __forceinline__ __device__ void
  take_work(int& chunk_start, int& chunk_end) {
    if (threadIdx.x != 0) {
      return;
    }
    constexpr uint64_t delta = WorkStealingTaskItem::delta(1, 0);
    WorkStealingTaskItem old;
    old.packed_ull_ = atomicAdd(&item_.packed_ull_, delta);
    auto remain_chunk_start = old.chunk_start();
    auto remain_chunk_end = old.chunk_end();
    if (remain_chunk_start >= remain_chunk_end) {
      return;
    }
    chunk_start = remain_chunk_start;
    chunk_end = remain_chunk_start + 1;
  }

  __forceinline__ __device__ void
  broadcast_work(void* broadcast_buffer, int& chunk_start, int& chunk_end) {
    auto broadcasted = broadcast0<int2>(broadcast_buffer, [&]() {
      return int2{chunk_start, chunk_end};
    });
    chunk_start = broadcasted.x;
    chunk_end = broadcasted.y;
  }

  // steal portion of the workload with consideration of workload chunk size
  __forceinline__ __device__ int
  steal(int& stolen_start, int& stolen_end, int& remain_start, int& remain_end) {
    // return start tok index and end tok index to work on
    WorkStealingTaskItem old, assumed_old, newval;
    old.packed_u64_ = volatile_load(&item_.packed_u64_);  // relaxed atomic load?
    int to_steal;
    do {
      auto old_chunk_start = old.chunk_start();
      auto old_chunk_end = old.chunk_end();
      auto remain = old_chunk_end - old_chunk_start;
      to_steal = (remain - 1) / 2 + 1;
      if (remain < 1 || to_steal < 1) {
        return -1;
      }
      newval.packed_u64_ = old.packed_u64_ + WorkStealingTaskItem::delta(0, -to_steal);
      assumed_old = old;
      old.packed_ull_ = atomicCAS(&item_.packed_ull_, assumed_old.packed_ull_, newval.packed_ull_);
    } while (assumed_old.packed_ull_ != old.packed_ull_);
    remain_start = newval.chunk_start();
    remain_end = newval.chunk_end();
    stolen_start = remain_end;
    stolen_end = stolen_start + to_steal;
    return old.seq_head();
  }
};

// NOTE: lbp_attention_init_workspace is in charge of workspace initialization
struct WorkStealingWorkspace {
  WorkStealingWorker workers_[WorkStealingConfig::MaxNumWorkers];
  BrokerWorkDistributor<next_power_of_two(WorkStealingConfig::MaxNumWorkers * 2), int> work_queue_;
  int logical_seq_head_idx_;
  int32_t num_busy_workers_;
  uint32_t steal_barrier_;
  float2 max_sum_[WorkStealingConfig::MaxNumSeqs * WorkStealingConfig::MaxNumHeads];

  static void init(WorkStealingWorkspace* self, stream_t stream) {
    size_t size_in_bytes = reinterpret_cast<char*>(&self->max_sum_) - reinterpret_cast<char*>(self);
    CUDA_CHECK(cudaMemsetAsync(self, 0, size_in_bytes, stream));
  }
};

template <int NumThreads, int HeadSize, int PageSize, typename TIO, typename TKV, typename TSB>
__global__ void
__launch_bounds__(NumThreads, NumThreads / constant::WarpSize) lbp_attention_work_stealing_kernel(
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
    const int q_stride
)
#if !defined(PAGED_SPLIT_COMPILATION) || defined(PAGED_ATTENTION_KERNEL_IMPL)
{
  using Task = PagedAttentionTask<
      NumThreads, HeadSize, PageSize, TIO, TIO, TKV,
      WorkStealingWorker, WorkStealingConfig, KVConfig<TSB>>;

  __shared__ char broadcast_buffer[BROADCAST0_BUFFER_SIZE_IN_BYTES];
  auto* ws = static_cast<WorkStealingWorkspace*>(workspace);

  constexpr int NumWarps = NumThreads / constant::WarpSize;
  constexpr int StatusAllWorkAssigned = -1;
  auto& w = ws->workers_[my_worker_id];

  while (true) {
    int my_seq_head_idx = broadcast0<int>(broadcast_buffer, [&]() {
      ws->work_queue_.enqueue(my_worker_id);
      int my_seq_head_idx = atomicAdd(&(ws->logical_seq_head_idx_), 1);
      if (my_seq_head_idx >= num_seqs * num_heads) {
        return StatusAllWorkAssigned;
      }

      atomic_add_global_i32<memory_order::Relaxed>(&ws->num_busy_workers_, 1);

      int seq_idx = my_seq_head_idx / num_heads;
      int head_idx = my_seq_head_idx % num_heads;

      if constexpr (WorkStealingConfig::UseHeadSeq) {
        int head_seq_idx = head_idx * WorkStealingConfig::MaxNumSeqs + seq_idx;
        atomic_init_max_sum(&ws->max_sum_[head_seq_idx]);
      } else {
        atomic_init_max_sum(&ws->max_sum_[my_seq_head_idx]);
      }

      w.initialize_tok(my_seq_head_idx, 0, context_lens[seq_idx], WorkStealingConfig::TaskChunkSeqLen);
      TASK_DPRINTF("worker[%d]: seqhead:%d, seq:%d, head:%d, tok[%d,%d)\n", my_worker_id, my_seq_head_idx, seq_idx, my_seq_head_idx % num_heads, 0, context_lens[seq_idx]);
      atomic_store_global_u32<memory_order::Relaxed>(&ws->steal_barrier_, 1);
      return my_seq_head_idx;
    });
    if (my_seq_head_idx == StatusAllWorkAssigned) {
      break;
    }
    __syncthreads();
    int seq_idx = my_seq_head_idx / num_heads;
    int head_idx = my_seq_head_idx % num_heads;
    int num_queries_per_kv = num_heads / num_kv_heads;
    int kv_head_idx = head_idx / (num_queries_per_kv / WorkStealingConfig::NumQueriesPerCta);  // broadcasting kv_head
    Task::attention(
        broadcast_buffer, &w, seq_idx, head_idx, kv_head_idx, out, ws->max_sum_,
        q, k_cache, v_cache, scalebias, page_table, context_lens, alibi_slopes,
        scale, num_seqs, num_heads, num_kv_heads, max_num_pages_per_seq, q_stride
    );
    if (threadIdx.x == (NumWarps - 1) * constant::WarpSize) {
      atomic_add_global_i32<memory_order::Relaxed>(&ws->num_busy_workers_, -1);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    while (!atomic_load_global_u32<memory_order::Relaxed>(&ws->steal_barrier_));
  }
  __syncthreads();
  TASK_DPRINTF1("worker[%d] enter stealing state...\n", my_worker_id);

  while (true) {
    constexpr int StatusAllWorkDone = -1;
    constexpr int StatusStealFailure = -2;

    // get worker_id to steal from
    int status_or_stolen_seq_head_idx = broadcast0<int>(broadcast_buffer, [&]() {
      if (atomic_load_global_u32<memory_order::Relaxed>((uint32_t*)&ws->num_busy_workers_) == 0) {
        return StatusAllWorkDone;
      }
      int victim_id = -1;
      if (!ws->work_queue_.dequeue(victim_id)) {
        return StatusStealFailure;
      }
      if (victim_id == my_worker_id) {
        return StatusStealFailure;
      }
      TASK_DPRINTF("??worker[%d]: tried to steal from worker[%d]\n", my_worker_id, victim_id);

      int stolen_seq_head_idx, stolen_chunk_start, stolen_chunk_end, remain_chunk_start, remain_chunk_end;
      auto& victim = ws->workers_[victim_id];
      stolen_seq_head_idx = victim.steal(stolen_chunk_start, stolen_chunk_end, remain_chunk_start, remain_chunk_end);
      if (stolen_seq_head_idx < 0) {
        return StatusStealFailure;
      }

      // the victim might be eligible being victim again, it is rich!
      if (remain_chunk_end - remain_chunk_start > 0) {
        ws->work_queue_.enqueue(victim_id);
      }
      TASK_DPRINTF(
          "!!worker[%d]: stole from worker[%d], seqhead:%d, chunk[%d,%d), remain chunk[%d,%d)\n",
          my_worker_id, victim_id, stolen_seq_head_idx, stolen_chunk_start, stolen_chunk_end, remain_chunk_start, remain_chunk_end
      );

      auto& w = ws->workers_[my_worker_id];
      w.initialize_chunk(stolen_seq_head_idx, stolen_chunk_start, stolen_chunk_end);
      atomic_add_global_i32<memory_order::Relaxed>(&ws->num_busy_workers_, 1);
      if (stolen_chunk_end - stolen_chunk_start > 1) {
        TASK_DPRINTF("$$worker[%d]: volunteered\n", my_worker_id);
        ws->work_queue_.enqueue(my_worker_id);  // rich now, volunteered to be stolen ==)
      }
      return stolen_seq_head_idx;
    });
    __syncthreads();

    int status = status_or_stolen_seq_head_idx;
    if (status == StatusAllWorkDone) {
      break;
    }
    if (status == StatusStealFailure) {
      continue;
    }

    int stolen_seq_head_idx = status_or_stolen_seq_head_idx;
    int seq_idx = stolen_seq_head_idx / num_heads;
    int head_idx = stolen_seq_head_idx % num_heads;
    int num_queries_per_kv = num_heads / num_kv_heads;
    int kv_head_idx = head_idx / (num_queries_per_kv / WorkStealingConfig::NumQueriesPerCta);  // broadcasting kv_head
    Task::attention(
        broadcast_buffer, &w, seq_idx, head_idx, kv_head_idx, out, ws->max_sum_,
        q, k_cache, v_cache, scalebias, page_table, context_lens, alibi_slopes,
        scale, num_seqs, num_heads, num_kv_heads, max_num_pages_per_seq, q_stride
    );
    if (threadIdx.x == (1 % NumWarps) * constant::WarpSize) {
      atomic_add_global_i32<memory_order::Relaxed>(&ws->num_busy_workers_, -1);
    }
    __syncthreads();
  }

  static_assert(WorkStealingConfig::InplaceFlashAcc);
  TASK_DPRINTF1("worker[%d] exit\n", my_worker_id);
}
#else
    ;
#endif

}  // namespace onnxruntime::contrib::paged

#undef TASK_DPRINTF_ENABLE
#undef TASK_DPRINTF
#undef TASK_DPRINTF1
#undef my_worker_id
