/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>

#include "core/platform/threadpool.h"
#include "core/common/common.h"
#include "core/common/eigen_common_wrapper.h"
#include "core/platform/EigenNonBlockingThreadPool.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {

namespace concurrency {

// A sharded loop counter distributes loop iterations between a set of worker threads.  The iteration space of
// the loop is divided (perhaps unevenly) between the shards.  Each thread has a home shard (perhaps not uniquely
// to it), and it claims iterations via atomic operations on its home shard.  It then proceeds through the other
// shards until all of the shards' iterations are complete.  This approach serves to purposes.  First, compared
// with atomic operations on a single counter, it reduces contention on a single counter in the case of loops with
// large numbers of short-running iteration.  Second, by having a thread work on its home shard initially, it
// promotes affinity between the work that a thread performs in one loop and the work that it performs in the next.

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324) /* Padding added to LoopCounterShard, LoopCounter for alignment */
#endif

static constexpr int CACHE_LINE_BYTES = 64;
static constexpr int NUM_SHARDS = 8;

struct alignas(CACHE_LINE_BYTES) LoopCounterShard {
  ::std::atomic<uint64_t> _next;
  uint64_t _end;
};

class alignas(CACHE_LINE_BYTES) LoopCounter {
public:
 LoopCounter(const ThreadPool& tp,
             uint64_t num_iterations,
             uint64_t block_size = 1) : _tp(tp),
                                        _block_size(block_size) {
   assert(sizeof(LoopCounterShard) == 64);
   assert(block_size != 0);

   // Divide the iteration space into NUM_SHARDS pieces.  If the iteration space does not
   // divide evenly into shards of multiples of block_size then the final shard is left uneven.
   double iterations_per_shard = static_cast<double>(num_iterations) / NUM_SHARDS;
   uint64_t split = 0;
   for (uint64_t shard = 0; shard < NUM_SHARDS; shard++) {
     _shards[shard]._next = split;
     split = (static_cast<uint64_t>((shard + 1) * iterations_per_shard) / block_size) * block_size;
     _shards[shard]._end = split;
   }

   // Ensure that the final shard finishes precisely at the end of the iteration space
   _shards[NUM_SHARDS - 1]._end = num_iterations;
 }

  int GetHomeShard() const {
    // Allocate each thread to a home shard, from which it starts claiming iterations.  The allocation
    // does not need to be unique, but we aim for a good distribution, particularly in the case where
    // most/all of the thread pool's threads are active in the loop.  Threads outside the pool may
    // also be claiming work, with CurrentThreadId -1.
    int d_of_p = ThreadPool::DegreeOfParallelism(&_tp);
    int my_thread_idx = (_tp.CurrentThreadId() + 1) % d_of_p;
    assert(my_thread_idx >= 0 && my_thread_idx < d_of_p);

    int home_shard;
    if (d_of_p >= NUM_SHARDS) {
      // More threads than shards => allocate them home shards round-robin, aiming to sprace the load across
      // the shards
      home_shard = my_thread_idx % NUM_SHARDS;
    } else {
      // Fewer threads than shards => spread the threads evenly across the shards, so each will work
      // on a run of successive shards before contention
      home_shard = (my_thread_idx * NUM_SHARDS) / d_of_p;
    }
    assert(home_shard >= 0 && home_shard < NUM_SHARDS);
    return home_shard;
  }

  // Attempt to claim iterations from the sharded counter.  The function either
  // returns true, along with a block of exactly block_size iterations, or it returns false
  // if all of the iterations have been claimed.
  bool ClaimIterations(int my_home_shard,
                       int& my_shard,
                       uint64_t& my_start,
                       uint64_t& my_end) {
    do {
      if (_shards[my_shard]._next < _shards[my_shard]._end) {
        // Appears to be work in the current shard, try to claim with atomic fetch-and-add
        uint64_t temp_start = _shards[my_shard]._next.fetch_add(_block_size);
        if (temp_start < _shards[my_shard]._end) {
          my_start = temp_start;
          my_end = std::min(_shards[my_shard]._end, temp_start + _block_size);
          return true;
        }
      }
      // Work in the current shard is exhausted, move to the next shard, until
      // we are back at the home shard.
      my_shard = (my_shard + 1) % NUM_SHARDS;
    } while (my_shard != my_home_shard);
    return false;
  }

private:
  alignas(CACHE_LINE_BYTES) LoopCounterShard _shards[NUM_SHARDS];
  const ThreadPool& _tp;
  const uint64_t _block_size;
};

#ifdef _MSC_VER
#pragma warning(pop) /* Padding added in LoopCounterShard, LoopCounter */
#endif

ThreadPool::ThreadPool(Env* env,
                       const ThreadOptions& thread_options,
                       const NAME_CHAR_TYPE* name,
                       int degree_of_parallelism,
                       bool low_latency_hint)
    : thread_options_(thread_options) {
  // In the current implementation, a thread pool with degree_of_parallelism==1 uses
  // the caller as one of the threads for executing work.  Hence we only create
  // additional thread(s) for degree_of_parallelism>=2.
  ORT_ENFORCE(degree_of_parallelism >= 1);
  if (degree_of_parallelism >= 2) {
    int threads_to_create = degree_of_parallelism - 1;
    extended_eigen_threadpool_ =
        onnxruntime::make_unique<ThreadPoolTempl<Env>>(name,
                                                       threads_to_create,
                                                       low_latency_hint,
                                                       *env,
                                                       thread_options_);
    underlying_threadpool_ = extended_eigen_threadpool_.get();
  }
}

ThreadPool::~ThreadPool() = default;

// Base case for parallel loops, running iterations 0..total, divided into blocks
// of block_size iterations, and calling into a function that takes a start..end
// range of indices to run.
void ThreadPool::ParallelForFixedBlockSizeScheduling(const std::ptrdiff_t total,
                                                     const std::ptrdiff_t block_size,
                                                     const std::function<void(std::ptrdiff_t, std::ptrdiff_t)>& fn) {
  if (total <= 0)
    return;

  if (total <= block_size) {
    fn(0, total);
    return;
  }

  // Split the work across threads in the pool.  Each work item will run a loop claiming iterations,
  // hence we need at most one for each thread, even if the numberof blocks of iterations is larger.
  auto d_of_p = DegreeOfParallelism(this);
  int num_work_items = static_cast<int>(std::min(static_cast<std::ptrdiff_t>(d_of_p), total));
  assert(num_work_items > 0);

  LoopCounter lc(*this, total, block_size);
  std::function<void()> run_work = [&]() {
    int my_home_shard = lc.GetHomeShard();
    int my_shard = my_home_shard;
    uint64_t my_iter_start, my_iter_end;
    while (lc.ClaimIterations(my_home_shard, my_shard, my_iter_start, my_iter_end)) {
      fn(static_cast<std::ptrdiff_t>(my_iter_start),
         static_cast<std::ptrdiff_t>(my_iter_end));
    }
  };

  // Run the work in the thread pool (and in the current thread).  Synchronization with helping
  // threads is handled within RunInParallel, hence we can deallocate lc and other state captured by
  // run_work.
  RunInParallel(run_work, num_work_items);
}

void ThreadPool::SimpleParallelFor(std::ptrdiff_t total, const std::function<void(std::ptrdiff_t)>& fn) {
  ParallelForFixedBlockSizeScheduling(total, 1, [&](std::ptrdiff_t first, std::ptrdiff_t last) {
    for (std::ptrdiff_t idx = first; idx < last; idx++) {
      fn(idx);
    }
  });
}

void ThreadPool::Schedule(std::function<void()> fn) {
  ORT_ENFORCE(fn != nullptr);
  if (underlying_threadpool_) {
    underlying_threadpool_->Schedule(std::move(fn));
  } else {
    fn();
  }
}

void ThreadPool::RunInParallel(std::function<void()> fn, int n) {
  ORT_ENFORCE(fn != nullptr);
  if (underlying_threadpool_) {
    underlying_threadpool_->RunInParallel(std::move(fn), n);
  } else {
    fn();
  }
}

bool ThreadPool::ShouldParallelizeLoop(const std::ptrdiff_t num_iterations,
                                       const std::ptrdiff_t block_size) const {
  // Do not parallelize trivial loops, with only a single block of work
  if (block_size <= 0 || num_iterations <= block_size) {
    return false;
  }

  // Do not parallelize loops with only a single thread available.  If the
  // caller is outside the current pool (ID == -1) then we parallelize
  // if the pool has any threads.  If the caller is inside the current pool
  // (ID != -1) then we require at least one additional thread in the pool.
  if ((CurrentThreadId() == -1 && NumThreads() == 0) ||
      (CurrentThreadId() != -1 && NumThreads() == 1)) {
    return false;
  }

  return true;
}

int ThreadPool::NumShardsUsedByFixedBlockSizeScheduling(const std::ptrdiff_t total,
                                                        const std::ptrdiff_t block_size) const {
  if (!ShouldParallelizeLoop(total, block_size)) {
    return 1;
  } else {
    // TODO:check overflow?
    return static_cast<int>((total + block_size - 1) / block_size);
  }
}

void ThreadPool::ParallelFor(std::ptrdiff_t total, const SchedulingParams& scheduling_params,
                             const std::function<void(std::ptrdiff_t, std::ptrdiff_t)>& fn) {
  switch (scheduling_params.strategy()) {
    case SchedulingStrategy::kAdaptive: {
      if (scheduling_params.cost_per_unit().has_value()) {
        ParallelFor(total, static_cast<double>(scheduling_params.cost_per_unit().value()), fn);
      }
      break;
    }
    case SchedulingStrategy::kFixedBlockSize: {
      if (scheduling_params.block_size().has_value()) {
        ParallelForFixedBlockSizeScheduling(total, scheduling_params.block_size().value(), fn);
      }
      break;
    }
  }
}

using CostModel = Eigen::TensorCostModel<Eigen::ThreadPoolDevice>;

// Calculates block size based on (1) the iteration cost and (2) parallel
// efficiency. We want blocks to be not too small to mitigate parallelization
// overheads; not too large to mitigate tail effect and potential load
// imbalance and we also want number of blocks to be evenly dividable across
// threads.
static ptrdiff_t CalculateParallelForBlock(const ptrdiff_t n, const Eigen::TensorOpCost& cost,
                                           std::function<ptrdiff_t(ptrdiff_t)> block_align, int num_threads) {
  const double block_size_f = 1.0 / CostModel::taskSize(1, cost);
  const ptrdiff_t max_oversharding_factor = 4;
  ptrdiff_t block_size = Eigen::numext::mini(
      n,
      Eigen::numext::maxi<ptrdiff_t>(Eigen::divup<ptrdiff_t>(n, max_oversharding_factor * num_threads), static_cast<ptrdiff_t>(block_size_f)));
  const ptrdiff_t max_block_size = Eigen::numext::mini(n, 2 * block_size);

  if (block_align) {
    ptrdiff_t new_block_size = block_align(block_size);
    assert(new_block_size >= block_size);
    block_size = Eigen::numext::mini(n, new_block_size);
  }

  ptrdiff_t block_count = Eigen::divup(n, block_size);

  // Calculate parallel efficiency as fraction of total CPU time used for
  // computations:
  double max_efficiency =
      static_cast<double>(block_count) / (Eigen::divup<ptrdiff_t>(block_count, num_threads) * num_threads);

  // Now try to increase block size up to max_block_size as long as it
  // doesn't decrease parallel efficiency.
  for (ptrdiff_t prev_block_count = block_count; max_efficiency < 1.0 && prev_block_count > 1;) {
    // This is the next block size that divides size into a smaller number
    // of blocks than the current block_size.
    ptrdiff_t coarser_block_size = Eigen::divup(n, prev_block_count - 1);
    if (block_align) {
      ptrdiff_t new_block_size = block_align(coarser_block_size);
      assert(new_block_size >= coarser_block_size);
      coarser_block_size = Eigen::numext::mini(n, new_block_size);
    }
    if (coarser_block_size > max_block_size) {
      break;  // Reached max block size. Stop.
    }
    // Recalculate parallel efficiency.
    const ptrdiff_t coarser_block_count = Eigen::divup(n, coarser_block_size);
    assert(coarser_block_count < prev_block_count);
    prev_block_count = coarser_block_count;
    const double coarser_efficiency =
        static_cast<double>(coarser_block_count) / (Eigen::divup<ptrdiff_t>(coarser_block_count, num_threads) * num_threads);
    if (coarser_efficiency + 0.01 >= max_efficiency) {
      // Taking it.
      block_size = coarser_block_size;
      if (max_efficiency < coarser_efficiency) {
        max_efficiency = coarser_efficiency;
      }
    }
  }

  return block_size;
}

void ThreadPool::ParallelFor(std::ptrdiff_t n, const TensorOpCost& c,
                             const std::function<void(std::ptrdiff_t first, std::ptrdiff_t)>& f) {
  ORT_ENFORCE(n >= 0);
  Eigen::TensorOpCost cost{c.bytes_loaded, c.bytes_stored, c.compute_cycles};
  auto d_of_p = DegreeOfParallelism(this);
  // Compute small problems directly in the caller thread.
  if ((!ShouldParallelizeLoop(n)) ||
      Eigen::TensorCostModel<Eigen::ThreadPoolDevice>::numThreads(static_cast<double>(n),
                                                                  cost,
                                                                  d_of_p) == 1) {
    f(0, n);
    return;
  }

  ptrdiff_t block = CalculateParallelForBlock(n, cost, nullptr, d_of_p);
  ParallelForFixedBlockSizeScheduling(n, block, f);
}

void ThreadPool::ParallelFor(std::ptrdiff_t total, double cost_per_unit,
                             const std::function<void(std::ptrdiff_t first, std::ptrdiff_t)>& fn) {
  ParallelFor(total, TensorOpCost{0, 0, static_cast<double>(cost_per_unit)}, fn);
}

int ThreadPool::DegreeOfParallelism(const concurrency::ThreadPool* tp) {
#ifdef _OPENMP
  // When using OpenMP, omp_get_num_threads() returns the number of threads in the
  // current parallel region.  Hence if this is 1 then we aim to parallelise
  // across the number of threads configured.  Otherwise, given that we do not
  // use nested parallelism, we do not parallelise further.
  ORT_UNUSED_PARAMETER(tp);
  return (omp_get_num_threads() == 1) ? omp_get_max_threads() : 1;
#else
  // When not using OpenMP, we parallelise over the N threads created by the pool
  // tp, plus 1 for the thread entering a loop.
  return tp ? (tp->NumThreads()+1) : 1;
#endif
}

// Return the number of threads created by the pool.
int ThreadPool::NumThreads() const {
  if (underlying_threadpool_) {
    return underlying_threadpool_->NumThreads();
  } else {
    return 0;
  }
}

// Return ID of the current thread within this pool.  Returns -1 for a thread outside the
// current pool.
int ThreadPool::CurrentThreadId() const {
  if (underlying_threadpool_) {
    return underlying_threadpool_->CurrentThreadId();
  } else {
    return -1;
  }
}

}  // namespace concurrency
}  // namespace onnxruntime
