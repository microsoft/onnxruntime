
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
namespace {
class BlockingCounter {
 public:
  BlockingCounter(int initial_count) : state_(initial_count << 1), notified_(false) {
    ORT_ENFORCE(initial_count >= 0);
#ifndef NDEBUG
    ORT_ENFORCE(((initial_count << 1) >> 1) == initial_count);
#endif
  }

  ~BlockingCounter() = default;

  inline void DecrementCount() {
    unsigned int v = state_.fetch_sub(2, std::memory_order_acq_rel) - 2;
    if (v != 1) {
#ifndef NDEBUG
      ORT_ENFORCE(((v + 2) & ~1) != 0);
#endif
      return;  // either count has not dropped to 0, or waiter is not waiting
    }
    std::lock_guard<OrtMutex> l(mu_);
    notified_ = true;
    cond_var_.notify_all();
  }

  inline void Wait() {
    unsigned int v = state_.fetch_or(1, std::memory_order_acq_rel);
    if ((v >> 1) == 0)
      return;
    std::unique_lock<OrtMutex> l(mu_);
    while (!notified_) {
      cond_var_.wait(l);
    }
  }
  // Wait for the specified time, return false iff the count has not dropped to
  // zero before the timeout expired.
  inline bool WaitFor(std::chrono::milliseconds ms) {
    unsigned int v = state_.fetch_or(1, std::memory_order_acq_rel);
    if ((v >> 1) == 0)
      return true;
    std::unique_lock<OrtMutex> l(mu_);
    while (!notified_) {
      const std::cv_status status = cond_var_.wait_for(l, ms);
      if (status == std::cv_status::timeout) {
        return false;
      }
    }
    return true;
  }

 private:
  OrtMutex mu_;
  OrtCondVar cond_var_;
  std::atomic<int> state_;  // low bit is waiter flag
  bool notified_;
};
}  // namespace
namespace concurrency {

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options, const NAME_CHAR_TYPE* name, int num_threads,
                       bool low_latency_hint)
    : thread_options_(thread_options) {
  ORT_ENFORCE(num_threads >= 1);
  eigen_threadpool_ =
      onnxruntime::make_unique<ThreadPoolTempl<Env>>(name, num_threads, low_latency_hint, *env, thread_options_);
  underlying_threadpool_ = eigen_threadpool_.get();
}

ThreadPool::ThreadPool(ThreadPoolInterface* user_threadpool,bool transfer_ownership)
    : thread_options_(ThreadOptions()), owns_underlying_threadpool_(transfer_ownership) {
  underlying_threadpool_ = user_threadpool;
}

ThreadPool::~ThreadPool() {
  if(owns_underlying_threadpool_)
     delete underlying_threadpool_;
}

void ThreadPool::SimpleParallelFor(std::ptrdiff_t total, const std::function<void(std::ptrdiff_t)>& fn) {
  if (total <= 0)
    return;

  if (total == 1) {
    fn(0);
    return;
  }

  Barrier barrier(static_cast<unsigned int>(total));
  std::function<void(std::ptrdiff_t)> handle_iteration = [&barrier, &fn](std::ptrdiff_t iteration) {
    fn(iteration);
    barrier.Notify();
  };

  for (std::ptrdiff_t id = 0; id < total; ++id) {
    Schedule([=, &handle_iteration]() { handle_iteration(id); });
  }

  barrier.Wait();
}

void ThreadPool::Schedule(std::function<void()> fn) {
  ORT_ENFORCE(fn != nullptr);
  underlying_threadpool_->Schedule(std::move(fn));
}

int ThreadPool::NumShardsUsedByFixedBlockSizeScheduling(const std::ptrdiff_t total, const std::ptrdiff_t block_size) {
  if (block_size <= 0 || total <= 1 || total <= block_size || NumThreads() == 1) {
    return 1;
  }
  // TODO:check overflow?
  return static_cast<int>((total + block_size - 1) / block_size);
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

// This functionality is similar to parallelFor, except that reasoning about
// the number of shards used is significantly easier.
void ThreadPool::ParallelForFixedBlockSizeScheduling(const std::ptrdiff_t total, const std::ptrdiff_t block_size,
                                                     const std::function<void(std::ptrdiff_t, std::ptrdiff_t)>& fn) {
  const int num_shards_used = NumShardsUsedByFixedBlockSizeScheduling(total, block_size);
  if (num_shards_used == 1) {
    fn(0, total);
    return;
  }

  // Adapted from Eigen's parallelFor implementation.
  BlockingCounter counter(num_shards_used);
  std::function<void(ptrdiff_t, ptrdiff_t)> handle_range = [=, &handle_range, &counter, &fn](std::ptrdiff_t first,
                                                                                             std::ptrdiff_t last) {
    while (last - first > block_size) {
      // Find something near the midpoint which is a multiple of block size.
      const std::ptrdiff_t mid = first + ((last - first) / 2 + block_size - 1) / block_size * block_size;
      Schedule([=, &handle_range]() { handle_range(mid, last); });
      last = mid;
    }
    // Single block or less, execute directly.
    fn(first, last);
    counter.DecrementCount();  // The shard is done.
  };

  // Execute the root in the thread pool to avoid running work on more than
  // numThreads() threads.
  Schedule([=, &handle_range]() { handle_range(0, total); });
  counter.Wait();
}

struct ParallelForBlock {
  ptrdiff_t size;   // block size
  ptrdiff_t count;  // number of blocks
};
using CostModel = Eigen::TensorCostModel<Eigen::ThreadPoolDevice>;

// Calculates block size based on (1) the iteration cost and (2) parallel
// efficiency. We want blocks to be not too small to mitigate parallelization
// overheads; not too large to mitigate tail effect and potential load
// imbalance and we also want number of blocks to be evenly dividable across
// threads.
static ParallelForBlock CalculateParallelForBlock(const ptrdiff_t n, const Eigen::TensorOpCost& cost,
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
      block_count = coarser_block_count;
      if (max_efficiency < coarser_efficiency) {
        max_efficiency = coarser_efficiency;
      }
    }
  }

  return {block_size, block_count};
}

void ThreadPool::ParallelFor(std::ptrdiff_t n, const TensorOpCost& c,
                             const std::function<void(std::ptrdiff_t first, std::ptrdiff_t)>& f) {
  ORT_ENFORCE(n >= 0);
  Eigen::TensorOpCost cost{c.bytes_loaded, c.bytes_stored, c.compute_cycles};
  // Compute small problems directly in the caller thread.
  if (n <= 1 || NumThreads() == 1 ||
      Eigen::TensorCostModel<Eigen::ThreadPoolDevice>::numThreads(static_cast<double>(n), cost, static_cast<int>(NumThreads())) == 1) {
    f(0, n);
    return;
  }

  // Compute block size and total count of blocks.
  ParallelForBlock block = CalculateParallelForBlock(n, cost, nullptr, NumThreads());

  // Recursively divide size into halves until we reach block_size.
  // Division code rounds mid to block_size, so we are guaranteed to get
  // block_count leaves that do actual computations.
  Barrier barrier(static_cast<unsigned int>(block.count));
  std::function<void(ptrdiff_t, ptrdiff_t)> handleRange;
  handleRange = [=, &handleRange, &barrier, &f](ptrdiff_t firstIdx, ptrdiff_t lastIdx) {
    while (lastIdx - firstIdx > block.size) {
      // Split into halves and schedule the second half on a different thread.
      const ptrdiff_t midIdx = firstIdx + Eigen::divup((lastIdx - firstIdx) / 2, block.size) * block.size;
      underlying_threadpool_->Schedule([=, &handleRange]() { handleRange(midIdx, lastIdx); });
      lastIdx = midIdx;
    }
    // Single block or less, execute directly.
    f(firstIdx, lastIdx);
    barrier.Notify();
  };

  underlying_threadpool_->Schedule([=, &handleRange]() { handleRange(0, n); });
  barrier.Wait();
}
void ThreadPool::ParallelFor(std::ptrdiff_t total, double cost_per_unit,
                             const std::function<void(std::ptrdiff_t first, std::ptrdiff_t)>& fn) {
  ParallelFor(total, TensorOpCost{0, 0, static_cast<double>(cost_per_unit)}, fn);
}

int ThreadPool::NumThreads(const concurrency::ThreadPool* tp) {
#ifdef _OPENMP
  ORT_UNUSED_PARAMETER(tp);
  return (omp_get_num_threads() == 1) ? omp_get_max_threads() : 1;
#else
  return tp ? tp->NumThreads() : 1;
#endif
}

int ThreadPool::NumThreads() const {
  return underlying_threadpool_->NumThreads();
}



ThreadPoolInterface* ThreadPool::AsEigenThreadPool() const {
  ORT_ENFORCE(underlying_threadpool_ != nullptr);
  return underlying_threadpool_;
}
}  // namespace concurrency
}  // namespace onnxruntime
