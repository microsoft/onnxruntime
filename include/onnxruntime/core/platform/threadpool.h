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

/* Modifications Copyright (c) Microsoft. */

#pragma once
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include "core/common/common.h"
#include "core/platform/env.h"
#include "core/common/optional.h"
#include "core/platform/thread_pool_interface.h"
#include <functional>
#include <memory>

// This file use PIMPL to avoid having eigen headers here


namespace onnxruntime {

struct TensorOpCost {
  double bytes_loaded;
  double bytes_stored;
  double compute_cycles;
};

template <typename Environment>
class ThreadPoolTempl;
namespace concurrency {
class ThreadPool {
 public:
  // Scheduling strategies for ParallelFor. The strategy governs how the given
  // units of work are distributed among the available threads in the
  // threadpool.
  enum class SchedulingStrategy {
    // The Adaptive scheduling strategy adaptively chooses the shard sizes based
    // on the cost of each unit of work, and the cost model of the underlying
    // threadpool device.
    //
    // The 'cost_per_unit' is an estimate of the number of CPU cycles (or
    // nanoseconds if not CPU-bound) to complete a unit of work. Overestimating
    // creates too many shards and CPU time will be dominated by per-shard
    // overhead, such as Context creation. Underestimating may not fully make
    // use of the specified parallelism, and may also cause inefficiencies due
    // to load balancing issues and stragglers.
    kAdaptive,
    // The Fixed Block Size scheduling strategy shards the given units of work
    // into shards of fixed size. In case the total number of units is not
    // evenly divisible by 'block_size', at most one of the shards may be of
    // smaller size. The exact number of shards may be found by a call to
    // NumShardsUsedByFixedBlockSizeScheduling.
    //
    // Each shard may be executed on a different thread in parallel, depending
    // on the number of threads available in the pool. Note that when there
    // aren't enough threads in the pool to achieve full parallelism, function
    // calls will be automatically queued.
    kFixedBlockSize
  };

  // Contains additional parameters for either the Adaptive or the Fixed Block
  // Size scheduling strategy.
  class SchedulingParams {
   public:
    explicit SchedulingParams(SchedulingStrategy strategy, optional<int64_t> cost_per_unit,
                              optional<std::ptrdiff_t> block_size)
        : strategy_(strategy), cost_per_unit_(cost_per_unit), block_size_(block_size) {
    }

    SchedulingStrategy strategy() const {
      return strategy_;
    }
    optional<int64_t> cost_per_unit() const {
      return cost_per_unit_;
    }
    optional<std::ptrdiff_t> block_size() const {
      return block_size_;
    }

   private:
    // The underlying Scheduling Strategy for which this instance contains
    // additional parameters.
    SchedulingStrategy strategy_;

    // The estimated cost per unit of work in number of CPU cycles (or
    // nanoseconds if not CPU-bound). Only applicable for Adaptive scheduling
    // strategy.
    optional<int64_t> cost_per_unit_;

    // The block size of each shard. Only applicable for Fixed Block Size
    // scheduling strategy.
    optional<std::ptrdiff_t> block_size_;
  };
#ifdef _WIN32
  using NAME_CHAR_TYPE = wchar_t;
#else
  using NAME_CHAR_TYPE = char;
#endif
  // Constructs a pool that contains "num_threads" threads with specified
  // "name". env->StartThread() is used to create individual threads with the
  // given ThreadOptions. If "low_latency_hint" is true the thread pool
  // implementation may use it as a hint that lower latency is preferred at the
  // cost of higher CPU usage, e.g. by letting one or more idle threads spin
  // wait. Conversely, if the threadpool is used to schedule high-latency
  // operations like I/O the hint should be set to false.
  //
  // REQUIRES: num_threads > 0
  // The allocator parameter is only used for creating a Eigen::ThreadPoolDevice to be used with Eigen Tensor classes.
  ThreadPool(Env* env, const ThreadOptions& thread_options, const NAME_CHAR_TYPE* name, int num_threads,
             bool low_latency_hint);
  // Constructs a pool that wraps around the thread::ThreadPoolInterface
  // instance provided by the caller. Caller retains ownership of
  // `user_threadpool` and must ensure its lifetime is longer than the
  // ThreadPool instance.
  ThreadPool(ThreadPoolInterface* user_threadpool, bool transfer_ownership);

  // Waits until all scheduled work has finished and then destroy the
  // set of threads.
  ~ThreadPool();

  // Schedules fn() for execution in the pool of threads.
  void Schedule(std::function<void()> fn);

  // Returns the number of shards used by ParallelForFixedBlockSizeScheduling
  // with these parameters.
  int NumShardsUsedByFixedBlockSizeScheduling(std::ptrdiff_t total, std::ptrdiff_t block_size);

  // ParallelFor shards the "total" units of work assuming each unit of work
  // having roughly "cost_per_unit" cost, in cycles. Each unit of work is
  // indexed 0, 1, ..., total - 1. Each shard contains 1 or more units of work
  // and the total cost of each shard is roughly the same.
  //
  // "cost_per_unit" is an estimate of the number of CPU cycles (or nanoseconds
  // if not CPU-bound) to complete a unit of work. Overestimating creates too
  // many shards and CPU time will be dominated by per-shard overhead, such as
  // Context creation. Underestimating may not fully make use of the specified
  // parallelism, and may also cause inefficiencies due to load balancing
  // issues and stragglers.
  void ParallelFor(std::ptrdiff_t total, double cost_per_unit,
                   const std::function<void(std::ptrdiff_t first, std::ptrdiff_t last)>& fn);
  static void TryParallelFor(concurrency::ThreadPool* tp, std::ptrdiff_t total, double cost_per_unit,
                             const std::function<void(std::ptrdiff_t first, std::ptrdiff_t last)>& fn) {
    TryParallelFor(tp, total, TensorOpCost{0, 0, static_cast<double>(cost_per_unit)}, fn);
  }

  void ParallelFor(std::ptrdiff_t total, const TensorOpCost& cost_per_unit,
                   const std::function<void(std::ptrdiff_t first, std::ptrdiff_t)>& fn);

  static void TryParallelFor(concurrency::ThreadPool* tp, std::ptrdiff_t total, const TensorOpCost& cost_per_unit,
                             const std::function<void(std::ptrdiff_t first, std::ptrdiff_t last)>& fn) {
#ifdef _OPENMP
    ORT_UNUSED_PARAMETER(cost_per_unit);
    std::ptrdiff_t num_threads = concurrency::ThreadPool::NumThreads(tp);
    if (total < num_threads) {
      num_threads = total;
    }
#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < num_threads; i++) {
      auto work = PartitionWork(i, num_threads, total);
      fn(work.start, work.end);
    }
#else
    if (tp == nullptr) {
      fn(0, total);
      return;
    }
    tp->ParallelFor(total, cost_per_unit, fn);
#endif
  }

  // Similar to ParallelFor above, but takes the specified scheduling strategy
  // into account.
  void ParallelFor(std::ptrdiff_t total, const SchedulingParams& scheduling_params,
                   const std::function<void(std::ptrdiff_t, std::ptrdiff_t)>& fn);

  static void TryParallelFor(concurrency::ThreadPool* tp, std::ptrdiff_t total,
                             const SchedulingParams& scheduling_params,
                             const std::function<void(std::ptrdiff_t first, std::ptrdiff_t last)>& fn) {
#ifdef _OPENMP
    ORT_UNUSED_PARAMETER(scheduling_params);
    std::ptrdiff_t num_threads = concurrency::ThreadPool::NumThreads(tp);
    if (total < num_threads) {
      num_threads = total;
    }
#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < num_threads; i++) {
      auto work = PartitionWork(i, num_threads, total);
      fn(work.start, work.end);
    }
#else
    if (tp == nullptr) {
      fn(0, total);
      return;
    }
    tp->ParallelFor(total, scheduling_params, fn);
#endif
  }

  // Prefer using this API to get the number of threads unless you know what you're doing.
  // This API takes into account if openmp is enabled/disabled and if the thread pool ptr is nullptr.
  static int NumThreads(const concurrency::ThreadPool* tp);

  // Returns the number of threads in the pool. Preferably use the static version of this API instead.
  int NumThreads() const;


  // If ThreadPool implementation is compatible with ThreadPoolInterface,
  // returns a non-null pointer. The caller does not own the object the returned
  // pointer points to, and should not attempt to delete.
  ThreadPoolInterface* AsEigenThreadPool() const;

  // Directly schedule the 'total' tasks to the underlying threadpool, without
  // cutting them by halves
  void SimpleParallelFor(std::ptrdiff_t total, const std::function<void(std::ptrdiff_t)>& fn);

  inline static void TrySimpleParallelFor(ThreadPool* tp, std::ptrdiff_t total,
                                          const std::function<void(std::ptrdiff_t)>& fn) {
#ifdef _OPENMP
    ORT_UNUSED_PARAMETER(tp);
#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < total; ++i) {
      fn(i);
    }
#else
    if (tp != nullptr) {
      tp->SimpleParallelFor(total, fn);
    } else {
      for (std::ptrdiff_t i = 0; i < total; ++i) {
        // In many cases, fn can be inlined here.
        fn(i);
      }
    }
#endif
  }

  /**
   * Tries to call the given function in parallel, with calls split into (num_batches) batches.
   *\param num_batches If it is zero, it will be replaced to the value of NumThreads().
   *\param fn A std::function or STL style functor with signature of "void f(int32_t);"
   * Pitfall: Caller should cap `num_batches` to a reasonable value based on the cost of `fn` and the value of `total`.
   *For example, if fn is as simple as: int sum=0; fn = [&](int i){sum +=i;} and `total` is 100, then num_batches should
   *be just 1.
   *
   * ```
   **/
  template <typename F>
  inline static void TryBatchParallelFor(ThreadPool* tp, std::ptrdiff_t total, F&& fn, std::ptrdiff_t num_batches) {
#ifdef _OPENMP
    ORT_UNUSED_PARAMETER(tp);
    ORT_UNUSED_PARAMETER(num_batches);
#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < total; ++i) {
      fn(i);
    }
#else
    if (tp == nullptr) {
      for (std::ptrdiff_t i = 0; i < total; ++i) {
        // In many cases, fn can be inlined here.
        fn(i);
      }
      return;
    }
    if (total <= 0)
      return;

    if (total == 1) {
      fn(0);
      return;
    }

    if (num_batches <= 0) {
      num_batches = std::min<ptrdiff_t>(total, tp->NumThreads());
    }

    if (num_batches <= 1) {
      for (int i = 0; i < total; i++) {
        fn(i);
      }
      return;
    }

    tp->SimpleParallelFor(num_batches, [&](std::ptrdiff_t batch_index) {
      auto work = PartitionWork(batch_index, num_batches, total);
      for (std::ptrdiff_t i = work.start; i < work.end; i++) {
        fn(i);
      }
    });
#endif
  }

  struct WorkInfo {
    std::ptrdiff_t start;
    std::ptrdiff_t end;
  };

  /** Calculate the start and end offsets for a batch.
      @remarks Based on MlasPartitionWork
  */
  static WorkInfo PartitionWork(std::ptrdiff_t batch_idx, std::ptrdiff_t num_batches, std::ptrdiff_t total_work) {
    const std::ptrdiff_t work_per_batch = total_work / num_batches;
    const std::ptrdiff_t work_per_batch_extra = total_work % num_batches;

    WorkInfo info;
    if (batch_idx < work_per_batch_extra) {
      info.start = (work_per_batch + 1) * batch_idx;
      info.end = info.start + work_per_batch + 1;
    } else {
      info.start = work_per_batch * batch_idx + work_per_batch_extra;
      info.end = info.start + work_per_batch;
    }

    return info;
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ThreadPool);

 private:
  // Divides the work represented by the range [0, total) into k shards.
  // Calls fn(i*block_size, (i+1)*block_size) from the ith shard (0 <= i < k).
  // Each shard may be executed on a different thread in parallel, depending on
  // the number of threads available in the pool.
  // When (i+1)*block_size > total, fn(i*block_size, total) is called instead.
  // Here, k = NumShardsUsedByFixedBlockSizeScheduling(total, block_size).
  // Requires 0 < block_size <= total.
  void ParallelForFixedBlockSizeScheduling(std::ptrdiff_t total, std::ptrdiff_t block_size,
                                           const std::function<void(std::ptrdiff_t, std::ptrdiff_t)>& fn);
  ThreadOptions thread_options_;
  // underlying_threadpool_ is the user_threadpool if user_threadpool is
  // provided in the constructor. Otherwise it is the eigen_threadpool_.
  ThreadPoolInterface* underlying_threadpool_;
  // eigen_threadpool_ is instantiated and owned by thread::ThreadPool if
  // user_threadpool is not in the constructor.
  std::unique_ptr<ThreadPoolTempl<Env> > eigen_threadpool_;
  bool owns_underlying_threadpool_ = false;
};

}  // namespace concurrency
}  // namespace onnxruntime
