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

#include <functional>
#include <memory>

// This file use PIMPL to avoid having eigen headers here

namespace Eigen {
class Allocator;
class ThreadPoolInterface;
}  // namespace Eigen

namespace onnxruntime {

struct TensorOpCost {
  double bytes_loaded;
  double bytes_stored;
  double compute_cycles;
};

template <typename Environment>
class ThreadPoolTempl;

namespace concurrency {

class ExtendedThreadPoolInterface;
class LoopCounter;

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
  // Constructs a pool for running with with "degree_of_parallelism" threads with
  // specified "name". env->StartThread() is used to create individual threads
  // with the given ThreadOptions. If "low_latency_hint" is true the thread pool
  // implementation may use it as a hint that lower latency is preferred at the
  // cost of higher CPU usage, e.g. by letting one or more idle threads spin
  // wait. Conversely, if the threadpool is used to schedule high-latency
  // operations like I/O the hint should be set to false.
  //
  // REQUIRES: degree_of_parallelism > 0
  // The allocator parameter is only used for creating a Eigen::ThreadPoolDevice to be used with Eigen Tensor classes.
  ThreadPool(Env* env,
             const ThreadOptions& thread_options,
             const NAME_CHAR_TYPE* name,
             int degree_of_parallelism,
             bool low_latency_hint);

  // Waits until all scheduled work has finished and then destroy the
  // set of threads.
  ~ThreadPool();

  // Schedules fn() for execution in the pool of threads.  The function may run
  // synchronously if it cannot be enqueued.  This will occur if the thread pool's
  // degree-of-parallelism is 1, but it may also occur for implementation-dependent
  // reasons such as if queues used for buffering work are full.
  void Schedule(std::function<void()> fn);

  // Returns the number of shards used by ParallelForFixedBlockSizeScheduling
  // with these parameters.
  int NumShardsUsedByFixedBlockSizeScheduling(std::ptrdiff_t total,
                                              std::ptrdiff_t block_size) const;

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
    std::ptrdiff_t num_threads = concurrency::ThreadPool::DegreeOfParallelism(tp);
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
    std::ptrdiff_t num_threads = concurrency::ThreadPool::DegreeOfParallelism(tp);
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

  // Return the degree of parallelism that code should assume when using the thread pool.
  // This API takes into account if OpenMP is enabled/disabled, and if the thread pool ptr is
  // nullptr.  It decouples the degree of parallelism for use with the thread pool from
  // the implementation choice of whether this matches the number of threads created in
  // the pool.
  //
  // Currently, a loop with degree-of-parallelism N is supported by a pool of N-1 threads
  // working in combination with the thread initiating the loop.
  static int DegreeOfParallelism(const concurrency::ThreadPool* tp);

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
   *\param num_batches If it is zero, it will be replaced to the value of DegreeOfParallelism().
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
      num_batches = std::min<ptrdiff_t>(total, DegreeOfParallelism(tp));
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
  friend class LoopCounter;

  // Returns the number of threads created in the pool.  This may be different from the
  // value returned by DegreeOfParallelism to code using the pool.
  int NumThreads() const;

  // Returns current thread id between 0 and NumThreads() - 1, if called from a
  // thread in the pool. Returns -1 otherwise.
  int CurrentThreadId() const;

  // Run fn with up to n degree-of-parallelism enlisting the thread pool for
  // help.  The degree-of-parallelism includes the caller, and so if n==1
  // then the function will run directly in the caller.  The fork-join
  // synchronization is handled in the thread pool, and so any state captured
  // by fn() is safe from concurrent access once RunWithHelp returns.
  void RunInParallel(std::function<void()> fn, int n);

  // Divides the work represented by the range [0, total) into k shards.
  // Calls fn(i*block_size, (i+1)*block_size) from the ith shard (0 <= i < k).
  // Each shard may be executed on a different thread in parallel, depending on
  // the number of threads available in the pool.
  // When (i+1)*block_size > total, fn(i*block_size, total) is called instead.
  // Here, k = NumShardsUsedByFixedBlockSizeScheduling(total, block_size).
  // Requires 0 < block_size <= total.
  void ParallelForFixedBlockSizeScheduling(std::ptrdiff_t total, std::ptrdiff_t block_size,
                                           const std::function<void(std::ptrdiff_t, std::ptrdiff_t)>& fn);


  // Return whether or not the calling thread should run a loop of
  // num_iterations divided in chunks of block_size in parallel.  If not,
  // the caller should run the loop sequentially.
  bool ShouldParallelizeLoop(const std::ptrdiff_t num_iterations,
                             const std::ptrdiff_t block_size = 1) const;

  ThreadOptions thread_options_;

  // If a thread pool is created with degree_of_parallelism != 1 then an underlying
  // EigenThreadPool is used to create OS threads and handle work distribution to them.
  // If degree_of_parallelism == 1 then underlying_threadpool_ is left as nullptr
  // and parallel work is run directly by the caller.
  ExtendedThreadPoolInterface* underlying_threadpool_ = nullptr;

  // If used, underlying_threadpool_ is instantiated and owned by the ThreadPool.
  std::unique_ptr<ThreadPoolTempl<Env> > extended_eigen_threadpool_;
};

}  // namespace concurrency
}  // namespace onnxruntime
