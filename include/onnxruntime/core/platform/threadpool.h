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

#include <functional>
#include <memory>

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


namespace concurrency {

template <typename Environment>
class ThreadPoolTempl;

class ExtendedThreadPoolInterface;
class LoopCounter;
class ThreadPoolParallelSection;

#ifdef _WIN32
using NAME_CHAR_TYPE = wchar_t;
#else
using NAME_CHAR_TYPE = char;
#endif
/*
class ThreadPool {
 public:

  ThreadPool(Env* env,
             const ThreadOptions& thread_options,
             const NAME_CHAR_TYPE* name,
             int degree_of_parallelism,
             bool low_latency_hint);
  ~ThreadPool();

  class ParallelSection {
  public:
    explicit ParallelSection(ThreadPool *tp);
    ~ParallelSection();

  private:
    friend class ThreadPool;

    std::unique_ptr<ThreadPoolParallelSection, void(*)(ThreadPoolParallelSection*)>
      ps_{nullptr, [](ThreadPoolParallelSection*){}};
#ifndef _OPENMP
    ThreadPool *tp_;
#endif
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ParallelSection);

    static thread_local ParallelSection *current_parallel_section;
    static_assert(std::is_trivially_destructible<decltype(current_parallel_section)>::value,
                  "Per-thread state should be trivially destructible");
  };

  static void Schedule(ThreadPool* tp,
                       std::function<void()> fn) {
    if (tp) {
      tp->Schedule(fn);
    } else {
      fn();
    }
  }

  static void TryParallelFor(ThreadPool* tp, std::ptrdiff_t total, double cost_per_unit,
                             const std::function<void(std::ptrdiff_t first, std::ptrdiff_t last)>& fn) {
    TryParallelFor(tp, total, TensorOpCost{0, 0, static_cast<double>(cost_per_unit)}, fn);
  }

  static void TryParallelFor(ThreadPool* tp, std::ptrdiff_t total, const TensorOpCost& cost_per_unit,
                             const std::function<void(std::ptrdiff_t first, std::ptrdiff_t last)>& fn);

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
        fn(i);
      }
    }
#endif
  }

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

  static bool ShouldParallelize(const ThreadPool* tp);

  static int DegreeOfParallelism(const ThreadPool* tp);

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ThreadPool);

 private:
  friend class LoopCounter;
  int NumThreads() const;
  int CurrentThreadId() const;
  void RunInParallel(std::function<void(unsigned idx)> fn, unsigned n);
  void ParallelForFixedBlockSizeScheduling(std::ptrdiff_t total, std::ptrdiff_t block_size,
                                           const std::function<void(std::ptrdiff_t, std::ptrdiff_t)>& fn);
  bool ShouldParallelizeLoop(const std::ptrdiff_t num_iterations,
                             const std::ptrdiff_t block_size = 1) const;
  void ParallelFor(std::ptrdiff_t total, double cost_per_unit,
                   const std::function<void(std::ptrdiff_t first, std::ptrdiff_t last)>& fn);
  void ParallelFor(std::ptrdiff_t total, const TensorOpCost& cost_per_unit,
                   const std::function<void(std::ptrdiff_t first, std::ptrdiff_t)>& fn);
  void SimpleParallelFor(std::ptrdiff_t total, const std::function<void(std::ptrdiff_t)>& fn);
  void Schedule(std::function<void()> fn);
  ThreadOptions thread_options_;
  ExtendedThreadPoolInterface* underlying_threadpool_ = nullptr;
  std::unique_ptr<ThreadPoolTempl<Env> > extended_eigen_threadpool_;
};*/

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct ThreadPoolImpl;

class ThreadPool {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ThreadPool);
  ThreadPool(Env*,
        const ThreadOptions&,
        const NAME_CHAR_TYPE*,
        int num_threads,
        bool);
  ThreadPool(int num_threads);
  ~ThreadPool();

  class ParallelSection {
    public:
      explicit ParallelSection(ThreadPool*) {};
      ~ParallelSection() = default;
  };

  using Fn = ::std::function<void(::std::ptrdiff_t, ::std::ptrdiff_t)>;
  using SimpleFn = ::std::function<void(::std::ptrdiff_t)>;

  static void Schedule(ThreadPool* tp, ::std::function<void()> fn);
  static void TryParallelFor(ThreadPool* tp, ::std::ptrdiff_t total, double cost_per_unit, const Fn& fn);
  static void TryParallelFor(ThreadPool* tp, ::std::ptrdiff_t total, const TensorOpCost& cost_per_unit, const Fn& fn);
  static void TrySimpleParallelFor(ThreadPool* tp, ::std::ptrdiff_t total, const SimpleFn& fn);
  template <typename F>
  inline static void TryBatchParallelFor(ThreadPool* tp, ::std::ptrdiff_t total, F&& fn, ::std::ptrdiff_t num_batches) {
    if (tp) {
      if (total <= 0) {
        return;
      }
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
    } else {
      for (std::ptrdiff_t i = 0; i < total; ++i) {
        fn(i);
      }
    }
  }

  struct WorkInfo {
    ::std::ptrdiff_t start;
    ::std::ptrdiff_t end;
  };

  static WorkInfo PartitionWork(::std::ptrdiff_t batch_idx, ::std::ptrdiff_t num_batches, ::std::ptrdiff_t total_work);
  static bool ShouldParallelize(const ThreadPool* tp);
  static int DegreeOfParallelism(const ThreadPool* tp);

 private:
  void Schedule(::std::function<void()> fn);
  void ParallelFor(::std::ptrdiff_t total, double cost_per_unit, const Fn& fn);
  void ParallelFor(::std::ptrdiff_t total, const TensorOpCost& cost_per_unit, const Fn& fn);
  void SimpleParallelFor(::std::ptrdiff_t total, const SimpleFn& fn);
  ThreadPoolImpl* threadPoolImpl_ = nullptr;
}; // class ThreadPool

}  // namespace concurrency
}  // namespace onnxruntime
