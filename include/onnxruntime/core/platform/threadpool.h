// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <vector>
#include <functional>
#include <memory>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#if __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#pragma GCC diagnostic ignored "-Wunused-parameter"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4267)
#pragma warning(disable : 4127)
#endif
#include <unsupported/Eigen/CXX11/ThreadPool>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace onnxruntime {

namespace concurrency {

/**
 * Generic class for instantiating thread pools.
 * Don't put any object of this type into a global variable in a Win32 DLL.
 */
class ThreadPool {
 public:
  /*
  Initializes a thread pool given the current environment.
  */
  ThreadPool(const std::string& name, int num_threads);

  /*
  Enqueue a unit of work.
  */
  void Schedule(std::function<void()> fn);

  /*
  Schedule work in the interval [0, total).
  */
  void ParallelFor(int32_t total, std::function<void(int32_t)> fn);

  /*
  Schedule work in the interval [0, total), with calls split into (num_batches) batches.
  */
  void BatchParallelFor(int32_t total, std::function<void(int32_t)> fn, int32_t num_batches = 0);

  /*
  Schedule work in the interval [first, last].
  */
  void ParallelForRange(int64_t first, int64_t last, std::function<void(int64_t, int64_t)> fn);

  // This is not supported until the latest Eigen
  // void SetStealPartitions(const std::vector<std::pair<unsigned, unsigned>>& partitions);

  /**
  Tries to call the given function in parallel, with calls split into (num_batches) batches.
  **/
  template <typename F>
  inline static void TryBatchParallelFor(concurrency::ThreadPool* tp, int32_t total, F&& fn, int32_t num_batches = 0) {
    if (tp != nullptr) {
      if (num_batches <= 0) {
        num_batches = tp->NumThreads() + 1;
      }
      tp->BatchParallelFor(total, std::forward<F>(fn), num_batches);
    } else {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
      for (int32_t i = 0; i < total; ++i) {
        fn(i);
      }
    }
  }

  /**
  Tries to call the given function in parallel.
  **/
  template <typename F>
  inline static void TryParallelFor(concurrency::ThreadPool* tp, int32_t total, F&& fn) {
    if (tp != nullptr) {
      tp->ParallelFor(total, std::forward<F>(fn));
    } else {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
      for (int32_t i = 0; i < total; ++i) {
        fn(i);
      }
    }
  }

  int NumThreads() const;

  int CurrentThreadId() const;

  Eigen::ThreadPool& GetHandler() { return impl_; }

 private:
  Eigen::ThreadPool impl_;
};

}  // namespace concurrency
}  // namespace onnxruntime
