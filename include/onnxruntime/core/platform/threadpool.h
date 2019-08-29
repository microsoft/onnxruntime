// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <vector>
#include <functional>
#include <memory>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#else
#pragma warning(push)
#pragma warning(disable : 4267)
#endif
#include <unsupported/Eigen/CXX11/ThreadPool>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#else
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
  Schedule work in the interval [first, last].
  */
  void ParallelForRange(int64_t first, int64_t last, std::function<void(int64_t, int64_t)> fn);

  // This is not supported until the latest Eigen
  // void SetStealPartitions(const std::vector<std::pair<unsigned, unsigned>>& partitions);

  int NumThreads() const;

  int CurrentThreadId() const;

  Eigen::ThreadPool& GetHandler() { return impl_; }

 private:
  Eigen::ThreadPool impl_;
};

}  // namespace concurrency
}  // namespace onnxruntime
