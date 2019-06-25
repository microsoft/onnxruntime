// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace onnxruntime {

namespace concurrency {

/**
 * Generic class for instantiating thread pools.
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
  void ParallelFor(int64_t total, std::function<void(int64_t)> fn);

  /*
  Schedule work in the interval [0, total) into shards of a unit_size.
  */
  void ParallelFor(int64_t total, int64_t unit_size, std::function<void(int64_t, int64_t)> fn);

  /*
  Estimate the number of shards to divide work into given the number of threads of the current
  threadpool, number of iterations to compute and approximate work complexity.
  */
  int64_t CalculateShardSize(int64_t total, float complexity = 0.0f) const;

  // This is not supported until the latest Eigen
  // void SetStealPartitions(const std::vector<std::pair<unsigned, unsigned>>& partitions);

  int NumThreads() const;

  int CurrentThreadId() const;

  /*
  Ensure that the pool has terminated and cleaned up all threads cleanly.
  */
  ~ThreadPool();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace concurrency
}  // namespace onnxruntime
