// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/threadpool.h"
#include "core/common/common.h"

#include <cassert>

#ifdef USE_EIGEN_THREADPOOL
#if defined(_MSC_VER)
#pragma warning(disable : 4267)
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <unsupported/Eigen/CXX11/ThreadPool>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#else
#include "task_thread_pool.h"
#endif

namespace onnxruntime {

namespace concurrency {

#ifdef USE_EIGEN_THREADPOOL
class ThreadPool::Impl : public Eigen::ThreadPool {
 public:
  Impl(const std::string& name, int num_threads)
      : Eigen::ThreadPool(num_threads) {
    ORT_UNUSED_PARAMETER(name);
  }

  void ParallelFor(int64_t total, std::function<void(int64_t)> fn) {
    // TODO: Eigen supports a more efficient ThreadPoolDevice mechanism
    // We will simply rely on the work queue and stealing in the short term.
    Eigen::Barrier barrier(static_cast<unsigned int>(total - 1));
    std::function<void(int64_t)> handle_iteration = [&barrier, &fn](int64_t iteration) {
      fn(iteration);
      barrier.Notify();
    };

    for (int64_t id = 1; id < total; ++id) {
      Schedule([=, &handle_iteration]() { handle_iteration(id); });
    }

    fn(0);
    barrier.Wait();
  }

  void ParallelFor(int64_t total, int64_t unit_size, std::function<void(int64_t, int64_t)> fn) {
    // TODO: Eigen supports a more efficient ThreadPoolDevice mechanism
    // We will simply rely on the work queue and stealing in the short term.
    unsigned int spawnedThreads = (unsigned int)ceil((double)total / (double)unit_size) - 1;
    Eigen::Barrier barrier(spawnedThreads);
    std::function<void(int64_t, int64_t)> handle_iteration = [&barrier, &fn](int64_t first, int64_t last) {
      fn(first, last);
      barrier.Notify();
    };

    for (int64_t first = unit_size; first < total; first += unit_size) {
      int64_t last = (first + unit_size < total) ? first + unit_size : total;
      Schedule([=, &handle_iteration]() { handle_iteration(first, last); });
    }

    // TODO: Make this iteration the last thread to simplify the last calculation.
    fn(0, unit_size);
    barrier.Wait();
  }
};
#else
class ThreadPool::Impl : public TaskThreadPool {
 public:
  Impl(const std::string& name, int num_threads)
      : TaskThreadPool(num_threads) {
    ORT_UNUSED_PARAMETER(name);
  }

  void Schedule(std::function<void()> fn) {
    std::packaged_task<void()> task(fn);
    RunTask(std::move(task));
  }

  void ParallelFor(int64_t total, std::function<void(int64_t)> fn) {
#ifdef USE_OPENMP
#pragma omp parallel for
    for (int64_t id = 0; id < total; ++id) {
      fn(id);
    }
#else
    Eigen::Barrier barrier(static_cast<unsigned int>(total - 1));
    std::function<void(int64_t)> handle_iteration = [&barrier, &fn](int64_t iteration) {
      fn(iteration);
      barrier.Notify();
    };
    for (int64_t id = 1; id < total; ++id) {
      std::packaged_task<void()> task(std::bind(handle_iteration, id));
      RunTask(std::move(task));
    }
    fn(0);
    barrier.Wait();
#endif
  }

  void ParallelFor(int64_t total, int64_t unit_size, std::function<void(int64_t, int64_t)> fn) {
#ifdef USE_OPENMP
#pragma omp parallel for
    for (int64_t first = 0; first < total; first += unit_size) {
      int64_t last = (first + unit_size < total) ? first + unit_size : total;
      fn(first, last);
    }
#else
    unsigned int spawnedThreads = (unsigned int)ceil(total / unit_size) - 1;
    Eigen::Barrier barrier(spawnedThreads);
    std::function<void(int64_t, int64_t)> handle_iteration = [&barrier, &fn](int64_t first, int64_t last) {
      fn(first, last);
      barrier.Notify();
    };
    for (int64_t first = unit_size; first < total; first += unit_size) {
      int64_t last = (first + unit_size < total) ? first + unit_size : total;
      std::packaged_task<void()> task(std::bind(handle_iteration, first, last));
      RunTask(std::move(task));
    }
    fn(0, unit_size);
    barrier.Wait();
#endif
  }
};
#endif

//
// ThreadPool
//
ThreadPool::ThreadPool(const std::string& name, int num_threads)
    : impl_(std::make_unique<Impl>(name, num_threads)) {
}

void ThreadPool::Schedule(std::function<void()> fn) { impl_->Schedule(fn); }

void ThreadPool::ParallelFor(int64_t total, std::function<void(int64_t)> fn) {
  if (total <= 0) return;

  if (total == 1 || NumThreads() == 1) {
    fn(0);
    return;
  }

  impl_->ParallelFor(total, fn);
}

void ThreadPool::ParallelFor(int64_t total, int64_t unit_size, std::function<void(int64_t, int64_t)> fn) {
  if (total <= 0 || unit_size <= 0) return;

  if (total <= unit_size || NumThreads() == 1) {
    fn(0, total);
    return;
  }

  impl_->ParallelFor(total, unit_size, fn);
}

int64_t ThreadPool::CalculateShardSize(int64_t total, float complexity) {
  // TODO: Simple implementation with four quadrants
  // Complexity Low / Total Low => 1 (total for now)
  // Complexity Low / Total High => total/threads
  // Complexity High / Total Low => total
  // Complexity High / Total High => total/threads
  if (total <= NumThreads()) {
    return 1;
  }
  
  // Block larger iteration counts across threads
  return (int64_t)ceil((double)total / (double)NumThreads());
}

// void ThreadPool::SetStealPartitions(const std::vector<std::pair<unsigned, unsigned>>& partitions) {
//   impl_->SetStealPartitions(partitions);
// }

int ThreadPool::NumThreads() const { return impl_->NumThreads(); }

int ThreadPool::CurrentThreadId() const { return impl_->CurrentThreadId(); }

ThreadPool::~ThreadPool() {}

}  // namespace concurrency
}  // namespace onnxruntime
