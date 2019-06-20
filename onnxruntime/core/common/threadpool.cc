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

#ifndef USE_OPENMP
namespace Eigen {

// TODO: This is temporarily taken from Eigen as the task based threadpool lacks support.
// Barrier is an object that allows one or more threads to wait until
// Notify has been called a specified number of times.
class Barrier {
 public:
  Barrier(unsigned int count) : state_(count << 1), notified_(false) {
    assert(((count << 1) >> 1) == count);
  }

  ~Barrier() {
    assert((state_ >> 1) == 0);
  }

  void Notify() {
    unsigned int v = state_.fetch_sub(2, std::memory_order_acq_rel) - 2;

    if (v != 1) {
      assert(((v + 2) & ~1) != 0);
      return;  // either count has not dropped to 0, or waiter is not waiting
    }

    std::unique_lock<std::mutex> l(mu_);
    assert(!notified_);
    notified_ = true;
    cv_.notify_all();
  }

  void Wait() {
    unsigned int v = state_.fetch_or(1, std::memory_order_acq_rel);

    if ((v >> 1) == 0) return;

    std::unique_lock<std::mutex> l(mu_);

    while (!notified_) {
      cv_.wait(l);
    }
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::atomic<unsigned int> state_;  // low bit is waiter flag
  bool notified_;
};

}
#endif
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
    int64_t remainder = total - unit_size * NumThreads();
    int64_t spawnedThreads = (unit_size == 1 ? total : NumThreads()) - 1;
    Eigen::Barrier barrier(static_cast<unsigned int>(spawnedThreads));
    std::function<void(int64_t, int64_t)> handle_iteration = [&barrier, &fn](int64_t first, int64_t last) {
      fn(first, last);
      barrier.Notify();
    };

    int64_t last = unit_size;
    for (int64_t iteration = 1; iteration <= spawnedThreads; iteration++) {
      int64_t first = last;
      last = first + unit_size + (remainder-- > 0 ? 1 : 0);
      Schedule([=, &handle_iteration]() { handle_iteration(first, last); });
    }

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
    int64_t remainder = total - unit_size * NumThreads();
    int64_t spawnedThreads = (unit_size == 1 ? total : NumThreads()) - 1;
    Eigen::Barrier barrier(static_cast<unsigned int>(spawnedThreads));
    std::function<void(int64_t, int64_t)> handle_iteration = [&barrier, &fn](int64_t first, int64_t last) {
      fn(first, last);
      barrier.Notify();
    };
    int64_t last = unit_size;
    for (int64_t iteration = 1; iteration <= spawnedThreads; iteration++) {
      int64_t first = last;
      last = first + unit_size + (remainder-- > 0 ? 1 : 0);
      Schedule([=, &handle_iteration]() { handle_iteration(first, last); });
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

  // If we have no need for threads, run sequentially
  if (total == 1 || NumThreads() == 1) {
    for (int i = 0; i < total; i++) {
      fn(0);
    }

    return;
  }

  impl_->ParallelFor(total, fn);
}

void ThreadPool::ParallelFor(int64_t total, int64_t unit_size, std::function<void(int64_t, int64_t)> fn) {
  if (total <= 0 || unit_size < 0) return;

  // If we have no need for threads, run one block
  if (total == 1 || NumThreads() == 1) {
    fn(0, total);
    return;
  }

  impl_->ParallelFor(total, unit_size, fn);
}

int64_t ThreadPool::CalculateShardSize(int64_t total, float complexity) {
  // TODO: Simplified calculation, ignores complexity.
  if (total <= NumThreads()) {
    return 1;
  }

  // Block larger iteration counts across threads
  return total / NumThreads();
}

// void ThreadPool::SetStealPartitions(const std::vector<std::pair<unsigned, unsigned>>& partitions) {
//   impl_->SetStealPartitions(partitions);
// }

int ThreadPool::NumThreads() const { return impl_->NumThreads(); }

int ThreadPool::CurrentThreadId() const { return impl_->CurrentThreadId(); }

ThreadPool::~ThreadPool() {}

}  // namespace concurrency
}  // namespace onnxruntime
