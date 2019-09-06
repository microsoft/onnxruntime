// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/threadpool.h"
#include "core/common/common.h"

#include <cassert>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#else
#pragma warning(push)
#pragma warning(disable : 4267)
#endif
#include <unsupported/Eigen/CXX11/src/ThreadPool/Barrier.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#else
#pragma warning(pop)
#endif

namespace onnxruntime {

namespace concurrency {
//
// ThreadPool
//
ThreadPool::ThreadPool(const std::string&, int num_threads) : impl_(num_threads) {}

void ThreadPool::Schedule(std::function<void()> fn) { impl_.Schedule(fn); }

void ThreadPool::ParallelFor(int32_t total, std::function<void(int32_t)> fn) {
  if (total <= 0) return;

  if (total == 1) {
    fn(0);
    return;
  }

  // TODO: Eigen supports a more efficient ThreadPoolDevice mechanism
  // We will simply rely on the work queue and stealing in the short term.
  EigenExtendedBarrier barrier(static_cast<unsigned int>(total));
  std::function<void(int32_t)> handle_iteration = [&barrier, &fn](int iteration) {
    fn(iteration);
    barrier.Notify();
  };

  for (int32_t id = 0; id < total; ++id) {
    Schedule([=, &handle_iteration]() { handle_iteration(id); });
  }

  if (impl_.CurrentThreadId() != -1) {
    // A thread in the threadpool can't wait or it will cause deadlocks.
    while (!barrier.Done()) {
      impl_.Help(impl_.CurrentThreadId());
      std::this_thread::sleep_for(std::chrono::milliseconds(barrier.Done() ? 0 : 1));
    }
  }

  barrier.Wait();
}

void ThreadPool::ParallelForRange(int64_t first, int64_t last, std::function<void(int64_t, int64_t)> fn) {
  if (last <= first) return;
  if (last - first == 1) {
    fn(first, last);
    return;
  }

  // TODO: Eigen supports a more efficient ThreadPoolDevice mechanism
  // We will simply rely on the work queue and stealing in the short term.
  EigenExtendedBarrier barrier(static_cast<unsigned int>(last - first));
  std::function<void(int64_t, int64_t)> handle_range = [&barrier, &fn](int64_t first, int64_t last) {
    fn(first, last);
    barrier.Notify();
  };

  for (int64_t id = first; id <= last; ++id) {
    Schedule([=, &handle_range]() { handle_range(id, id + 1); });
  }

  if (impl_.CurrentThreadId() != -1) {
    // A thread in the threadpool can't wait or it will cause deadlocks.
    while (!barrier.Done()) {
      impl_.Help(impl_.CurrentThreadId());
      std::this_thread::sleep_for(std::chrono::milliseconds(barrier.Done() ? 0 : 1));
    }
  }

  barrier.Wait();
}

// void ThreadPool::SetStealPartitions(const std::vector<std::pair<unsigned, unsigned>>& partitions) {
//   impl_->SetStealPartitions(partitions);
// }

int ThreadPool::NumThreads() const { return impl_.NumThreads(); }

int ThreadPool::CurrentThreadId() const { return impl_.CurrentThreadId(); }


bool ThreadPool::EigenExtendedBarrier::Done() {
  return (state_.fetch_or(1, std::memory_order_acq_rel) >> 1) == 0;
}


void ThreadPool::EigenExtendedThreadPool::Help(int thread_id) {
  Queue& q = thread_data_[thread_id].queue;
  if (!cancelled_) {
    Task t = q.PopFront();
    if (!t.f) {
      t = LocalSteal();
      if (!t.f) {
        t = GlobalSteal();
        if (!t.f) {
          return;
        }
      }
    }
    if (t.f) {
      env_.ExecuteTask(t);
    }
  }
}

}  // namespace concurrency
}  // namespace onnxruntime
