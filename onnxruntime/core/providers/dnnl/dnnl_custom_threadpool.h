// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License
#pragma once

#include <algorithm>
#include "dnnl_threadpool_iface.hpp"
#include "core/platform/threadpool.h"
#include "core/platform/EigenNonBlockingThreadPool.h"

// oneDNN limitations using eigen
#if !(EIGEN_WORLD_VERSION == 3 && EIGEN_MAJOR_VERSION == 3) && defined(DNNL_EIGEN_THREAD)
#error Unsupported Eigen version (need 3.3.x)
#endif

// To improve readability
using DnnlThreadPoolIface = dnnl::threadpool_interop::threadpool_iface;
using ORTThreadPool = onnxruntime::concurrency::ThreadPool;
using EigenThreadPool = Eigen::ThreadPool;


inline int DnnlCalcNumThreads() {
  int num_threads = 0;
#if _WIN32
  // Indeed 64 should be enough. However, it's harmless to have a little more.
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer[256];
  DWORD returnLength = sizeof(buffer);
  if (GetLogicalProcessorInformation(buffer, &returnLength) == FALSE) {
    num_threads = std::thread::hardware_concurrency();
  } else {
    int count = (int)(returnLength / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    for (int i = 0; i != count; ++i) {
      if (buffer[i].Relationship == RelationProcessorCore) {
        ++num_threads;
      }
    }
  }
#endif
  if (!num_threads)
    num_threads = std::thread::hardware_concurrency();

  return num_threads;
}

template <typename T>
struct remove_reference {
    typedef T type;
};
// This code distributs the work along the available threads,
// this function was extracted with minimal changes from the oneDNN code..
template <typename T, typename U>
inline void balance211(T n, U team, U tid, T& n_start, T& n_end) {
  T n_min = 1;
  T& n_my = n_end;
  if (team <= 1 || n == 0) {
    n_start = 0;
    n_my = n;
  } else if (n_min == 1) {
    // team = T1 + T2
    // n = T1*n1 + T2*n2  (n1 - n2 = 1)
    T c_team = (T)team;
    assert(c_team);
    T n1 = static_cast<typename remove_reference<T>::type>((n + c_team - 1) / c_team);
    T n2 = n1 - 1;
    T T1 = n - n2 * c_team;
    n_my = (T)tid < T1 ? n1 : n2;
    n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
  }

  n_end += n_start;
}

// Define base beheaviour for all custom threadpools
template <class T>
class DnnlThreadPool : public DnnlThreadPoolIface {
 public:
  explicit DnnlThreadPool(T* threadpool) : inner_threadpool_(threadpool) {}
  uint64_t get_flags() const override { return ASYNCHRONOUS; }

 protected:
  T* inner_threadpool_;
};


class DnnlORTThreadPool : public DnnlThreadPool<ORTThreadPool> {
 public:
  explicit DnnlORTThreadPool(ORTThreadPool* threadpool) : DnnlThreadPool<ORTThreadPool>(threadpool) {}
  int get_num_threads() const override { return inner_threadpool_->DegreeOfParallelism(inner_threadpool_); }
  bool get_in_parallel() const override { return inner_threadpool_->CurrentThreadId() != -1; }
  uint64_t get_flags() const override { return !ASYNCHRONOUS; }

  void parallel_for(int n, const std::function<void(int, int)>& fn) override {
    // TryBatchParallelFor uses a function with just one argument so we fix this here by using lambda fn
    inner_threadpool_->TryBatchParallelFor(
        inner_threadpool_, n, [&](ptrdiff_t i) { fn(static_cast<int>(i), n); }, 0);
  }
};


class DnnlEigenThreadPool : public DnnlThreadPool<EigenThreadPool> {
 public:
  // Here we modify a little the constructor in order to create the thread pool
  explicit DnnlEigenThreadPool(int num_threads = 0) :
            DnnlThreadPool(num_threads <= 0 ?
                          (new EigenThreadPool(DnnlCalcNumThreads())):
                          (new EigenThreadPool(num_threads))) {}

  int get_num_threads() const override { return inner_threadpool_->NumThreads(); }
  bool get_in_parallel() const override { return inner_threadpool_->CurrentThreadId() != -1; }

  void parallel_for(int n, const std::function<void(int, int)>& fn) override {
    int nthr = get_num_threads();
    int njobs = std::min(n, nthr);

    for (int i = 0; i < njobs; i++) {
      inner_threadpool_->Schedule([i, n, njobs, fn]() {
        int start, end;
        balance211(n, njobs, i, start, end);
        for (int j = start; j < end; j++)
          fn(j, n);
      });
    }
  }
};
