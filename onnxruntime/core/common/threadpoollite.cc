#include "core/platform/threadpoollite.h"
#include "core/common/spin_pause.h"

namespace onnxruntime {

namespace concurrency {

ThreadPoolLite::ThreadPoolLite(Env*,
                               const ThreadOptions&,
                               const NAME_CHAR_TYPE*,
                               int num_threads,
                               bool) : num_sub_threads_(num_threads - 1), profiler_(num_sub_threads_, ORT_TSTR("ThreadPoolLite")) {
  ORT_ENFORCE(num_sub_threads_ > -1);
  size_t affinity_mask = 3;
  for (int i = 0; i < num_sub_threads_; ++i) {
    sub_threads_.emplace_back(&ThreadPoolLite::MainLoop, this, i);
    SetThreadAffinityMask(sub_threads_.back().native_handle(), affinity_mask);
    affinity_mask <<= 2;
  }
}

ThreadPoolLite::~ThreadPoolLite() {
  exit_ = true;
  for (std::thread& t : sub_threads_) {
    t.join();
  }
}

void ThreadPoolLite::ParallelFor(std::ptrdiff_t total, double, const Fn& fn) {
  if (total <= 0) {
    return;
  }
  auto all_threads = num_sub_threads_ + 1;
  auto share_per_thread = total / all_threads + ((total % all_threads == 0) ? 0 : 1);
  auto shares = total / share_per_thread + ((total % share_per_thread) == 0 ? 0 : 1);
  SimpleFn simpe_fn = [&](std::ptrdiff_t idx) {
    auto from = share_per_thread * idx;
    auto to = std::min(total, from + share_per_thread);
    if (from < to) {
      fn(from, to);
    }
  };
  SimpleParallelFor(shares, simpe_fn);
}

void ThreadPoolLite::ParallelFor(std::ptrdiff_t total, const TensorOpCost&, const Fn& fn) {
  ParallelFor(total, 0.f, fn);
}

void ThreadPoolLite::SimpleParallelFor(std::ptrdiff_t total, const SimpleFn& fn) {
  profiler_.LogCoreAndBlock(1);
  if (total <= 0) {
    return;
  } else if (1 == total) {
    fn(0);
  } else {
    int insert_at = 0;
    profiler_.LogStart();
    for (; insert_at < MAX_NUM_TASK; ++insert_at) {
      Task& task = tasks_[insert_at];
      Status status{Empty};
      if (task.status_.compare_exchange_weak(status, Loading, std::memory_order_relaxed)) {
        task.progress_.store(total - 1);
        task.done_.store(0);
        task.fn_ = &fn;
        task.status_.store(Ready, std::memory_order_release);
        break;
      }
    }
    profiler_.LogEndAndStart(ThreadPoolProfiler::DISTRIBUTION);
    if (insert_at == MAX_NUM_TASK) {
      for (int i = 0; i < total; ++i) {
        fn(i);
      }
      profiler_.LogEndAndStart(ThreadPoolProfiler::RUN);
      profiler_.LogEnd(ThreadPoolProfiler::WAIT);
    } else {
      Task& task = tasks_[insert_at];
      long long progress = -1;
      while ((progress = task.progress_.fetch_sub(1, std::memory_order_relaxed)) > -1) {
        fn(progress);
        task.done_.fetch_add(1, std::memory_order_relaxed);
      }
      profiler_.LogEndAndStart(ThreadPoolProfiler::RUN);
      while (task.done_.load(std::memory_order_relaxed) < total) {
        onnxruntime::concurrency::SpinPause();
      }
      task.status_.store(Empty, std::memory_order_relaxed);
      profiler_.LogEnd(ThreadPoolProfiler::WAIT);
    }
  }
}

void ThreadPoolLite::Schedule(SchdFn) {
  ORT_ENFORCE(false);
}

void ThreadPoolLite::StartProfiling() {
  profiler_.Start();
}

std::string ThreadPoolLite::StopProfiling() {
  return profiler_.Stop();
}

void ThreadPoolLite::MainLoop(int idx) {
  profiler_.LogThreadId(idx);
  while (!exit_) {
    for (int i = 0; i < MAX_NUM_TASK; ++i) {
      Task& task = tasks_[i];
      if (task.status_.load(std::memory_order_acquire) == Ready) {
        profiler_.LogRun(idx);
        long long progress = -1;
        while ((progress = task.progress_.fetch_sub(1, std::memory_order_relaxed)) > -1) {
          const SimpleFn& simple_fn = *task.fn_;
          simple_fn(progress);
          task.done_.fetch_add(1, std::memory_order_relaxed);
          if (0 == progress) {
            break;
          }
        }
      }
    }
  }
}

}  // namespace concurrency
}  // namespace onnxruntime