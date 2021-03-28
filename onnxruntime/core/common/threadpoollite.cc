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
  profiler_.LogStartAndCoreAndBlock(1);
  if (total <= 0) {
    return;
  } else if (1 == total) {
    fn(0);
    profiler_.LogEnd(ThreadPoolProfiler::RUN);
  } else {
    int at = 0;
    Task task{(std::ptrdiff_t)(&fn), total - 1, 0};
    for (; at < MAX_NUM_TASK; ++at) {
      Task candidate = tasks_[at].load(std::memory_order_relaxed);
      if (0 == candidate.fn_) {
        if (tasks_[at].compare_exchange_weak(candidate, task, std::memory_order_relaxed)) {
          break;
        }
      }
    }
    profiler_.LogEndAndStart(ThreadPoolProfiler::DISTRIBUTION);
    if (at == MAX_NUM_TASK) {
      for (int i = 0; i < total; ++i) {
        fn(i);
      }
    } else {
      while (true) {
        Task inserted = tasks_[at].load(std::memory_order_relaxed);
        ORT_ENFORCE(0 != inserted.fn_, "function ptr must be non-empty!");
        if (total == inserted.done_) {
          if (tasks_[at].compare_exchange_weak(inserted, {0, 0, 0}, std::memory_order_relaxed)) {
            break;
          }
        } else if (inserted.progress_ >= 0) {
          Task next = {inserted.fn_, inserted.progress_ - 1, inserted.done_};
          if (tasks_[at].compare_exchange_weak(inserted, next, std::memory_order_relaxed)) {
            fn(inserted.progress_);
            while (!tasks_[at].compare_exchange_weak(next,
                                                     {next.fn_, next.progress_, next.done_ + 1},
                                                     std::memory_order_relaxed))
              ;
          }
        }
      }
    }
    profiler_.LogEnd(ThreadPoolProfiler::RUN);
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
      Task task = tasks_[i].load(std::memory_order_relaxed);
      if (0 != task.fn_ && task.progress_ >= 0) {
        if (tasks_[i].compare_exchange_weak(task,
                                            {task.fn_, task.progress_ - 1, task.done_},
                                            std::memory_order_relaxed)) {
          const SimpleFn* fn = (const SimpleFn*)(task.fn_);
          (*fn)(task.progress_--);
          profiler_.LogRun(idx);
          while (!tasks_[i].compare_exchange_weak(task,
                                                  {task.fn_, task.progress_, task.done_ + 1},
                                                  std::memory_order_relaxed))
            ;
        }
      }
    }
  }
}

}  // namespace concurrency
}  // namespace onnxruntime