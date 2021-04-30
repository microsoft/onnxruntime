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
    assert(total < (1UL << 15));
    int16_t progress = static_cast<int16_t>(total);
    int16_t num_threads = static_cast<int16_t>(num_sub_threads_) + 1;
    int16_t step = progress / num_threads + ((progress % num_threads) ? 1 : 0);
    Task task{(std::ptrdiff_t)(&fn), progress - 1, step, 0};
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
          if (tasks_[at].compare_exchange_weak(inserted, {0, 0, 0, 0}, std::memory_order_relaxed)) {
            break;
          }
        } else if (inserted.progress_ >= 0) {
          Task next = {inserted.fn_, inserted.progress_ - inserted.step_, inserted.step_, inserted.done_};
          if (tasks_[at].compare_exchange_weak(inserted, next, std::memory_order_relaxed)) {
            int16_t run_to = std::max<int16_t>(next.progress_, -1);
            int16_t run_count = inserted.progress_ - run_to;
            for (int16_t i = inserted.progress_; i > run_to; --i) {
              fn(static_cast<std::ptrdiff_t>(i));
            }
            while (!tasks_[at].compare_exchange_weak(next,
                                                     {next.fn_, next.progress_, next.step_, next.done_ + run_count},
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
                                            {task.fn_, task.progress_ - task.step_, task.step_, task.done_},
                                            std::memory_order_relaxed)) {
          const SimpleFn* fn = (const SimpleFn*)(task.fn_);
          int16_t run_to = std::max<int16_t>(task.progress_ - task.step_, -1);
          int16_t run_count = task.progress_ - run_to;
          for (int16_t j = task.progress_; j > run_to; --j) {
            (*fn)(static_cast<std::ptrdiff_t>(j));
          }
          profiler_.LogRun(idx);
          task.progress_ -= task.step_;
          while (!tasks_[i].compare_exchange_weak(task,
                                                  {task.fn_, task.progress_, task.step_, task.done_ + run_count},
                                                  std::memory_order_relaxed))
            ;
        }
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t ThreadPerPool, int32_t PoolSize>
ThreadPoolLite2<ThreadPerPool, PoolSize>::ThreadPoolLite2(Env*,
                                                          const ThreadOptions&,
                                                          const NAME_CHAR_TYPE*,
                                                          int num_threads,
                                                          bool) : profiler_(num_threads - 1, ORT_TSTR("ThreadPoolLite")) {
  int num_sub_threads = num_threads - 1;
  num_pools_ = num_sub_threads / ThreadPerPool + (num_sub_threads % ThreadPerPool ? 1 : 0);
  num_slots_ = num_pools_ * PoolSize;
  slots_.reset(new Slot[num_slots_]);
  size_t affinity_mask = 3;
  for (int i = 0; i < num_sub_threads; ++i) {
    sub_threads_.emplace_back(&ThreadPoolLite2::MainLoop, this, i);
    SetThreadAffinityMask(sub_threads_.back().native_handle(), affinity_mask);
    affinity_mask <<= 2;
  }
}

template <int32_t ThreadPerPool, int32_t PoolSize>
ThreadPoolLite2<ThreadPerPool, PoolSize>::~ThreadPoolLite2() {
  exit_ = true;
  for (std::thread& t : sub_threads_) {
    t.join();
  }
  slots_.reset();
}

template <int32_t ThreadPerPool, int32_t PoolSize>
void ThreadPoolLite2<ThreadPerPool, PoolSize>::StartProfiling() {
  profiler_.Start();
}

template <int32_t ThreadPerPool, int32_t PoolSize>
std::string ThreadPoolLite2<ThreadPerPool, PoolSize>::StopProfiling() {
  return profiler_.Stop();
}

template <int32_t ThreadPerPool, int32_t PoolSize>
void ThreadPoolLite2<ThreadPerPool, PoolSize>::ParallelFor(std::ptrdiff_t total, double, const Fn& fn) {
  SimpleFn simple_fn = [&](std::ptrdiff_t i) {
    if (i < total) {
      fn(i, i + 1);
    };
  };
  SimpleParallelFor(total, simple_fn);
}

template <int32_t ThreadPerPool, int32_t PoolSize>
void ThreadPoolLite2<ThreadPerPool, PoolSize>::ParallelFor(std::ptrdiff_t total, const TensorOpCost&, const Fn& fn) {
  double cost = 0;
  ParallelFor(total, cost, fn);
}

template <int32_t ThreadPerPool, int32_t PoolSize>
void ThreadPoolLite2<ThreadPerPool, PoolSize>::SimpleParallelFor(std::ptrdiff_t total, const SimpleFn& fn) {
  std::atomic<std::ptrdiff_t> iter{0};
  SchdFn schd_fn = [&]() {
    std::ptrdiff_t i{0};
    while ((i = iter.fetch_add(1, std::memory_order_relaxed)) < total) {
      fn(i);
    }
  };
  std::vector<int32_t> pushed;
  for (int32_t i = 0; i < num_pools_; i++) {
    auto at_from = i * PoolSize;
    auto at_to = at_from + PoolSize;
    for (auto at = at_from; at < at_to; at++) {
      auto progress = slots_[at].progress_.load(std::memory_order_acquire);
      if (-1 == progress && slots_[at].progress_.compare_exchange_weak(progress, 0, std::memory_order_relaxed)) {
        slots_[at].schd_fn_ = schd_fn;
        slots_[at].done_.store(0, std::memory_order_relaxed);
        slots_[at].progress_.store(ThreadPerPool, std::memory_order_release); //ready
        pushed.push_back(at);
        break;
      }
    }
  }
  schd_fn();
  for (auto at : pushed) {
    int32_t progress = slots_[at].progress_.load(std::memory_order_relaxed);
    while (progress > 0 && !slots_[at].progress_.compare_exchange_weak(progress, 0, std::memory_order_relaxed)) {
      progress = slots_[at].progress_.load(std::memory_order_relaxed); //revoke
    }
    int32_t expected_done = ThreadPerPool - progress;
    while (slots_[at].done_.load(std::memory_order_relaxed) < expected_done) //wait for done
      ;
    slots_[at].progress_.store(-1, std::memory_order_relaxed); //release slot
  }
}

template <int32_t ThreadPerPool, int32_t PoolSize>
void ThreadPoolLite2<ThreadPerPool, PoolSize>::Schedule(SchdFn) {}

template <int32_t ThreadPerPool, int32_t PoolSize>
void ThreadPoolLite2<ThreadPerPool, PoolSize>::MainLoop(int idx) {
  profiler_.LogThreadId(idx);
  auto slot_from = idx / ThreadPerPool * PoolSize;
  auto slot_to = slot_from + PoolSize;
  while (!exit_) {
    for (auto i = slot_from; i < slot_to; i++) {
      auto progress = slots_[i].progress_.load(std::memory_order_acquire);
      if (progress > 0 && slots_[i].progress_.compare_exchange_weak(progress, progress - 1, std::memory_order_relaxed)) {
        slots_[i].schd_fn_();
        slots_[i].done_.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }
}

template ThreadPoolLite2<2, 8>;
}  // namespace concurrency
}  // namespace onnxruntime