// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/threadpoollite.h"
#include "core/common/spin_pause.h"

namespace onnxruntime {

namespace concurrency {

ThreadPoolLite::ThreadPoolLite(Env*,
                               const ThreadOptions& options,
                               const NAME_CHAR_TYPE*,
                               int num_threads,
                               bool) : profiler_(num_threads - 1, ORT_TSTR("ThreadPoolLite")) {
  num_sub_threads_ = num_threads - 1;
  slots_.assign(num_threads - 1, {});
  set_denormal_as_zero_ = options.set_denormal_as_zero;
  auto cores = static_cast<int>(std::thread::hardware_concurrency()) >> 1;
  ORT_ENFORCE(cores > 0, "failed to get a valid number of cpu cores");
#ifdef _WIN32
  size_t affinity_mask = 3;
  for (int i = 0; i < num_sub_threads_; ++i) {
    if (i > 0 && (i % cores) == 0) {
      affinity_mask = 3;
    }
    sub_threads_.emplace_back(&ThreadPoolLite::ThreadLoop, this, i);
    ORT_ENFORCE(0 != SetThreadAffinityMask(sub_threads_.back().native_handle(), affinity_mask),
        "Failed to set thread affinity on windows.");
    affinity_mask <<= 2;
  }
#else
  for (int i = 0; i < num_sub_threads_; ++i) {
    sub_threads_.emplace_back(&ThreadPoolLite::ThreadLoop, this, i);
#if !defined(__APPLE__) && !defined(__ANDROID__) && !defined(__wasm__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(i%cores, &cpuset);
    ORT_ENFORCE(0 == pthread_setaffinity_np(sub_threads_.back().native_handle(), sizeof(cpu_set_t), &cpuset),
                "Failed to set thread affinity on posix system.");
#endif
  }
#endif
}

ThreadPoolLite::~ThreadPoolLite() {
  exit_ = true;
  for (std::thread& t : sub_threads_) {
    t.join();
  }
}

void ThreadPoolLite::ParallelFor(std::ptrdiff_t total, double c, const Fn& fn) {
  TensorOpCost cost{0, 0, c};
  ParallelFor(total, cost, fn);
}

void ThreadPoolLite::ParallelFor(std::ptrdiff_t total, const TensorOpCost& cost, const Fn& fn) {
  if (0 == total) {
    return;
  }
  std::ptrdiff_t block_size = GetBlockSize(total, cost, num_sub_threads_ + 1);
  ORT_ALIGN_TO_AVOID_FALSE_SHARING std::atomic<std::ptrdiff_t> iter{0};
  SchdFn schd_fn = [&]() {
    std::ptrdiff_t i{0};
    while ((i = iter.fetch_add(block_size, std::memory_order_relaxed)) < total) {
      fn(i, std::min(total, i + block_size));
    }
  };
  ParallelForImpl(schd_fn, block_size);
}

void ThreadPoolLite::SimpleParallelFor(std::ptrdiff_t total, const SimpleFn& fn) {
  ORT_ALIGN_TO_AVOID_FALSE_SHARING std::atomic<std::ptrdiff_t> iter{0};
  SchdFn schd_fn = [&]() {
    std::ptrdiff_t i{0};
    while ((i = iter.fetch_add(1, std::memory_order_relaxed)) < total) {
      fn(i);
    }
  };
  ParallelForImpl(schd_fn, 1);
}

void ThreadPoolLite::ParallelForImpl(const SchdFn& schd_fn, std::ptrdiff_t block_size) {
  profiler_.LogStartAndCoreAndBlock(block_size);
  std::vector<Slot*> engaged_slots;
  for (int i = 0; i < num_sub_threads_; ++i) {
    Stage stage = Stage::empty;
    if (slots_[i].stage_.compare_exchange_weak(stage, Stage::loading, std::memory_order_relaxed)) {
      slots_[i].schd_fn_ = schd_fn;
      slots_[i].stage_.store(Stage::ready, std::memory_order_release);
      engaged_slots.push_back(&slots_[i]);
    }
  }
  profiler_.LogEndAndStart(ThreadPoolProfiler::DISTRIBUTION);
  schd_fn();
  profiler_.LogEndAndStart(ThreadPoolProfiler::RUN);
  for (auto slot : engaged_slots) {
    Stage stage_ready = Stage::ready;
    Stage stage_done = Stage::done;
    while (!slot->stage_.compare_exchange_weak(stage_ready, Stage::empty, std::memory_order_relaxed) &&
           !slot->stage_.compare_exchange_weak(stage_done, Stage::empty, std::memory_order_relaxed)) {
      stage_ready = Stage::ready;
      stage_done = Stage::done;
      onnxruntime::concurrency::SpinPause();
    }
  }
  profiler_.LogEnd(ThreadPoolProfiler::WAIT);
}

void ThreadPoolLite::Schedule(SchdFn schd_fn) {
  schd_fn();
}

void ThreadPoolLite::StartProfiling() {
  profiler_.Start();
}

std::string ThreadPoolLite::StopProfiling() {
  return profiler_.Stop();
}

void ThreadPoolLite::ThreadLoop(int thread_id) {
  profiler_.LogThreadId(thread_id);
  auto& slot = slots_[thread_id];
  SetDenormalAsZero(set_denormal_as_zero_);
  while (!exit_) {
    Stage stage = Stage::ready;
    if (slot.stage_.compare_exchange_weak(stage, Stage::running, std::memory_order_acquire, std::memory_order_relaxed)) {
      slot.schd_fn_();
      profiler_.LogRun(thread_id);
      slot.stage_.store(Stage::done, std::memory_order_relaxed);
    } else {
      onnxruntime::concurrency::SpinPause();
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////

#define ZERO ((integer)0)
std::memory_order relaxed = std::memory_order_relaxed;
std::memory_order release = std::memory_order_release;
std::memory_order acquire = std::memory_order_acquire;

template<size_t size>
ret mpmcqueue<size>::pop() {
  return pop_at(front_.load(relaxed));
}

template <size_t size>
ret mpmcqueue<size>::pop_at(integer at) {
  ret r;
  if (at == back_.load(acquire).at_) return r;
  auto to = nodes_[at].available_.fetch_sub(nodes_[at].block_size_, relaxed);
  if (to > ZERO) {
    r.at_ = at;
    r.from_ = std::max((integer)0, to - nodes_[at].block_size_);
    r.to_ = to;
  } else {
    auto next_at = next(at);
    front_.compare_exchange_weak(at, next_at, relaxed);
  }
  return r;
}

template <size_t size>
integer mpmcqueue<size>::push(func fn, integer available, integer block_size) {
  landmark back = back_.load(relaxed);
  landmark next_back;
  do {
    auto next_at = next(back.at_);
    if (next_at == front_.load(relaxed) || !nodes_[back.at_].empty_.load(relaxed)) return -1;
    back.state_ = state::free;
    next_back.at_ = back.at_;
    next_back.state_ = state::occupied;
  } while (!back_.compare_exchange_weak(back, next_back, relaxed));
  auto insert_at = back.at_;
  nodes_[back.at_].fn_ = std::move(fn);
  nodes_[back.at_].block_size_ = block_size;
  nodes_[back.at_].done_.store(ZERO, relaxed);
  nodes_[back.at_].empty_.store(false, relaxed);
  nodes_[back.at_].available_.store(available, relaxed);
  next_back.at_ = next(next_back.at_);
  next_back.state_ = state::free;
  back_.store(next_back, release);
  return insert_at;
}

ThreadPoolLiteII::ThreadPoolLiteII(Env*,
                                   const ThreadOptions& options,
                                   const NAME_CHAR_TYPE*,
                                   int num_threads,
                                   bool) {
  num_sub_threads_ = num_threads - 1;
  set_denormal_as_zero_ = options.set_denormal_as_zero;
  // size_t affinity_mask = 3;
  // auto cores = static_cast<int>(std::thread::hardware_concurrency()) >> 1;
  for (int i = 0; i < num_sub_threads_; ++i) {
    sub_threads_.emplace_back([this, i]() {
      SetDenormalAsZero(set_denormal_as_zero_);
      while (!exit_) {
        ret r = que_.pop();
        if (r.at_ > -1) {
          que_.nodes_[r.at_].fn_(r.from_, r.to_);
          que_.nodes_[r.at_].done_.fetch_add(r.to_ - r.from_, relaxed);
        }
      }
    });
    /*
    if (i > 0 && (i % cores) == 0) {
      affinity_mask = 1;
    }
    ORT_ENFORCE(0 != SetThreadAffinityMask(sub_threads_.back().native_handle(), affinity_mask),
                "Failed to set thread affinity on windows.");
    affinity_mask <<= 2;
    */
  }  //for
}

ThreadPoolLiteII::~ThreadPoolLiteII() {
  exit_ = true;
  for (std::thread& t : sub_threads_) {
    t.join();
  }
}

void ThreadPoolLiteII::ParallelFor(std::ptrdiff_t total, double c, const Fn& fn) {
  TensorOpCost cost{0, 0, c};
  ParallelFor(total, cost, fn);
}

void ThreadPoolLiteII::ParallelFor(std::ptrdiff_t total, const TensorOpCost& cost, const Fn& fn) {
  if (0 == total) {
    return;
  }
  std::ptrdiff_t block_size = GetBlockSize(total, cost, num_sub_threads_ + 1);
  std::ptrdiff_t insert_at = -1;
  Fn* queued_fn{};
  std::atomic<integer>* done{};
  std::atomic<bool>* empty{};
  if ((insert_at = que_.push(fn, total, block_size)) > -1) {
    queued_fn = &que_.nodes_[insert_at].fn_;
    done = &que_.nodes_[insert_at].done_;
    empty = &que_.nodes_[insert_at].empty_;
    for (;;) {
      ret r = que_.pop_at(insert_at);
      if (r.at_ >- 1) {
        (*queued_fn)(r.from_, r.to_);
        done->fetch_add(r.to_ - r.from_, relaxed);
      } else {
        break;
      }
    }
    while (done->load(relaxed) < total) {
      SpinPause();
    }
    empty->store(true, relaxed);
  } else {
    fn(0, total);
  }
}

void ThreadPoolLiteII::SimpleParallelFor(std::ptrdiff_t total, const SimpleFn& simpleFn) {
  ORT_ALIGN_TO_AVOID_FALSE_SHARING std::atomic<std::ptrdiff_t> iter{0};
  func fn = [&](integer from, integer to) {
    for (integer i = from; i < to; ++i) {
      simpleFn(i);
    }
  };
  std::ptrdiff_t insert_at = -1;
  Fn* queued_fn{};
  std::atomic<integer>* done{};
  std::atomic<bool>* empty{};
  if ((insert_at = que_.push(fn, total)) > -1) {
    queued_fn = &que_.nodes_[insert_at].fn_;
    done = &(que_.nodes_[insert_at].done_);
    empty = &(que_.nodes_[insert_at].empty_);
    for (;;) {
      ret r = que_.pop_at(insert_at);
      if (r.at_ >-1) {
        (*queued_fn)(r.from_, r.to_);
        done->fetch_add(r.to_ - r.from_, relaxed);
      } else {
        break;
      }
    }
    while (done->load(relaxed) < total) {
      SpinPause();
    }
    empty->store(true, relaxed);
  } else {
    fn(0, total);
  }
}

void ThreadPoolLiteII::Schedule(SchdFn schd_fn) {
  schd_fn();
}

void ThreadPoolLiteII::StartProfiling() {}

std::string ThreadPoolLiteII::StopProfiling() {
  return {};
}

}  // namespace concurrency
}  // namespace onnxruntime
