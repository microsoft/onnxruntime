// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace onnxruntime::contrib::cuda::qmoe {

inline size_t SpinPoolBatchSize(size_t count, size_t threads, size_t requested) {
  return std::min(requested, std::max(size_t{1}, count / threads));
}

// Unified bounded CPU expert service (v3 Section 8.3 / F9).
// Used by both synchronous and asynchronous (Option A stream callback) execution paths.
// Generation-tagged claims prevent late workers from executing a later job with stale metadata.
class SpinPool final {
 public:
  explicit SpinPool(int threads, int spin_us = 2000) : spin_us_(spin_us) {
    if (threads < 1) throw std::invalid_argument("QMoE pool threads must be positive.");
    for (int worker = 1; worker < threads; ++worker) workers_.emplace_back([this] { Loop(); });
  }

  ~SpinPool() {
    stop_.store(true);
    {
      std::lock_guard<std::mutex> lock(sleep_mutex_);
      wake_.notify_all();
    }
    for (auto& worker : workers_) worker.join();
  }

  void SetSpinDurationUs(int spin_us) noexcept {
    spin_us_.store(std::max(0, spin_us), std::memory_order_relaxed);
  }

  int ThreadCount() const noexcept {
    return static_cast<int>(workers_.size()) + 1;
  }

  template <typename Function>
  void ParallelFor(ptrdiff_t count, Function& function, int batch = 1) {
    if (count < 0 || batch < 1 || batch > 16) throw std::invalid_argument("Invalid QMoE pool job/batch.");
    Run(static_cast<size_t>(count),
        [](void* data, size_t task) { (*static_cast<Function*>(data))(static_cast<ptrdiff_t>(task)); },
        &function, static_cast<size_t>(batch));
  }

 private:
  static constexpr uint64_t kFieldBits = 20;
  static constexpr uint64_t kFieldMask = (uint64_t{1} << kFieldBits) - 1;

  static uint64_t Pack(uint64_t generation, uint64_t count, uint64_t next) {
    return (generation << (2 * kFieldBits)) | (count << kFieldBits) | next;
  }
  static bool HasWork(uint64_t ticket) { return (ticket & kFieldMask) < ((ticket >> kFieldBits) & kFieldMask); }

  void Run(size_t count, void (*function)(void*, size_t), void* data, size_t requested_batch) {
    if (count == 0) return;
    if (workers_.empty() || count > kFieldMask) {
      for (size_t task = 0; task < count; ++task) function(data, task);
      return;
    }
    std::lock_guard<std::mutex> serial(call_mutex_);
    function_ = function;
    data_ = data;
    batch_atomic_.store(SpinPoolBatchSize(count, workers_.size() + 1, requested_batch), std::memory_order_relaxed);
    completed_.store(0, std::memory_order_relaxed);
    ++generation_;
    ticket_.store(Pack(generation_, count, 0), std::memory_order_seq_cst);
    if (sleepers_.load(std::memory_order_seq_cst) > 0) {
      std::lock_guard<std::mutex> lock(sleep_mutex_);
      wake_.notify_all();
    }
    RunTasks();
    while (completed_.load(std::memory_order_acquire) != count) Pause();
  }

  void RunTasks() {
    uint64_t ticket = ticket_.load(std::memory_order_acquire);
    while (HasWork(ticket)) {
      const uint64_t next = ticket & kFieldMask;
      const uint64_t count = (ticket >> kFieldBits) & kFieldMask;
      const uint64_t batch = std::min(count - next, batch_atomic_.load(std::memory_order_relaxed));
      if (ticket_.compare_exchange_weak(ticket, ticket + batch, std::memory_order_acq_rel, std::memory_order_acquire)) {
        for (uint64_t offset = 0; offset < batch; ++offset) function_(data_, static_cast<size_t>(next + offset));
        completed_.fetch_add(static_cast<size_t>(batch), std::memory_order_release);
        ticket = ticket_.load(std::memory_order_acquire);
      }
    }
  }

  static void Pause() {
#if defined(__x86_64__)
    __builtin_ia32_pause();
#endif
  }

  void Loop() {
    while (!stop_.load(std::memory_order_relaxed)) {
      RunTasks();
      auto idle_since = std::chrono::steady_clock::now();
      for (uint32_t spins = 1; !stop_.load(std::memory_order_relaxed); ++spins) {
        if (HasWork(ticket_.load(std::memory_order_acquire))) break;
        Pause();
        const int cur_spin_us = spin_us_.load(std::memory_order_relaxed);
        if ((spins & 0x3ff) == 0 &&
            std::chrono::steady_clock::now() - idle_since > std::chrono::microseconds(cur_spin_us)) {
          std::unique_lock<std::mutex> lock(sleep_mutex_);
          sleepers_.fetch_add(1, std::memory_order_seq_cst);
          wake_.wait(lock, [this] { return stop_.load() || HasWork(ticket_.load(std::memory_order_seq_cst)); });
          sleepers_.fetch_sub(1, std::memory_order_seq_cst);
          idle_since = std::chrono::steady_clock::now();
        }
      }
    }
  }

  std::atomic<int> spin_us_{2000};
  std::vector<std::thread> workers_;
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> ticket_{0};
  std::atomic<size_t> completed_{0};
  std::atomic<int> sleepers_{0};
  std::atomic<uint64_t> batch_atomic_{1};
  std::mutex call_mutex_;
  std::mutex sleep_mutex_;
  std::condition_variable wake_;
  uint64_t generation_ = 0;
  void (*function_)(void*, size_t) = nullptr;
  void* data_ = nullptr;
};

}  // namespace onnxruntime::contrib::cuda::qmoe
