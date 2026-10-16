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

#if defined(_MSC_VER) && (defined(_M_AMD64) || defined(_M_IX86))
#include <intrin.h>
#endif

#if defined(__linux__)
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace onnxruntime::contrib::cuda::qmoe {

#if defined(__linux__)
inline void ParseLinuxCpuRangeList(const std::string& text, std::vector<int>* out) {
  size_t pos = 0;
  while (pos < text.size()) {
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == ',')) {
      ++pos;
    }
    if (pos >= text.size()) break;
    size_t end = pos;
    while (end < text.size() && text[end] != ',' && text[end] != '\n') ++end;
    const std::string token = text.substr(pos, end - pos);
    const size_t dash = token.find('-');
    try {
      if (dash == std::string::npos) {
        out->push_back(std::stoi(token));
      } else {
        const int lo = std::stoi(token.substr(0, dash));
        const int hi = std::stoi(token.substr(dash + 1));
        for (int v = lo; v <= hi; ++v) out->push_back(v);
      }
    } catch (...) {
    }
    pos = end + 1;
  }
}

inline std::vector<int> DiscoverLinuxMemoryNodes() {
  std::vector<int> nodes;
  std::ifstream in("/sys/devices/system/node/has_memory");
  if (!in) in.open("/sys/devices/system/node/online");
  if (in) {
    std::string line;
    std::getline(in, line);
    ParseLinuxCpuRangeList(line, &nodes);
  }
  return nodes;
}

inline std::vector<cpu_set_t> DiscoverAllowedNumaNodeCpuSets() {
  std::vector<cpu_set_t> node_sets;
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(cpu_set_t), &allowed) != 0) {
    return node_sets;
  }
  const std::vector<int> nodes = DiscoverLinuxMemoryNodes();
  for (int node : nodes) {
    std::ifstream in("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
    if (!in) continue;
    std::string line;
    std::getline(in, line);
    std::vector<int> cpus;
    ParseLinuxCpuRangeList(line, &cpus);
    cpu_set_t mask;
    CPU_ZERO(&mask);
    int count = 0;
    for (int cpu : cpus) {
      if (cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, &allowed)) {
        CPU_SET(cpu, &mask);
        ++count;
      }
    }
    if (count > 0) {
      node_sets.push_back(mask);
    }
  }
  return node_sets;
}
#endif

// Applies MPOL_INTERLEAVE page striping across online NUMA memory nodes on Linux so dual-socket
// DDR5 channels are balanced regardless of which socket first faults the expert weight pages.
inline bool ApplyNumaInterleavePolicy(const void* addr, size_t bytes) noexcept {
#if defined(__linux__) && defined(__NR_mbind)
  if (addr == nullptr || bytes == 0) return false;
  if (const char* env = std::getenv("ORT_QMOE_NUMA_INTERLEAVE"); env != nullptr && std::strcmp(env, "0") == 0) {
    return false;
  }
  static const unsigned long kNodeMask = []() -> unsigned long {
    unsigned long mask = 0;
    const auto nodes = DiscoverLinuxMemoryNodes();
    if (nodes.size() < 2) return 0;
    for (int n : nodes) {
      if (n >= 0 && static_cast<size_t>(n) < sizeof(unsigned long) * 8) {
        mask |= (1UL << n);
      }
    }
    return mask;
  }();
  if (kNodeMask == 0) return false;
  const size_t page_size = 4096;
  const uintptr_t raw = reinterpret_cast<uintptr_t>(addr);
  const uintptr_t aligned_start = raw & ~(static_cast<uintptr_t>(page_size) - 1);
  const size_t aligned_len = ((raw + bytes - aligned_start) + page_size - 1) & ~(page_size - 1);
  constexpr int kMpolInterleave = 3;
  constexpr unsigned kMpolMfMove = (1u << 1);
  const unsigned long maxnode = sizeof(unsigned long) * 8;
  long rc = syscall(__NR_mbind, reinterpret_cast<void*>(aligned_start), static_cast<unsigned long>(aligned_len),
                    kMpolInterleave, &kNodeMask, maxnode, 0u);
  if (rc != 0) {
    rc = syscall(__NR_mbind, reinterpret_cast<void*>(aligned_start), static_cast<unsigned long>(aligned_len),
                 kMpolInterleave, &kNodeMask, maxnode, kMpolMfMove);
  }
  return rc == 0;
#else
  (void)addr;
  (void)bytes;
  return false;
#endif
}

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
#if defined(__linux__)
    const char* pin_env = std::getenv("ORT_QMOE_NUMA_PIN_WORKERS");
    const bool enable_pin = (pin_env == nullptr || std::strcmp(pin_env, "0") != 0);
    if (enable_pin) {
      node_masks_ = DiscoverAllowedNumaNodeCpuSets();
    }
    numa_node_count_ = static_cast<int>(node_masks_.size());
#endif
    for (int worker = 1; worker < threads; ++worker) {
      workers_.emplace_back([this, worker] { Loop(worker); });
    }
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

  int SpinDurationUs() const noexcept {
    return spin_us_.load(std::memory_order_relaxed);
  }

  int ThreadCount() const noexcept {
    return static_cast<int>(workers_.size()) + 1;
  }

  int NumaNodeCount() const noexcept {
    return numa_node_count_;
  }

  int PinnedWorkerCount() const noexcept {
    return pinned_workers_.load(std::memory_order_acquire);
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
#if defined(_MSC_VER) && (defined(_M_AMD64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) && defined(__GNUC__)
    __asm__ __volatile__("yield" ::: "memory");
#endif
  }

  void Loop(int worker_idx) {
#if defined(__linux__)
    if (!node_masks_.empty()) {
      const size_t node_slot = static_cast<size_t>(worker_idx) % node_masks_.size();
      if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &node_masks_[node_slot]) == 0) {
        pinned_workers_.fetch_add(1, std::memory_order_release);
      }
    }
#else
    (void)worker_idx;
#endif
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
  int numa_node_count_{0};
  std::atomic<int> pinned_workers_{0};
#if defined(__linux__)
  std::vector<cpu_set_t> node_masks_;
#endif
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

// Host CPU budget group coordinating SpinPool spin policy across prefill/embedding/Engram and decode phases (v3 Section 8.3 / F9).
class CpuBudgetGroup final {
 public:
  enum class Phase : int {
    kDecode = 0,
    kPrefillOrAux = 1,
  };

  static CpuBudgetGroup& Default() {
    static CpuBudgetGroup instance;
    return instance;
  }

  void ConfigureDecodeSpinUs(int spin_us) noexcept {
    decode_spin_us_.store(std::max(0, spin_us), std::memory_order_relaxed);
  }

  void EnterPrefillOrAuxPhase(SpinPool* pool) noexcept {
    active_phase_.store(static_cast<int>(Phase::kPrefillOrAux), std::memory_order_release);
    if (pool) {
      pool->SetSpinDurationUs(0);
    }
  }

  void EnterDecodePhase(SpinPool* pool) noexcept {
    active_phase_.store(static_cast<int>(Phase::kDecode), std::memory_order_release);
    if (pool) {
      pool->SetSpinDurationUs(decode_spin_us_.load(std::memory_order_relaxed));
    }
  }

  Phase CurrentPhase() const noexcept {
    return static_cast<Phase>(active_phase_.load(std::memory_order_acquire));
  }

 private:
  std::atomic<int> decode_spin_us_{2000};
  std::atomic<int> active_phase_{static_cast<int>(Phase::kDecode)};
};

}  // namespace onnxruntime::contrib::cuda::qmoe
