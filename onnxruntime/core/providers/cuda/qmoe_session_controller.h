// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Session-scoped QMoE Offload Controller (v3 Design Sections 5.1, 7.1, 7.6, 7.7, 8, 9).
// Owned per CUDA EP / session instance so plugin and built-in EPs share the exact same
// lifecycle, completion, landing-ring, CPU service, and observation commit contracts without
// crossing the C ABI with core-internal types.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/common/status.h"
#include "core/providers/cuda/cuda_device_memory_budget.h"

namespace onnxruntime::cuda {

// Per-invocation completion & observation record (v3 Section 7.1).
struct QMoEInvocationRecord {
  uint64_t run_id = 0;
  uint64_t invocation_id = 0;
  int64_t layer_index = -1;
  std::string node_name;

  std::mutex mutex;
  std::condition_variable cv;
  bool cpu_done = false;
  bool completed = false;
  bool failed = false;
  std::string error_message;

  // Optional callback invoked when the run commits its observations upon successful completion.
  std::function<void()> commit_observation;
  // Optional drain callback invoked during OnRunEnd if the invocation is still in flight.
  std::function<common::Status()> drain_device;
  cudaStream_t main_stream = nullptr;
  bool has_main_stream = false;

  void MarkFailed(std::string msg) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!failed) {
      failed = true;
      error_message = std::move(msg);
    }
    cpu_done = true;
    cv.notify_all();
  }

  void MarkCpuDone() {
    std::lock_guard<std::mutex> lock(mutex);
    cpu_done = true;
    cv.notify_all();
  }

  void MarkCompleted() {
    std::lock_guard<std::mutex> lock(mutex);
    cpu_done = true;
    completed = true;
    cv.notify_all();
  }
};

// Session-scoped landing ring with per-bank, per-generation event ordering (v3 Section 7.6 / F5 & F4).
struct QMoESessionLandingRing {
  std::mutex mutex;
  int device_id = -1;
  uint8_t* landing = nullptr;
  size_t landing_bytes = 0;
  uint8_t* pinned = nullptr;
  size_t pinned_bytes = 0;
  cudaStream_t copy_stream = nullptr;
  std::vector<cudaEvent_t> copy_done;
  std::vector<cudaEvent_t> gemm_done;

  // Per-bank (0, 1) cross-layer handoff events (F5):
  //   landing_ready[b]       recorded on copy_stream after H2D wave into bank b
  //   last_gemm_done[b]      recorded on compute stream after wave GEMM reads bank b
  //   promotion_read_done[b] recorded on promotion/compute stream after D2D promotion reads bank b
  cudaEvent_t landing_ready[2]{nullptr, nullptr};
  cudaEvent_t last_gemm_done[2]{nullptr, nullptr};
  cudaEvent_t promotion_read_done[2]{nullptr, nullptr};
  bool bank_has_gemm[2]{false, false};
  bool bank_has_promotion[2]{false, false};
  uint64_t bank_generation[2]{0, 0};

  ~QMoESessionLandingRing() {
    Shutdown();
  }

  void Shutdown() noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    if (device_id >= 0) {
      static_cast<void>(cudaSetDevice(device_id));
    }
    if (copy_stream) {
      static_cast<void>(cudaStreamSynchronize(copy_stream));
    }
    for (int b = 0; b < 2; ++b) {
      if (landing_ready[b]) {
        static_cast<void>(cudaEventDestroy(landing_ready[b]));
        landing_ready[b] = nullptr;
      }
      if (last_gemm_done[b]) {
        static_cast<void>(cudaEventDestroy(last_gemm_done[b]));
        last_gemm_done[b] = nullptr;
      }
      if (promotion_read_done[b]) {
        static_cast<void>(cudaEventDestroy(promotion_read_done[b]));
        promotion_read_done[b] = nullptr;
      }
      bank_has_gemm[b] = false;
      bank_has_promotion[b] = false;
    }
    for (auto ev : copy_done) {
      if (ev) static_cast<void>(cudaEventDestroy(ev));
    }
    copy_done.clear();
    for (auto ev : gemm_done) {
      if (ev) static_cast<void>(cudaEventDestroy(ev));
    }
    gemm_done.clear();
    if (copy_stream) {
      static_cast<void>(cudaStreamDestroy(copy_stream));
      copy_stream = nullptr;
    }
    if (pinned) {
      auto pinned_budget = GetCudaHostPinnedMemoryBudget();
      if (pinned_budget && pinned_bytes > 0) pinned_budget->BeginFreeAllocated(pinned_bytes);
      const bool ok = (cudaFreeHost(pinned) == cudaSuccess);
      if (pinned_budget && pinned_bytes > 0) pinned_budget->FinishFreeAllocated(pinned_bytes, ok);
      pinned = nullptr;
      pinned_bytes = 0;
    }
    if (landing) {
      auto dev_budget = (device_id >= 0) ? GetCudaDeviceMemoryBudgetByOrdinal(device_id) : nullptr;
      if (dev_budget && landing_bytes > 0) dev_budget->BeginFree(landing_bytes);
      const bool ok = (cudaFree(landing) == cudaSuccess);
      if (dev_budget && landing_bytes > 0) dev_budget->FinishFree(landing_bytes, ok);
      landing = nullptr;
      landing_bytes = 0;
    }
  }

  common::Status Ensure(int dev_id, size_t landing_needed, size_t pinned_needed, size_t waves) {
    device_id = dev_id;
    if (!copy_stream) {
      cudaError_t err = cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking);
      if (err != cudaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload failed to create copy_stream: ", cudaGetErrorString(err));
      }
    }
    for (int b = 0; b < 2; ++b) {
      if (!landing_ready[b]) {
        cudaError_t err = cudaEventCreateWithFlags(&landing_ready[b], cudaEventDisableTiming);
        if (err != cudaSuccess) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, cudaGetErrorString(err));
      }
      if (!last_gemm_done[b]) {
        cudaError_t err = cudaEventCreateWithFlags(&last_gemm_done[b], cudaEventDisableTiming);
        if (err != cudaSuccess) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, cudaGetErrorString(err));
      }
      if (!promotion_read_done[b]) {
        cudaError_t err = cudaEventCreateWithFlags(&promotion_read_done[b], cudaEventDisableTiming);
        if (err != cudaSuccess) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, cudaGetErrorString(err));
      }
    }
    if (landing_needed > landing_bytes) {
      if (copy_stream) static_cast<void>(cudaStreamSynchronize(copy_stream));
      for (int b = 0; b < 2; ++b) {
        if (bank_has_gemm[b]) static_cast<void>(cudaEventSynchronize(last_gemm_done[b]));
        if (bank_has_promotion[b]) static_cast<void>(cudaEventSynchronize(promotion_read_done[b]));
        bank_has_gemm[b] = false;
        bank_has_promotion[b] = false;
      }
      auto dev_budget = GetCudaDeviceMemoryBudgetByOrdinal(dev_id);
      CudaDeviceMemoryBudget::Reservation res;
      try {
        if (dev_budget) res = dev_budget->Reserve(landing_needed);
      } catch (const std::exception& ex) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload landing budget reservation failed: ", ex.what());
      }
      uint8_t* new_landing = nullptr;
      cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&new_landing), landing_needed);
      if (err != cudaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload landing cudaMalloc failed: ", cudaGetErrorString(err));
      }
      if (res.Active()) res.Commit();
      if (landing) {
        if (dev_budget && landing_bytes > 0) dev_budget->BeginFree(landing_bytes);
        const bool ok = (cudaFree(landing) == cudaSuccess);
        if (dev_budget && landing_bytes > 0) dev_budget->FinishFree(landing_bytes, ok);
      }
      landing = new_landing;
      landing_bytes = landing_needed;
    }
    if (pinned_needed > pinned_bytes) {
      if (copy_stream) static_cast<void>(cudaStreamSynchronize(copy_stream));
      auto pinned_budget = GetCudaHostPinnedMemoryBudget();
      CudaDeviceMemoryBudget::Reservation res;
      try {
        if (pinned_budget) res = pinned_budget->ReserveAllocated(pinned_needed);
      } catch (const std::exception& ex) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload landing pinned budget reservation failed: ", ex.what());
      }
      uint8_t* new_pinned = nullptr;
      cudaError_t err = cudaHostAlloc(reinterpret_cast<void**>(&new_pinned), pinned_needed, cudaHostAllocDefault);
      if (err != cudaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload landing cudaHostAlloc failed: ", cudaGetErrorString(err));
      }
      if (res.Active()) res.Commit();
      if (pinned) {
        if (pinned_budget && pinned_bytes > 0) pinned_budget->BeginFreeAllocated(pinned_bytes);
        const bool ok = (cudaFreeHost(pinned) == cudaSuccess);
        if (pinned_budget && pinned_bytes > 0) pinned_budget->FinishFreeAllocated(pinned_bytes, ok);
      }
      pinned = new_pinned;
      pinned_bytes = pinned_needed;
    }
    while (copy_done.size() < waves) {
      cudaEvent_t copy_ev = nullptr;
      cudaEvent_t gemm_ev = nullptr;
      cudaError_t err1 = cudaEventCreateWithFlags(&copy_ev, cudaEventDisableTiming);
      if (err1 != cudaSuccess) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, cudaGetErrorString(err1));
      cudaError_t err2 = cudaEventCreateWithFlags(&gemm_ev, cudaEventDisableTiming);
      if (err2 != cudaSuccess) {
        static_cast<void>(cudaEventDestroy(copy_ev));
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, cudaGetErrorString(err2));
      }
      copy_done.push_back(copy_ev);
      gemm_done.push_back(gemm_ev);
    }
    return common::Status::OK();
  }
};

class QMoESessionController {
 public:
  struct GraphReplayNodeHook {
    const void* owner = nullptr;
    std::function<common::Status(cudaStream_t)> before_replay;
    std::function<common::Status(cudaStream_t, bool)> after_replay;
  };

  explicit QMoESessionController(int device_id = 0)
      : device_id_(device_id),
        landing_ring_(std::make_shared<QMoESessionLandingRing>()) {}

  ~QMoESessionController() {
    Shutdown();
  }

  void RegisterOffloadNode() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++registered_nodes_;
  }

  void RegisterGraphReplayHook(const void* owner,
                               std::function<common::Status(cudaStream_t)> before_replay,
                               std::function<common::Status(cudaStream_t, bool)> after_replay = nullptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& h : graph_replay_hooks_) {
      if (h.owner == owner) {
        h.before_replay = std::move(before_replay);
        h.after_replay = std::move(after_replay);
        return;
      }
    }
    graph_replay_hooks_.push_back({owner, std::move(before_replay), std::move(after_replay)});
  }

  void UnregisterOffloadNode(const void* owner) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = graph_replay_hooks_.begin(); it != graph_replay_hooks_.end();) {
      if (it->owner == owner) {
        it = graph_replay_hooks_.erase(it);
      } else {
        ++it;
      }
    }
    if (registered_nodes_ > 0) --registered_nodes_;
  }

  void BumpGraphGeneration() noexcept {
    graph_generation_.fetch_add(1, std::memory_order_acq_rel);
  }

  uint64_t GraphGeneration() const noexcept {
    return graph_generation_.load(std::memory_order_acquire);
  }

  common::Status OnBeforeGraphReplay(cudaStream_t stream) {
    std::vector<GraphReplayNodeHook> hooks;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (shutting_down_.load(std::memory_order_acquire)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload session controller is shutting down.");
      }
      hooks = graph_replay_hooks_;
    }
    for (auto& h : hooks) {
      if (h.before_replay) {
        common::Status st = h.before_replay(stream);
        if (!st.IsOK()) return st;
      }
    }
    return common::Status::OK();
  }

  common::Status OnAfterGraphReplay(cudaStream_t stream, bool sync_stream) {
    std::vector<GraphReplayNodeHook> hooks;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      hooks = graph_replay_hooks_;
    }
    common::Status first_err = common::Status::OK();
    for (auto& h : hooks) {
      if (h.after_replay) {
        common::Status st = h.after_replay(stream, sync_stream);
        if (!st.IsOK() && first_err.IsOK()) first_err = st;
      }
    }
    if (sync_stream && stream != nullptr) {
      cudaError_t err = cudaStreamSynchronize(stream);
      if (err != cudaSuccess && first_err.IsOK()) {
        first_err = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload graph replay stream sync failed: ", cudaGetErrorString(err));
      }
    }
    return first_err;
  }

  bool HasOffloadNodes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return registered_nodes_ > 0;
  }

  bool IsShuttingDown() const noexcept {
    return shutting_down_.load(std::memory_order_acquire);
  }

  common::Status OnRunStart() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutting_down_.load(std::memory_order_acquire)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload session controller is shutting down.");
    }
    if (registered_nodes_ == 0) {
      return common::Status::OK();
    }
    if (run_in_flight_) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "QMoEOffload does not support concurrent overlapping InferenceSession::Run calls on the same session.");
    }
    run_in_flight_ = true;
    ++current_run_id_;
    active_invocations_.clear();
    return common::Status::OK();
  }

  void RegisterInvocation(std::shared_ptr<QMoEInvocationRecord> inv) {
    std::lock_guard<std::mutex> lock(mutex_);
    inv->run_id = current_run_id_;
    inv->invocation_id = ++next_invocation_id_;
    active_invocations_.push_back(std::move(inv));
  }

  common::Status OnRunEnd(bool sync_stream, bool run_succeeded = true) {
    std::vector<std::shared_ptr<QMoEInvocationRecord>> invocations;
    bool had_offload_nodes = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      had_offload_nodes = (registered_nodes_ > 0);
      if (!had_offload_nodes && !run_in_flight_) {
        return common::Status::OK();
      }
      run_in_flight_ = false;
      invocations = std::move(active_invocations_);
      active_invocations_.clear();
    }

    common::Status first_error = common::Status::OK();
    if (had_offload_nodes && !sync_stream) {
      const char* strict_env = std::getenv("ORT_QMOE_REJECT_DISABLE_SYNC");
      if (strict_env != nullptr && std::string(strict_env) == "1") {
        first_error = ORT_MAKE_STATUS(
            ONNXRUNTIME, INVALID_ARGUMENT,
            "QMoEOffload does not support disable_synchronize_execution_providers=1; required completion sync cannot be skipped.");
      }
    }
    cudaStream_t last_main_stream = nullptr;
    bool sync_last_main = false;
    for (auto& inv : invocations) {
      if (!inv) continue;
      if (inv->has_main_stream) {
        last_main_stream = inv->main_stream;
        sync_last_main = true;
      }
      if (inv->drain_device) {
        common::Status drain_st = inv->drain_device();
        if (!drain_st.IsOK() && first_error.IsOK()) {
          first_error = drain_st;
        }
      }
      std::lock_guard<std::mutex> inv_lock(inv->mutex);
      if (inv->failed && first_error.IsOK()) {
        first_error = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                      "QMoEOffload invocation failed on node '", inv->node_name,
                                      "' (layer ", inv->layer_index, "): ", inv->error_message);
      }
    }
    if (sync_last_main) {
      cudaError_t err = cudaStreamSynchronize(last_main_stream);
      if (err != cudaSuccess && first_error.IsOK()) {
        first_error = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                      "QMoEOffload compute stream sync failed: ", cudaGetErrorString(err));
      }
    }

    // Whole-run observation atomicity (v3 Section 5.1 & 7.1):
    // Commit staged observations only when the overall run and all invocations succeeded.
    if (run_succeeded && first_error.IsOK()) {
      for (auto& inv : invocations) {
        if (inv && inv->commit_observation) {
          inv->commit_observation();
        }
      }
    }
    return first_error;
  }

  void Shutdown() noexcept {
    shutting_down_.store(true, std::memory_order_release);
    std::vector<std::shared_ptr<QMoEInvocationRecord>> invocations;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      invocations = std::move(active_invocations_);
      active_invocations_.clear();
      run_in_flight_ = false;
    }
    for (auto& inv : invocations) {
      if (inv && inv->drain_device) {
        static_cast<void>(inv->drain_device());
      }
    }
    if (landing_ring_) {
      landing_ring_->Shutdown();
    }
  }

  std::shared_ptr<QMoESessionLandingRing> LandingRing() const { return landing_ring_; }
  int DeviceId() const noexcept { return device_id_; }

 private:
  int device_id_ = 0;
  std::atomic<bool> shutting_down_{false};
  std::atomic<uint64_t> graph_generation_{0};
  mutable std::mutex mutex_;
  size_t registered_nodes_ = 0;
  bool run_in_flight_ = false;
  uint64_t current_run_id_ = 0;
  uint64_t next_invocation_id_ = 0;
  std::vector<std::shared_ptr<QMoEInvocationRecord>> active_invocations_;
  std::vector<GraphReplayNodeHook> graph_replay_hooks_;
  std::shared_ptr<QMoESessionLandingRing> landing_ring_;
};

}  // namespace onnxruntime::cuda
