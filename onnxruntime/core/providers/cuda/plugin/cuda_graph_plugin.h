// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Plugin-side CUDA graph manager. Manages cudaGraph_t / cudaGraphExec_t lifecycle
// for CUDA graph capture, replay, and retirement in the plugin EP. This is a
// standalone port of the bundled CUDAGraphManager (cuda_graph.h) without
// framework dependencies.

#pragma once

#include "cuda_plugin_utils.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>

namespace onnxruntime {
namespace cuda_plugin {

using CudaGraphAnnotation_t = int;

constexpr CudaGraphAnnotation_t kCudaGraphAnnotationSkip = -1;
constexpr CudaGraphAnnotation_t kCudaGraphAnnotationDefault = 0;

/// Storage for captured CUDA graph executables, keyed by annotation ID.
class CudaGraphSet {
 public:
  using DestroyGraphExecFn = cudaError_t (*)(cudaGraphExec_t);

  /// Move-only RAII owner for an instantiated cudaGraphExec_t handle.
  /// Erasing an entry from CudaGraphSet::cuda_graphs_ invokes ~OwnedGraphExec()
  /// or ReleaseAndDestroy() at most once, preventing double destruction during
  /// subsequent Clear() / teardown.
  class OwnedGraphExec {
   public:
    OwnedGraphExec() = default;
    explicit OwnedGraphExec(cudaGraphExec_t graph_exec,
                            DestroyGraphExecFn destroy_fn = &cudaGraphExecDestroy) noexcept
        : graph_exec_(graph_exec),
          destroy_fn_(destroy_fn != nullptr ? destroy_fn : &cudaGraphExecDestroy) {}

    ~OwnedGraphExec() {
      if (graph_exec_ != nullptr) {
        static_cast<void>(destroy_fn_(graph_exec_));
      }
    }

    OwnedGraphExec(OwnedGraphExec&& other) noexcept
        : graph_exec_(other.graph_exec_), destroy_fn_(other.destroy_fn_) {
      other.graph_exec_ = nullptr;
    }

    OwnedGraphExec& operator=(OwnedGraphExec&& other) noexcept {
      if (this != &other) {
        if (graph_exec_ != nullptr) {
          static_cast<void>(destroy_fn_(graph_exec_));
        }
        graph_exec_ = other.graph_exec_;
        destroy_fn_ = other.destroy_fn_;
        other.graph_exec_ = nullptr;
      }
      return *this;
    }

    OwnedGraphExec(const OwnedGraphExec&) = delete;
    OwnedGraphExec& operator=(const OwnedGraphExec&) = delete;

    cudaGraphExec_t Get() const noexcept { return graph_exec_; }

    cudaError_t ReleaseAndDestroy() noexcept {
      if (graph_exec_ == nullptr) {
        return cudaSuccess;
      }
      cudaGraphExec_t handle = graph_exec_;
      graph_exec_ = nullptr;
      return destroy_fn_(handle);
    }

   private:
    cudaGraphExec_t graph_exec_ = nullptr;
    DestroyGraphExecFn destroy_fn_ = &cudaGraphExecDestroy;
  };

  CudaGraphSet() = default;
  explicit CudaGraphSet(DestroyGraphExecFn destroy_fn) noexcept
      : destroy_fn_(destroy_fn != nullptr ? destroy_fn : &cudaGraphExecDestroy) {}
  ~CudaGraphSet();

  CudaGraphSet(const CudaGraphSet&) = delete;
  CudaGraphSet& operator=(const CudaGraphSet&) = delete;

  void Clear();
  bool Contains(CudaGraphAnnotation_t id) const;
  void Put(CudaGraphAnnotation_t id, cudaGraphExec_t graph_exec);
  cudaGraphExec_t Get(CudaGraphAnnotation_t id) const;
  bool Remove(CudaGraphAnnotation_t id, cudaError_t* out_destroy_err = nullptr);
  size_t Size() const noexcept { return cuda_graphs_.size(); }
  void SetInjectPutAllocFailureForTesting(bool enable) noexcept {
    inject_put_alloc_failure_for_testing_ = enable;
  }

 private:
  std::unordered_map<CudaGraphAnnotation_t, OwnedGraphExec> cuda_graphs_;
  DestroyGraphExecFn destroy_fn_ = &cudaGraphExecDestroy;
  bool inject_put_alloc_failure_for_testing_ = false;
};

/// Orchestrates CUDA graph capture, instantiation, replay, and retirement.
class CudaGraphManager {
 public:
  using SyncStreamFn = cudaError_t (*)(cudaStream_t);
  using QueryStreamFn = cudaError_t (*)(cudaStream_t);
  using LaunchGraphFn = cudaError_t (*)(cudaGraphExec_t, cudaStream_t);
  using EndCaptureFn = cudaError_t (*)(cudaStream_t, cudaGraph_t*);
  using DestroyGraphFn = cudaError_t (*)(cudaGraph_t);
  using GetLastErrorFn = cudaError_t (*)();
  using DestroyGraphExecFn = CudaGraphSet::DestroyGraphExecFn;

  CudaGraphManager() = default;
  explicit CudaGraphManager(cudaStream_t stream,
                            DestroyGraphExecFn destroy_fn = &cudaGraphExecDestroy,
                            SyncStreamFn sync_stream_fn = &cudaStreamSynchronize,
                            QueryStreamFn query_stream_fn = &cudaStreamQuery);
  ~CudaGraphManager();

  CudaGraphManager(const CudaGraphManager&) = delete;
  CudaGraphManager& operator=(const CudaGraphManager&) = delete;

  void SetStream(cudaStream_t stream);
  cudaError_t SyncStream() { return sync_stream_fn_(stream_); }
  cudaError_t QueryStream() { return query_stream_fn_(stream_); }

  /// Begin capturing CUDA operations on the stream.
  void CaptureBegin(CudaGraphAnnotation_t id);

  /// End capture, instantiate the graph, and store the executable.
  void CaptureEnd(CudaGraphAnnotation_t id);

  /// End the in-progress capture and drop what was recorded, without instantiating or storing a graph.
  /// For a run that failed while being captured, where the recorded work is incomplete.
  void DiscardCapture() noexcept;

  /// Launch a previously captured graph. Returns OrtStatus on error.
  OrtStatus* Replay(CudaGraphAnnotation_t id, bool sync = true);

  /// Release a previously captured graph and its per-ID warm-up bookkeeping.
  /// No-op if the annotation ID was never captured or has already been released.
  OrtStatus* Release(CudaGraphAnnotation_t id, bool sync_stream = true);

  /// Returns false if the annotation ID indicates capture should be skipped.
  bool IsGraphCaptureAllowedOnRun(CudaGraphAnnotation_t id) const;

  /// Returns true if a graph has been captured for the given annotation ID.
  bool IsGraphCaptured(CudaGraphAnnotation_t id) const;

  /// Returns true if a captured graph or warm-up run counter exists for the given annotation ID.
  bool HasGraphOrRunCount(CudaGraphAnnotation_t id) const;

  /// Destroy all captured graph executables.
  void Reset();

  /// Track warm-up runs before allowing graph capture.
  /// Returns true when enough warm-up runs have occurred for the given annotation ID.
  bool IsGraphCaptureAllowed(CudaGraphAnnotation_t id, int min_runs) const;

  /// Increment the warm-up run count for a given annotation ID.
  void IncrementRegularRunCount(CudaGraphAnnotation_t id);

  /// Host-test seams to exercise graph retirement, DiscardCapture, and async Replay on CPU.
  void PutGraphExecForTesting(CudaGraphAnnotation_t id, cudaGraphExec_t graph_exec);
  void SetInjectPutAllocFailureForTesting(bool enable) noexcept {
    cuda_graph_set_.SetInjectPutAllocFailureForTesting(enable);
  }
  void SetRuntimeCallbacksForTesting(QueryStreamFn query_stream_fn,
                                     LaunchGraphFn launch_graph_fn = nullptr,
                                     EndCaptureFn end_capture_fn = nullptr,
                                     DestroyGraphFn destroy_graph_fn = nullptr,
                                     GetLastErrorFn get_last_error_fn = nullptr) noexcept;

  size_t CapturedGraphCount() const noexcept { return cuda_graph_set_.Size(); }
  size_t RunCountEntryCount() const noexcept { return run_count_.size(); }
  bool HasCapturedAny() const noexcept { return has_captured_any_; }

 private:
  CudaGraphSet cuda_graph_set_;
  cudaStream_t stream_ = nullptr;  // Does not own the stream.
  SyncStreamFn sync_stream_fn_ = &cudaStreamSynchronize;
  QueryStreamFn query_stream_fn_ = &cudaStreamQuery;
  LaunchGraphFn launch_graph_fn_ = &cudaGraphLaunch;
  EndCaptureFn end_capture_fn_ = &cudaStreamEndCapture;
  DestroyGraphFn destroy_graph_fn_ = &cudaGraphDestroy;
  GetLastErrorFn get_last_error_fn_ = &cudaGetLastError;

  /// Tracks the number of regular (non-captured) runs per annotation ID.
  /// Graph capture is deferred until the count reaches `min_num_runs_before_cuda_graph_capture`.
  std::unordered_map<CudaGraphAnnotation_t, int> run_count_;
  bool has_captured_any_ = false;
  bool recapture_immediately_ = false;
};

/// Manages per-thread CUDA graph contexts, streams, and cross-thread ownership checks for a CudaEp instance.
class CudaEpGraphController {
 public:
  struct PerThreadContext {
    explicit PerThreadContext(
        int device_id, bool use_external_stream = false,
        cudaStream_t external_stream = nullptr,
        CudaGraphManager::DestroyGraphExecFn destroy_fn = &cudaGraphExecDestroy,
        CudaGraphManager::SyncStreamFn sync_stream_fn = &cudaStreamSynchronize,
        CudaGraphManager::QueryStreamFn query_stream_fn = &cudaStreamQuery,
        bool is_host_mock_context = false);
    ~PerThreadContext();

    int device_id;
    bool owns_graph_stream;
    cudaStream_t graph_stream = nullptr;
    CudaGraphManager cuda_graph;
    size_t pre_capture_free_mem = 0;
    uint64_t last_qmoe_graph_gen = 0;
    bool is_host_mock_context = false;
    mutable std::mutex mutex;
  };

  class LockedPerThreadContext {
   public:
    explicit LockedPerThreadContext(std::shared_ptr<PerThreadContext> context)
        : context_(std::move(context)), lock_(context_->mutex) {}
    LockedPerThreadContext(LockedPerThreadContext&&) noexcept = default;
    LockedPerThreadContext& operator=(LockedPerThreadContext&&) = delete;
    LockedPerThreadContext(const LockedPerThreadContext&) = delete;
    LockedPerThreadContext& operator=(const LockedPerThreadContext&) = delete;

    PerThreadContext& operator*() const noexcept { return *context_; }
    PerThreadContext* operator->() const noexcept { return context_.get(); }

   private:
    // context_ is declared before lock_ so ~LockedPerThreadContext() destroys lock_
    // (unlocking context_->mutex) before releasing the last shared_ptr reference to context_.
    // Move assignment is deleted to prevent memberwise assignment from overwriting context_
    // while lock_ still holds the old context's mutex.
    std::shared_ptr<PerThreadContext> context_;
    std::unique_lock<std::mutex> lock_;
  };

  CudaEpGraphController(int device_id,
                        bool enable_cuda_graph,
                        int min_num_runs_before_cuda_graph_capture,
                        bool has_user_compute_stream,
                        void* user_compute_stream) noexcept;
  ~CudaEpGraphController();

  CudaEpGraphController(const CudaEpGraphController&) = delete;
  CudaEpGraphController& operator=(const CudaEpGraphController&) = delete;

  bool IsGraphCaptureEnabled() const noexcept { return enable_cuda_graph_; }
  cudaStream_t GetGraphStreamForCurrentThread() const;
  LockedPerThreadContext GetLockedPerThreadContext() const;
  void ClearPerThreadContexts() noexcept;

  OrtStatus* OnRunStart(CudaGraphAnnotation_t id, uint64_t qmoe_graph_gen = 0) noexcept;
  OrtStatus* OnRunEnd(CudaGraphAnnotation_t id, bool sync_stream,
                      bool* out_allocated_during_capture = nullptr,
                      bool run_failed = false) noexcept;
  bool IsGraphCaptured(CudaGraphAnnotation_t id, uint64_t qmoe_graph_gen = 0) const noexcept;
  OrtStatus* ReplayGraph(CudaGraphAnnotation_t id, uint64_t qmoe_graph_gen = 0) noexcept;
  OrtStatus* ReleaseCapturedGraph(CudaGraphAnnotation_t id) noexcept;

  /// Host-test helpers for exercising per-thread graph ownership and ReleaseCapturedGraph on CPU.
  void RegisterGraphForTestingOnCurrentThread(
      CudaGraphAnnotation_t id,
      cudaGraphExec_t graph_exec,
      int warmup_runs,
      CudaGraphManager::DestroyGraphExecFn destroy_fn,
      CudaGraphManager::SyncStreamFn sync_stream_fn);
  bool HasPerThreadContextForCurrentThreadForTesting() const noexcept {
    return FindPerThreadContextForCurrentThread() != nullptr;
  }
  size_t CapturedGraphCountOnCurrentThreadForTesting() const noexcept;
  size_t RunCountEntryCountOnCurrentThreadForTesting() const noexcept;

 private:
  struct PerThreadContextMap;

  static const std::shared_ptr<PerThreadContextMap>& PerThreadContextCache();
  std::shared_ptr<PerThreadContext> FindPerThreadContextForCurrentThread() const noexcept;
  std::shared_ptr<PerThreadContext> GetOrCreatePerThreadContext() const;
  PerThreadContext& GetPerThreadContext() const;

  int device_id_ = 0;
  bool enable_cuda_graph_ = false;
  int min_num_runs_before_cuda_graph_capture_ = 2;
  bool has_user_compute_stream_ = false;
  void* user_compute_stream_ = nullptr;

  // Lock ordering across the controller registry, shared per-thread cache, and context:
  //   Level 1: per_thread_contexts_mutex_ (guards per_thread_context_caches_)
  //   Level 2: PerThreadContextMap::mutex (guards PerThreadContextMap::contexts)
  //   Level 3: PerThreadContext::mutex    (guards PerThreadContext::cuda_graph)
  // Nested acquisitions only ever proceed Level 1 -> Level 2; Level 3 is acquired after
  // releasing Level 1 and Level 2 while holding a std::shared_ptr<PerThreadContext>.
  mutable std::mutex per_thread_contexts_mutex_;
  // The thread-local cache owns contexts so they are released when a thread exits.
  // The controller tracks live caches only to remove its entry when the EP is destroyed.
  mutable std::set<std::weak_ptr<PerThreadContextMap>, std::owner_less<std::weak_ptr<PerThreadContextMap>>>
      per_thread_context_caches_;
};

}  // namespace cuda_plugin
}  // namespace onnxruntime
