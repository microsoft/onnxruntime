// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_graph_plugin.h"

#include <algorithm>
#include <cuda_runtime_api.h>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace onnxruntime {
namespace cuda_plugin {

namespace {

void DestroyCudaStreamForDevice(cudaStream_t stream, int device_id) {
  if (stream == nullptr) {
    return;
  }

  int prev_device = -1;
  const bool restore_prev_device = TryGetCurrentCudaDevice(prev_device);
  static_cast<void>(cudaSetDevice(device_id));
  static_cast<void>(cudaStreamDestroy(stream));
  if (restore_prev_device) {
    static_cast<void>(cudaSetDevice(prev_device));
  }
}

cudaStream_t CreateCudaStreamForDevice(int device_id) {
  int prev_device = -1;
  const bool restore_prev_device = TryGetCurrentCudaDevice(prev_device);
  cudaStream_t stream = nullptr;

  try {
    PL_CUDA_CALL_THROW(cudaSetDevice(device_id));
    PL_CUDA_CALL_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (restore_prev_device) {
      PL_CUDA_CALL_THROW(cudaSetDevice(prev_device));
    }
  } catch (...) {
    if (stream != nullptr) {
      DestroyCudaStreamForDevice(stream, device_id);
    }
    if (restore_prev_device) {
      static_cast<void>(cudaSetDevice(prev_device));
    }
    throw;
  }

  return stream;
}

}  // namespace

// --- CudaGraphSet ---

CudaGraphSet::~CudaGraphSet() {
  Clear();
}

void CudaGraphSet::Clear() {
  cuda_graphs_.clear();
}

bool CudaGraphSet::Contains(CudaGraphAnnotation_t id) const {
  return cuda_graphs_.find(id) != cuda_graphs_.end();
}

void CudaGraphSet::Put(CudaGraphAnnotation_t id, cudaGraphExec_t graph_exec) {
  // Take immediate RAII ownership of graph_exec so any exception (duplicate ID, node allocation
  // failure, or bucket rehash bad_alloc) destroys graph_exec_ exactly once without caller catch cleanup.
  OwnedGraphExec owned(graph_exec, destroy_fn_);
  if (Contains(id)) {
    throw std::runtime_error(
        "CudaGraphSet: trying to capture a graph with annotation id " +
        std::to_string(id) + " that is already used. Use a different annotation id.");
  }
  if (inject_put_alloc_failure_for_testing_) {
    throw std::bad_alloc();
  }
  auto [it, inserted] = cuda_graphs_.emplace(id, std::move(owned));
  static_cast<void>(it);
  static_cast<void>(inserted);
}

cudaGraphExec_t CudaGraphSet::Get(CudaGraphAnnotation_t id) const {
  auto it = cuda_graphs_.find(id);
  if (it == cuda_graphs_.end()) {
    throw std::runtime_error(
        "CudaGraphSet: no captured graph for annotation id " + std::to_string(id));
  }
  return it->second.Get();
}

bool CudaGraphSet::Remove(CudaGraphAnnotation_t id, cudaError_t* out_destroy_err) {
  if (out_destroy_err != nullptr) {
    *out_destroy_err = cudaSuccess;
  }
  auto it = cuda_graphs_.find(id);
  if (it == cuda_graphs_.end()) {
    return false;
  }
  const cudaError_t destroy_err = it->second.ReleaseAndDestroy();
  cuda_graphs_.erase(it);
  if (out_destroy_err != nullptr) {
    *out_destroy_err = destroy_err;
  }
  return true;
}

// --- CudaGraphManager ---

CudaGraphManager::CudaGraphManager(cudaStream_t stream,
                                   DestroyGraphExecFn destroy_fn,
                                   SyncStreamFn sync_stream_fn,
                                   QueryStreamFn query_stream_fn)
    : cuda_graph_set_(destroy_fn),
      stream_(stream),
      sync_stream_fn_(sync_stream_fn != nullptr ? sync_stream_fn : &cudaStreamSynchronize),
      query_stream_fn_(query_stream_fn != nullptr ? query_stream_fn : &cudaStreamQuery) {
}

CudaGraphManager::~CudaGraphManager() {
  Reset();
}

void CudaGraphManager::SetStream(cudaStream_t stream) {
  stream_ = stream;
}

void CudaGraphManager::SetRuntimeCallbacksForTesting(
    QueryStreamFn query_stream_fn,
    LaunchGraphFn launch_graph_fn,
    EndCaptureFn end_capture_fn,
    DestroyGraphFn destroy_graph_fn,
    GetLastErrorFn get_last_error_fn) noexcept {
  query_stream_fn_ = query_stream_fn != nullptr ? query_stream_fn : &cudaStreamQuery;
  launch_graph_fn_ = launch_graph_fn != nullptr ? launch_graph_fn : &cudaGraphLaunch;
  end_capture_fn_ = end_capture_fn != nullptr ? end_capture_fn : &cudaStreamEndCapture;
  destroy_graph_fn_ = destroy_graph_fn != nullptr ? destroy_graph_fn : &cudaGraphDestroy;
  get_last_error_fn_ = get_last_error_fn != nullptr ? get_last_error_fn : &cudaGetLastError;
}

void CudaGraphManager::CaptureBegin(CudaGraphAnnotation_t id) {
  if (!IsGraphCaptureAllowedOnRun(id)) {
    throw std::runtime_error("CudaGraphManager: capture not allowed for annotation id " +
                             std::to_string(id));
  }

  if (cuda_graph_set_.Contains(id)) {
    throw std::runtime_error("CudaGraphManager: graph with annotation id " +
                             std::to_string(id) + " already captured");
  }

  PL_CUDA_CALL_THROW(sync_stream_fn_(stream_));
  PL_CUDA_CALL_THROW(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
}

void CudaGraphManager::CaptureEnd(CudaGraphAnnotation_t id) {
  cudaGraph_t graph = nullptr;
  PL_CUDA_CALL_THROW(end_capture_fn_(stream_, &graph));
  if (graph == nullptr) {
    throw std::runtime_error("CudaGraphManager: cudaStreamEndCapture returned NULL graph");
  }

  cudaGraphExec_t graph_exec = nullptr;
  cudaError_t instantiate_err = cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
  cudaError_t destroy_err = destroy_graph_fn_(graph);
  if (instantiate_err != cudaSuccess) {
    if (graph_exec != nullptr) {
      cudaGraphExecDestroy(graph_exec);
    }
    PL_CUDA_CALL_THROW(instantiate_err);
  }
  if (destroy_err != cudaSuccess) {
    cudaGraphExecDestroy(graph_exec);
    PL_CUDA_CALL_THROW(destroy_err);
  }

  cuda_graph_set_.Put(id, graph_exec);
  has_captured_any_ = true;
}

void CudaGraphManager::DiscardCapture() noexcept {
  cudaGraph_t graph = nullptr;
  // A capture invalidated by the failed run reports that here and yields no graph. The stream leaves capture mode
  // either way.
  const cudaError_t end_err = end_capture_fn_(stream_, &graph);
  if (graph != nullptr) {
    static_cast<void>(destroy_graph_fn_(graph));
  }
  if (end_err != cudaSuccess) {
    // Clear the recorded error so it is not attributed to later CUDA calls on this thread.
    static_cast<void>(get_last_error_fn_());
  }
}

OrtStatus* CudaGraphManager::Replay(CudaGraphAnnotation_t id, bool sync) {
  if (!cuda_graph_set_.Contains(id)) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT,
        (std::string("CUDA graph replay error: graph not found for id ") +
         std::to_string(id))
            .c_str());
  }

  cudaGraphExec_t graph_exec = cuda_graph_set_.Get(id);

  cudaError_t err = launch_graph_fn_(graph_exec, stream_);
  if (err != cudaSuccess) {
    return Ort::GetApi().CreateStatus(
        ORT_EP_FAIL,
        (std::string("CUDA graph launch error: ") + cudaGetErrorName(err) +
         ": " + cudaGetErrorString(err))
            .c_str());
  }

  if (sync) {
    err = sync_stream_fn_(stream_);
    if (err != cudaSuccess) {
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL,
          (std::string("CUDA graph sync error: ") + cudaGetErrorName(err) +
           ": " + cudaGetErrorString(err))
              .c_str());
    }
  } else {
    // Flush WDDM command queue so cudaGraphLaunch is submitted to the GPU hardware immediately.
    err = query_stream_fn_(stream_);
    if (err != cudaSuccess && err != cudaErrorNotReady) {
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL,
          (std::string("CUDA graph stream query error: ") + cudaGetErrorName(err) +
           ": " + cudaGetErrorString(err))
              .c_str());
    }
  }

  return nullptr;
}

OrtStatus* CudaGraphManager::Release(CudaGraphAnnotation_t id, bool sync_stream) {
  if (!IsGraphCaptureAllowedOnRun(id)) {
    return nullptr;
  }

  const bool was_captured = cuda_graph_set_.Contains(id);
  const auto run_count_it = run_count_.find(id);
  const bool had_run_count = (run_count_it != run_count_.end());
  if (!was_captured && !had_run_count) {
    return nullptr;
  }

  if (was_captured && sync_stream) {
    cudaError_t err = sync_stream_fn_(stream_);
    if (err != cudaSuccess) {
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL,
          (std::string("CUDA graph release stream sync error: ") + cudaGetErrorName(err) +
           ": " + cudaGetErrorString(err))
              .c_str());
    }
  }

  if (had_run_count) {
    run_count_.erase(run_count_it);
  }
  if (was_captured) {
    cudaError_t destroy_err = cudaSuccess;
    static_cast<void>(cuda_graph_set_.Remove(id, &destroy_err));
    if (destroy_err != cudaSuccess) {
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL,
          (std::string("CUDA graph exec destroy error: ") + cudaGetErrorName(destroy_err) +
           ": " + cudaGetErrorString(destroy_err))
              .c_str());
    }
  }

  return nullptr;
}

bool CudaGraphManager::IsGraphCaptureAllowedOnRun(CudaGraphAnnotation_t id) const {
  return id != kCudaGraphAnnotationSkip;
}

bool CudaGraphManager::IsGraphCaptured(CudaGraphAnnotation_t id) const {
  return cuda_graph_set_.Contains(id);
}

bool CudaGraphManager::HasGraphOrRunCount(CudaGraphAnnotation_t id) const {
  return cuda_graph_set_.Contains(id) || run_count_.find(id) != run_count_.end();
}

void CudaGraphManager::Reset() {
  recapture_immediately_ = false;
  cuda_graph_set_.Clear();
  run_count_.clear();
}

bool CudaGraphManager::IsGraphCaptureAllowed(CudaGraphAnnotation_t id, int min_runs) const {
  if (!IsGraphCaptureAllowedOnRun(id)) {
    return false;
  }
  auto it = run_count_.find(id);
  if (it == run_count_.end()) {
    return false;
  }
  if (has_captured_any_) {
    return it->second >= std::max(1, min_runs);
  }
  return it->second >= min_runs;
}

void CudaGraphManager::IncrementRegularRunCount(CudaGraphAnnotation_t id) {
  if (!IsGraphCaptureAllowedOnRun(id)) {
    return;
  }
  auto it = run_count_.find(id);
  if (it == run_count_.end()) {
    run_count_[id] = 1;
    return;
  }
  it->second++;
}

void CudaGraphManager::PutGraphExecForTesting(CudaGraphAnnotation_t id, cudaGraphExec_t graph_exec) {
  cuda_graph_set_.Put(id, graph_exec);
  has_captured_any_ = true;
}

// --- CudaEpGraphController ---

// When use_external_stream is true (user_compute_stream combined with CUDA graph), capture and
// replay happen on that user-owned stream so they see the same stream as the kernels; the
// context neither creates nor destroys it. Ownership is derived from the caller's intent rather
// than from external_stream being non-null, because a user may legitimately select the CUDA
// default stream (cudaStream_t(0), i.e. nullptr) as the compute stream — that is still an
// external, user-owned stream and must not be destroyed by the context. When use_external_stream
// is false the context creates and owns a dedicated graph stream.
CudaEpGraphController::PerThreadContext::PerThreadContext(
    int device_id, bool use_external_stream,
    cudaStream_t external_stream,
    CudaGraphManager::DestroyGraphExecFn destroy_fn,
    CudaGraphManager::SyncStreamFn sync_stream_fn,
    CudaGraphManager::QueryStreamFn query_stream_fn,
    bool is_host_mock_context)
    : device_id(device_id),
      owns_graph_stream(!use_external_stream),
      graph_stream(use_external_stream ? external_stream
                                       : CreateCudaStreamForDevice(device_id)),
      cuda_graph(graph_stream, destroy_fn, sync_stream_fn, query_stream_fn),
      is_host_mock_context(is_host_mock_context) {
}

CudaEpGraphController::PerThreadContext::~PerThreadContext() {
  // Destroy captured graph execs before destroying the stream they replay on.
  cuda_graph.Reset();
  if (owns_graph_stream) {
    DestroyCudaStreamForDevice(graph_stream, device_id);
  }
  graph_stream = nullptr;
}

struct CudaEpGraphController::PerThreadContextMap {
  mutable std::mutex mutex;
  std::unordered_map<const CudaEpGraphController*, std::shared_ptr<PerThreadContext>> contexts;
};

CudaEpGraphController::CudaEpGraphController(int device_id,
                                             bool enable_cuda_graph,
                                             int min_num_runs_before_cuda_graph_capture,
                                             bool has_user_compute_stream,
                                             void* user_compute_stream) noexcept
    : device_id_(device_id),
      enable_cuda_graph_(enable_cuda_graph),
      min_num_runs_before_cuda_graph_capture_(min_num_runs_before_cuda_graph_capture),
      has_user_compute_stream_(has_user_compute_stream),
      user_compute_stream_(user_compute_stream) {
}

CudaEpGraphController::~CudaEpGraphController() {
  ClearPerThreadContexts();
}

void CudaEpGraphController::ClearPerThreadContexts() noexcept {
  std::vector<std::shared_ptr<PerThreadContext>> removed_contexts;
  {
    std::lock_guard<std::mutex> controller_lock(per_thread_contexts_mutex_);
    removed_contexts.reserve(per_thread_context_caches_.size());
    for (const auto& cache_weak : per_thread_context_caches_) {
      auto cache = cache_weak.lock();
      if (!cache) {
        continue;
      }
      std::lock_guard<std::mutex> cache_lock(cache->mutex);
      auto it = cache->contexts.find(this);
      if (it != cache->contexts.end()) {
        removed_contexts.push_back(std::move(it->second));
        cache->contexts.erase(it);
      }
    }
    per_thread_context_caches_.clear();
  }
}

const std::shared_ptr<CudaEpGraphController::PerThreadContextMap>&
CudaEpGraphController::PerThreadContextCache() {
  thread_local const std::shared_ptr<PerThreadContextMap> per_thread_context_cache =
      std::make_shared<PerThreadContextMap>();
  return per_thread_context_cache;
}

std::shared_ptr<CudaEpGraphController::PerThreadContext>
CudaEpGraphController::FindPerThreadContextForCurrentThread() const noexcept {
  const auto& per_thread_context_cache = PerThreadContextCache();
  std::lock_guard<std::mutex> cache_lock(per_thread_context_cache->mutex);
  auto cached_context_it = per_thread_context_cache->contexts.find(this);
  if (cached_context_it != per_thread_context_cache->contexts.end()) {
    return cached_context_it->second;
  }
  return nullptr;
}

std::shared_ptr<CudaEpGraphController::PerThreadContext>
CudaEpGraphController::GetOrCreatePerThreadContext() const {
  if (std::shared_ptr<PerThreadContext> existing = FindPerThreadContextForCurrentThread();
      existing != nullptr) {
    return existing;
  }

  const auto& per_thread_context_cache = PerThreadContextCache();
  const bool use_external_stream = has_user_compute_stream_ && enable_cuda_graph_;
  cudaStream_t external_stream =
      use_external_stream ? static_cast<cudaStream_t>(user_compute_stream_) : nullptr;
  auto context = std::make_shared<PerThreadContext>(device_id_, use_external_stream, external_stream);
  {
    std::lock_guard<std::mutex> controller_lock(per_thread_contexts_mutex_);
    for (auto it = per_thread_context_caches_.begin(); it != per_thread_context_caches_.end();) {
      if (it->expired()) {
        it = per_thread_context_caches_.erase(it);
      } else {
        ++it;
      }
    }
    ORT_IGNORE_RETURN_VALUE(per_thread_context_caches_.insert(per_thread_context_cache));
    std::lock_guard<std::mutex> cache_lock(per_thread_context_cache->mutex);
    auto insert_result = per_thread_context_cache->contexts.emplace(this, context);
    ORT_ENFORCE(insert_result.second);
  }

  return context;
}

CudaEpGraphController::PerThreadContext& CudaEpGraphController::GetPerThreadContext() const {
  return *GetOrCreatePerThreadContext();
}

CudaEpGraphController::LockedPerThreadContext
CudaEpGraphController::GetLockedPerThreadContext() const {
  return LockedPerThreadContext(GetOrCreatePerThreadContext());
}

cudaStream_t CudaEpGraphController::GetGraphStreamForCurrentThread() const {
  return GetLockedPerThreadContext()->graph_stream;
}

void CudaEpGraphController::RegisterGraphForTestingOnCurrentThread(
    CudaGraphAnnotation_t id,
    cudaGraphExec_t graph_exec,
    int warmup_runs,
    CudaGraphManager::DestroyGraphExecFn destroy_fn,
    CudaGraphManager::SyncStreamFn sync_stream_fn) {
  std::shared_ptr<PerThreadContext> ctx = FindPerThreadContextForCurrentThread();
  if (ctx == nullptr) {
    const auto& per_thread_context_cache = PerThreadContextCache();
    cudaStream_t mock_stream = static_cast<cudaStream_t>(user_compute_stream_);
    auto context = std::make_shared<PerThreadContext>(
        device_id_, /*use_external_stream=*/true, /*external_stream=*/mock_stream,
        destroy_fn, sync_stream_fn, &cudaStreamQuery, /*is_host_mock_context=*/true);
    ctx = context;
    std::lock_guard<std::mutex> controller_lock(per_thread_contexts_mutex_);
    for (auto it = per_thread_context_caches_.begin(); it != per_thread_context_caches_.end();) {
      if (it->expired()) {
        it = per_thread_context_caches_.erase(it);
      } else {
        ++it;
      }
    }
    ORT_IGNORE_RETURN_VALUE(per_thread_context_caches_.insert(per_thread_context_cache));
    std::lock_guard<std::mutex> cache_lock(per_thread_context_cache->mutex);
    auto insert_result = per_thread_context_cache->contexts.emplace(this, std::move(context));
    ORT_ENFORCE(insert_result.second);
  }
  std::lock_guard<std::mutex> ctx_lock(ctx->mutex);
  for (int i = 0; i < warmup_runs; ++i) {
    ctx->cuda_graph.IncrementRegularRunCount(id);
  }
  if (graph_exec != nullptr) {
    ctx->cuda_graph.PutGraphExecForTesting(id, graph_exec);
  }
}

size_t CudaEpGraphController::CapturedGraphCountOnCurrentThreadForTesting() const noexcept {
  std::shared_ptr<PerThreadContext> ctx = FindPerThreadContextForCurrentThread();
  if (ctx == nullptr) {
    return 0;
  }
  std::lock_guard<std::mutex> ctx_lock(ctx->mutex);
  return ctx->cuda_graph.CapturedGraphCount();
}

size_t CudaEpGraphController::RunCountEntryCountOnCurrentThreadForTesting() const noexcept {
  std::shared_ptr<PerThreadContext> ctx = FindPerThreadContextForCurrentThread();
  if (ctx == nullptr) {
    return 0;
  }
  std::lock_guard<std::mutex> ctx_lock(ctx->mutex);
  return ctx->cuda_graph.RunCountEntryCount();
}

OrtStatus* CudaEpGraphController::OnRunStart(CudaGraphAnnotation_t id, uint64_t qmoe_graph_gen) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  if (!enable_cuda_graph_) {
    return nullptr;
  }

  // Recover from any previous failed run on this thread before deciding whether
  // this run will enter capture mode.
  IsThreadCapturingCudaGraph() = false;

  auto& context = GetPerThreadContext();
  std::lock_guard<std::mutex> ctx_lock(context.mutex);
  if (qmoe_graph_gen != context.last_qmoe_graph_gen) {
    context.cuda_graph.Reset();
    context.last_qmoe_graph_gen = qmoe_graph_gen;
  }
  if (!context.cuda_graph.IsGraphCaptured(id) &&
      context.cuda_graph.IsGraphCaptureAllowed(id, min_num_runs_before_cuda_graph_capture_)) {
    // Keep the current CUDA device aligned with the graph stream for the full
    // capture window. Kernel Compute() skips cudaSetDevice() while capturing.
    PL_CUDA_CALL_THROW(cudaSetDevice(device_id_));

    // Record free GPU memory before capture for allocation-during-capture detection.
    context.pre_capture_free_mem = 0;
    size_t free_mem = 0;
    size_t total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
      context.pre_capture_free_mem = free_mem;
    }
    context.cuda_graph.CaptureBegin(id);
    IsThreadCapturingCudaGraph() = true;
  }

  return nullptr;

  EXCEPTION_TO_STATUS_END
}

OrtStatus* CudaEpGraphController::OnRunEnd(
    CudaGraphAnnotation_t id, bool sync_stream,
    bool* out_allocated_during_capture,
    bool run_failed) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  if (out_allocated_during_capture != nullptr) {
    *out_allocated_during_capture = false;
  }
  if (!enable_cuda_graph_) {
    return nullptr;
  }

  // Always clear the flag before leaving this callback so a failed capture or
  // failed replay cannot poison later runs on the same thread.
  IsThreadCapturingCudaGraph() = false;

  auto& context = GetPerThreadContext();
  bool replayed_graph = false;
  {
    std::lock_guard<std::mutex> ctx_lock(context.mutex);
    if (!context.cuda_graph.IsGraphCaptured(id)) {
      const bool capture_allowed =
          context.cuda_graph.IsGraphCaptureAllowed(id, min_num_runs_before_cuda_graph_capture_);
      if (capture_allowed && run_failed) {
        // What was recorded is incomplete: drop it instead of instantiating and replaying it. The warm-up count is
        // kept so the next successful run captures again.
        context.cuda_graph.DiscardCapture();
        context.pre_capture_free_mem = 0;
      } else if (capture_allowed) {
        context.cuda_graph.CaptureEnd(id);

        // Check if GPU memory was allocated during capture (which would make the
        // captured graph invalid since CUDA graph replay cannot reproduce allocations).
        if (context.pre_capture_free_mem > 0) {
          size_t post_free_mem = 0;
          size_t total_mem = 0;
          if (cudaMemGetInfo(&post_free_mem, &total_mem) == cudaSuccess) {
            if (post_free_mem < context.pre_capture_free_mem && out_allocated_during_capture != nullptr) {
              *out_allocated_during_capture = true;
            }
          }
          context.pre_capture_free_mem = 0;
        }

        // CUDA work issued to a capturing stream doesn't actually run on the GPU,
        // so replay the captured graph to actually execute the work.
        OrtStatus* status = context.cuda_graph.Replay(id, sync_stream);
        if (status != nullptr) {
          return status;
        }
        replayed_graph = true;
      } else if (!run_failed) {
        // A failed warm-up run did not exercise the whole graph, so it does not count towards the runs required
        // before capture.
        context.cuda_graph.IncrementRegularRunCount(id);
      }
    }
  }

  if (sync_stream && !replayed_graph) {
    PL_CUDA_RETURN_IF_ERROR(context.cuda_graph.SyncStream());
  } else if (!sync_stream) {
    cudaError_t query_err = context.cuda_graph.QueryStream();
    if (query_err != cudaSuccess && query_err != cudaErrorNotReady) {
      PL_CUDA_RETURN_IF_ERROR(query_err);
    }
  }
  return nullptr;

  EXCEPTION_TO_STATUS_END
}

bool CudaEpGraphController::IsGraphCaptured(CudaGraphAnnotation_t id, uint64_t qmoe_graph_gen) const noexcept {
  if (!enable_cuda_graph_) {
    return false;
  }

  try {
    auto& context = GetPerThreadContext();
    std::lock_guard<std::mutex> ctx_lock(context.mutex);
    if (qmoe_graph_gen != context.last_qmoe_graph_gen) {
      context.cuda_graph.Reset();
      context.last_qmoe_graph_gen = qmoe_graph_gen;
    }
    return context.cuda_graph.IsGraphCaptured(id);
  } catch (...) {
    return false;
  }
}

OrtStatus* CudaEpGraphController::ReplayGraph(CudaGraphAnnotation_t id, uint64_t qmoe_graph_gen) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  if (!enable_cuda_graph_) {
    return Ort::GetApi().CreateStatus(
        ORT_EP_FAIL, "ReplayGraph called but CUDA graph manager is not initialized");
  }
  auto& context = GetPerThreadContext();
  std::lock_guard<std::mutex> ctx_lock(context.mutex);
  if (!context.is_host_mock_context) {
    PL_CUDA_CALL_THROW(cudaSetDevice(device_id_));
  }
  if (qmoe_graph_gen != context.last_qmoe_graph_gen) {
    context.cuda_graph.Reset();
    context.last_qmoe_graph_gen = qmoe_graph_gen;
    return Ort::GetApi().CreateStatus(
        ORT_FAIL, "QMoEOffload captured CUDA graph was invalidated by state reclamation; recapture required.");
  }
  // Launch graph without sync. The caller (PluginExecutionProvider::ReplayGraph)
  // handles synchronization based on disable_synchronize_execution_providers.
  return context.cuda_graph.Replay(id, /*sync=*/false);

  EXCEPTION_TO_STATUS_END
}

OrtStatus* CudaEpGraphController::ReleaseCapturedGraph(CudaGraphAnnotation_t graph_annotation_id) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  if (!enable_cuda_graph_ || graph_annotation_id == kCudaGraphAnnotationSkip) {
    return nullptr;
  }

  if (IsThreadCapturingCudaGraph()) {
    return Ort::GetApi().CreateStatus(
        ORT_EP_FAIL,
        (std::string("Cannot release CUDA graph annotation id ") +
         std::to_string(graph_annotation_id) +
         " while stream capture is in progress on the current thread.")
            .c_str());
  }

  std::shared_ptr<PerThreadContext> current_context = FindPerThreadContextForCurrentThread();
  if (current_context != nullptr) {
    std::lock_guard<std::mutex> ctx_lock(current_context->mutex);
    if (current_context->cuda_graph.HasGraphOrRunCount(graph_annotation_id)) {
      const bool was_captured = current_context->cuda_graph.IsGraphCaptured(graph_annotation_id);
      const bool manage_cuda_device = was_captured && !current_context->is_host_mock_context;
      int prev_device = -1;
      const bool restore_prev_device = manage_cuda_device && TryGetCurrentCudaDevice(prev_device);
      if (manage_cuda_device) {
        OrtStatus* set_dev_status = StatusFromCudaError(cudaSetDevice(device_id_)).release();
        if (set_dev_status != nullptr) {
          return set_dev_status;
        }
      }

      OrtStatus* release_status = current_context->cuda_graph.Release(
          graph_annotation_id, /*sync_stream=*/was_captured);

      if (restore_prev_device && prev_device != device_id_) {
        OrtStatus* restore_status = StatusFromCudaError(cudaSetDevice(prev_device)).release();
        if (release_status == nullptr) {
          release_status = restore_status;
        } else if (restore_status != nullptr) {
          Ort::GetApi().ReleaseStatus(restore_status);
        }
      }
      return release_status;
    }
  }

  // Not found in the calling thread's context. Collect strong references to this controller's
  // contexts in other threads' caches under Level 1 (per_thread_contexts_mutex_) -> Level 2
  // (cache->mutex), then inspect each context under Level 3 (other_ctx->mutex) after releasing
  // the cache and controller locks.
  std::vector<std::shared_ptr<PerThreadContext>> other_contexts;
  {
    std::lock_guard<std::mutex> controller_lock(per_thread_contexts_mutex_);
    other_contexts.reserve(per_thread_context_caches_.size());
    for (auto it_cache = per_thread_context_caches_.begin();
         it_cache != per_thread_context_caches_.end();) {
      auto cache = it_cache->lock();
      if (!cache) {
        it_cache = per_thread_context_caches_.erase(it_cache);
        continue;
      }
      {
        std::lock_guard<std::mutex> cache_lock(cache->mutex);
        auto it_ctx = cache->contexts.find(this);
        if (it_ctx != cache->contexts.end() && it_ctx->second != nullptr &&
            it_ctx->second.get() != current_context.get()) {
          other_contexts.push_back(it_ctx->second);
        }
      }
      ++it_cache;
    }
  }

  for (const auto& other_ctx : other_contexts) {
    std::lock_guard<std::mutex> other_lock(other_ctx->mutex);
    if (other_ctx->cuda_graph.HasGraphOrRunCount(graph_annotation_id)) {
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL,
          (std::string("CUDA graph annotation id ") +
           std::to_string(graph_annotation_id) +
           " is owned by another thread's CUDA graph context and must be released on the capturing thread.")
              .c_str());
    }
  }

  return nullptr;

  EXCEPTION_TO_STATUS_END
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
