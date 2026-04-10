// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_stream_plugin.h"
#include "cuda_ep_factory.h"
#include <atomic>
#include <mutex>
#include <shared_mutex>

namespace onnxruntime {
namespace cuda_plugin {

namespace {

// Global stream-to-CudaSyncStream mapping.
// Required because migrated CUDA kernels receive only a raw cudaStream_t
// but need access to associated cuBLAS/cuDNN handles.
using StreamMap = std::unordered_map<cudaStream_t, CudaSyncStream*>;

StreamMap& GetStreamMap() {
  static StreamMap stream_map;
  return stream_map;
}

std::shared_mutex& GetStreamMapMutex() {
  static std::shared_mutex stream_map_mutex;
  return stream_map_mutex;
}

// Monotonically increasing generation counter, bumped on every UnregisterStream
// so that TLS caches can detect stale entries without acquiring a lock.
std::atomic<uint64_t>& GetStreamMapGeneration() {
  static std::atomic<uint64_t> generation{0};
  return generation;
}
}  // namespace

// ---------------------------------------------------------------------------
// CudaSyncStream
// ---------------------------------------------------------------------------

CudaSyncStream::CudaSyncStream(CudaEpFactory& factory, int device_id,
                               const OrtEp* /*ep*/)
    : OrtSyncStreamImpl{},
      factory_(factory),
      device_id_(device_id) {
  ort_version_supported = ORT_API_VERSION;
  GetHandle = GetHandleImpl;
  CreateNotification = CreateNotificationImpl;
  Flush = FlushImpl;
  OnSessionRunEnd = OnSessionRunEndImpl;
  Release = ReleaseImpl;
}

CudaSyncStream::~CudaSyncStream() {
  bool has_deferred_cpu_buffers = false;
  {
    std::lock_guard<std::mutex> lock(deferred_cpu_buffers_mutex_);
    has_deferred_cpu_buffers = !deferred_cpu_buffers_.empty();
  }

  if (has_deferred_cpu_buffers) {
    if (cuda_stream_ != nullptr) {
      OrtStatus* status = OnSessionRunEndImpl(this);
      if (status != nullptr) {
        Ort::GetApi().ReleaseStatus(status);
      }
    } else {
      OrtStatus* status = CleanupDeferredCPUBuffers();
      if (status != nullptr) {
        Ort::GetApi().ReleaseStatus(status);
      }
    }
  }

  if (cublas_handle_) cublasDestroy(cublas_handle_);
  if (cudnn_handle_) cudnnDestroy(cudnn_handle_);
  if (cublas_lt_handle_) cublasLtDestroy(cublas_lt_handle_);
  if (cuda_stream_) {
    // Unregister the stream from the global map *after* destroying handles but
    // *before* destroying the stream itself. This ordering ensures:
    //   1. No concurrent kernel can obtain cuBLAS/cuDNN handles from a destroyed
    //      CudaSyncStream during the brief window before unregistration.
    //   2. UnregisterStream bumps the TLS generation counter, invalidating cached
    //      lookups in other threads.
    //   3. The stream is destroyed only after it is no longer discoverable.
    // Only unregister if the stream was actually registered (InitHandles
    // succeeded fully). Otherwise we'd bump the global generation counter
    // for a stream that was never in the map, causing unnecessary TLS
    // invalidations in other threads.
    if (registered_) {
      UnregisterStream(cuda_stream_);
    }

    if (owns_stream_) {
      auto destroy_result = cudaStreamDestroy(cuda_stream_);
      if (destroy_result == cudaSuccess && !deferred_cpu_buffers_.empty()) {
        // Fallback: we only reach here when the earlier cudaStreamSynchronize in
        // OnSessionRunEndImpl failed, leaving some buffers un-freed.
        // cudaStreamDestroy on a non-blocking stream returns immediately (async
        // cleanup), so in-flight ops may still reference these buffers. However,
        // a prior sync failure indicates a serious CUDA error, so best-effort
        // cleanup is the most we can do here.
        OrtStatus* status = CleanupDeferredCPUBuffers();
        if (status != nullptr) {
          Ort::GetApi().ReleaseStatus(status);
        }
      }
    }  // else: external stream — do NOT destroy it.
  }
}

OrtStatus* CudaSyncStream::InitHandles() {
  int prev_device = -1;
  const bool restore_prev_device = TryGetCurrentCudaDevice(prev_device);

  Ort::Status status = StatusFromCudaError(cudaSetDevice(device_id_));
  if (status.IsOK()) {
    status = StatusFromCudaError(cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking));
  }
  if (status.IsOK()) {
    status = StatusFromCublasError(cublasCreate(&cublas_handle_));
  }
  if (status.IsOK()) {
    status = StatusFromCublasError(cublasSetStream(cublas_handle_, cuda_stream_));
  }
  if (status.IsOK()) {
    status = StatusFromCudnnError(cudnnCreate(&cudnn_handle_));
  }
  if (status.IsOK()) {
    status = StatusFromCudnnError(cudnnSetStream(cudnn_handle_, cuda_stream_));
  }
  if (status.IsOK()) {
    status = StatusFromCublasError(cublasLtCreate(&cublas_lt_handle_));
  }

  if (restore_prev_device) {
    Ort::Status restore_status = StatusFromCudaError(cudaSetDevice(prev_device));
    if (status.IsOK()) {
      status = std::move(restore_status);
    }
  }

  if (status.IsOK()) {
    RegisterStream(cuda_stream_, this);
    registered_ = true;
  }

  return status.release();
}

OrtStatus* CudaSyncStream::InitHandlesWithExternalStream(cudaStream_t external_stream) {
  int prev_device = -1;
  const bool restore_prev_device = TryGetCurrentCudaDevice(prev_device);

  Ort::Status status = StatusFromCudaError(cudaSetDevice(device_id_));
  if (status.IsOK()) {
    // Graph-mode wrappers only need to publish the raw stream identity. CUDA
    // library handles fall back to per-thread defaults at kernel dispatch time.
    cuda_stream_ = external_stream;
    owns_stream_ = false;
  }

  if (restore_prev_device) {
    Ort::Status restore_status = StatusFromCudaError(cudaSetDevice(prev_device));
    if (status.IsOK()) {
      status = std::move(restore_status);
    }
  }

  if (status.IsOK()) {
    RegisterStream(cuda_stream_, this);
    registered_ = true;
  }

  return status.release();
}

void CudaSyncStream::EnqueueDeferredCPUBuffer(void* cpu_buffer) {
  std::lock_guard<std::mutex> lock(deferred_cpu_buffers_mutex_);
  deferred_cpu_buffers_.push_back(cpu_buffer);
}

OrtStatus* CudaSyncStream::CleanupDeferredCPUBuffers() noexcept {
  std::vector<void*> buffers_to_free;
  {
    std::lock_guard<std::mutex> lock(deferred_cpu_buffers_mutex_);
    buffers_to_free.swap(deferred_cpu_buffers_);
  }

  OrtStatus* first_error = nullptr;
  for (void* buf : buffers_to_free) {
    cudaError_t err = cudaFreeHost(buf);
    if (err != cudaSuccess && first_error == nullptr) {
      first_error = Ort::GetApi().CreateStatus(
          ORT_EP_FAIL,
          (std::string("CUDA error: ") + cudaGetErrorName(err) + ": " + cudaGetErrorString(err)).c_str());
    }
  }
  return first_error;
}

/*static*/ void* ORT_API_CALL CudaSyncStream::GetHandleImpl(OrtSyncStreamImpl* this_ptr) noexcept {
  auto* stream = static_cast<CudaSyncStream*>(this_ptr);
  return stream->cuda_stream_;
}

/*static*/ OrtStatus* ORT_API_CALL CudaSyncStream::CreateNotificationImpl(
    OrtSyncStreamImpl* this_ptr, OrtSyncNotificationImpl** notification) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  auto* stream = static_cast<CudaSyncStream*>(this_ptr);
  auto notif = std::make_unique<CudaSyncNotification>(*stream);
  *notification = notif.release();
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

/*static*/ OrtStatus* ORT_API_CALL CudaSyncStream::FlushImpl(OrtSyncStreamImpl* this_ptr) noexcept {
  auto* stream = static_cast<CudaSyncStream*>(this_ptr);

  // During CUDA graph capture, cudaStreamSynchronize is not permitted on a
  // capturing stream. Skip the synchronize − the captured graph preserves
  // kernel ordering and the stream will be synchronized after capture ends.
  if (!stream->owns_stream_) {
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cudaError_t query_err = cudaStreamIsCapturing(stream->cuda_stream_, &capture_status);
    if (query_err == cudaSuccess && capture_status == cudaStreamCaptureStatusActive) {
      return nullptr;
    }
  }

  PL_CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream->cuda_stream_));
  return nullptr;
}

/*static*/ OrtStatus* ORT_API_CALL CudaSyncStream::OnSessionRunEndImpl(OrtSyncStreamImpl* this_ptr) noexcept {
  auto* stream = static_cast<CudaSyncStream*>(this_ptr);
  if (stream->cuda_stream_ == nullptr) {
    return stream->CleanupDeferredCPUBuffers();
  }
  // Synchronize before releasing deferred CPU buffers to ensure
  // all async copies using those buffers have completed.
  PL_CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream->cuda_stream_));

  // Reset arena chunk-to-stream assignments for this device's current arena.
  // Uses ResetDeviceArenaChunksUsingStream to hold the arena_mutex across the
  // entire operation, preventing a concurrent ReleaseAllocatorImpl from destroying
  // the arena while we hold a raw pointer to it.
  {
    OrtStatus* arena_status = stream->factory_.ResetDeviceArenaChunksUsingStream(
        stream->device_id_, this_ptr);
    if (arena_status != nullptr) {
      // Ignore the arena reset error and continue session run end — buffer cleanup is more critical.
      Ort::GetApi().ReleaseStatus(arena_status);
    }
  }

  return stream->CleanupDeferredCPUBuffers();
}

/*static*/ void ORT_API_CALL CudaSyncStream::ReleaseImpl(OrtSyncStreamImpl* this_ptr) noexcept {
  delete static_cast<CudaSyncStream*>(this_ptr);
}

/*static*/ CudaSyncStream* CudaSyncStream::FromCudaStream(cudaStream_t stream) {
  if (stream == nullptr) {
    return nullptr;
  }

  // Thread-local TLS cache to mitigate lock contention on the hot path.
  // The generation counter is bumped on every UnregisterStream() so that
  // stale TLS entries (pointing to destroyed CudaSyncStream objects) are
  // automatically invalidated without requiring per-thread notification.
  thread_local cudaStream_t tls_last_stream = nullptr;
  thread_local CudaSyncStream* tls_last_sync_stream = nullptr;
  thread_local uint64_t tls_generation = 0;

  uint64_t current_gen = GetStreamMapGeneration().load(std::memory_order_acquire);
  if (stream == tls_last_stream && tls_generation == current_gen) {
    return tls_last_sync_stream;
  }

  auto& stream_map = GetStreamMap();
  std::shared_lock<std::shared_mutex> lock(GetStreamMapMutex());
  auto it = stream_map.find(stream);
  if (it != stream_map.end()) {
    tls_last_stream = stream;
    tls_last_sync_stream = it->second;
    tls_generation = current_gen;
    return it->second;
  }
  return nullptr;
}

/*static*/ void CudaSyncStream::RegisterStream(cudaStream_t stream, CudaSyncStream* sync_stream) {
  auto& stream_map = GetStreamMap();
  std::unique_lock<std::shared_mutex> lock(GetStreamMapMutex());
  stream_map[stream] = sync_stream;
}

/*static*/ void CudaSyncStream::UnregisterStream(cudaStream_t stream) {
  auto& stream_map = GetStreamMap();
  std::unique_lock<std::shared_mutex> lock(GetStreamMapMutex());
  stream_map.erase(stream);
  // Bump generation so TLS caches in other threads are invalidated.
  GetStreamMapGeneration().fetch_add(1, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// CudaSyncNotification
// ---------------------------------------------------------------------------

CudaSyncNotification::CudaSyncNotification(CudaSyncStream& stream)
    : OrtSyncNotificationImpl{},
      stream_(stream) {
  ort_version_supported = ORT_API_VERSION;
  Activate = ActivateImpl;
  WaitOnDevice = WaitOnDeviceImpl;
  WaitOnHost = WaitOnHostImpl;
  Release = ReleaseImpl;

  // Create a CUDA event for synchronization (disable timing for performance)
  int prev_device = -1;
  const bool restore_prev_device = TryGetCurrentCudaDevice(prev_device);
  const auto restore_prev_device_status = [&]() {
    if (!restore_prev_device) {
      return Ort::Status{};
    }

    return StatusFromCudaError(cudaSetDevice(prev_device));
  };
  try {
    PL_CUDA_CALL_THROW(cudaSetDevice(stream_.GetDeviceId()));
    PL_CUDA_CALL_THROW(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  } catch (const std::exception& ex) {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
      event_ = nullptr;
    }
    Ort::Status restore_status = restore_prev_device_status();
    if (!restore_status.IsOK()) {
      // Surface both failures instead of silently dropping the restore error
      // or masking the original constructor failure.
      throw std::runtime_error(
          "CudaSyncNotification construction failed: " + std::string(ex.what()) +
          ". Additionally, failed to restore previous CUDA device " +
          std::to_string(prev_device) + ": " + restore_status.GetErrorMessage());
    }
    throw;
  }
  Ort::Status restore_status = restore_prev_device_status();
  if (!restore_status.IsOK()) {
    if (event_ != nullptr) {
      // The constructor can still throw after event creation if restoring the
      // caller's previous device fails, so clean up here instead of relying on
      // the destructor of a not-fully-constructed object.
      cudaEventDestroy(event_);
      event_ = nullptr;
    }
    throw std::runtime_error(
        "Failed to restore previous CUDA device " + std::to_string(prev_device) +
        " after creating CUDA sync notification: " + restore_status.GetErrorMessage());
  }
}

CudaSyncNotification::~CudaSyncNotification() {
  if (event_) {
    cudaEventDestroy(event_);
  }
}

/*static*/ OrtStatus* ORT_API_CALL CudaSyncNotification::ActivateImpl(
    OrtSyncNotificationImpl* this_ptr) noexcept {
  auto* notif = static_cast<CudaSyncNotification*>(this_ptr);
  PL_CUDA_RETURN_IF_ERROR(cudaEventRecord(notif->event_, notif->stream_.GetCudaStream()));
  return nullptr;
}

/*static*/ OrtStatus* ORT_API_CALL CudaSyncNotification::WaitOnDeviceImpl(
    OrtSyncNotificationImpl* this_ptr, OrtSyncStream* stream) noexcept {
  auto* notif = static_cast<CudaSyncNotification*>(this_ptr);
  // SyncStream_GetHandle is in the main ORT API
  cudaStream_t wait_stream = static_cast<cudaStream_t>(Ort::GetApi().SyncStream_GetHandle(stream));
  PL_CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(wait_stream, notif->event_, 0));
  return nullptr;
}

/*static*/ OrtStatus* ORT_API_CALL CudaSyncNotification::WaitOnHostImpl(
    OrtSyncNotificationImpl* this_ptr) noexcept {
  auto* notif = static_cast<CudaSyncNotification*>(this_ptr);
  PL_CUDA_RETURN_IF_ERROR(cudaEventSynchronize(notif->event_));
  return nullptr;
}

/*static*/ void ORT_API_CALL CudaSyncNotification::ReleaseImpl(
    OrtSyncNotificationImpl* this_ptr) noexcept {
  delete static_cast<CudaSyncNotification*>(this_ptr);
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
