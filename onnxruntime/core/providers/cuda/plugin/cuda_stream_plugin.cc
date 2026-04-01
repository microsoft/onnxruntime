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
    UnregisterStream(cuda_stream_);

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
  }
}

OrtStatus* CudaSyncStream::InitHandles() {
  PL_CUDA_RETURN_IF_ERROR(cudaSetDevice(device_id_));

  PL_CUDA_RETURN_IF_ERROR(cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking));

  PL_CUBLAS_RETURN_IF_ERROR(cublasCreate(&cublas_handle_));
  PL_CUBLAS_RETURN_IF_ERROR(cublasSetStream(cublas_handle_, cuda_stream_));

  PL_CUDNN_RETURN_IF_ERROR(cudnnCreate(&cudnn_handle_));
  PL_CUDNN_RETURN_IF_ERROR(cudnnSetStream(cudnn_handle_, cuda_stream_));

  PL_CUBLAS_RETURN_IF_ERROR(cublasLtCreate(&cublas_lt_handle_));
  RegisterStream(cuda_stream_, this);

  return nullptr;
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
  PL_CUDA_CALL_THROW(cudaSetDevice(stream_.GetDeviceId()));
  PL_CUDA_CALL_THROW(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
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
