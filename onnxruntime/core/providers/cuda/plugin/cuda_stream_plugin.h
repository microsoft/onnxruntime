// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// CUDA stream and event-based synchronization primitives for the plugin EP.
// CudaSyncStream wraps a cudaStream_t and, for owned streams, cuBLAS/cuDNN/
// cuBLASLt handles. External graph streams are registered without owning
// library handles and migrated kernels fall back to thread-local defaults.
// CudaSyncNotification wraps a cudaEvent_t for cross-stream synchronization.
// A global stream registry (with TLS-cached lookups) allows migrated kernels
// to obtain their compute handles from a raw cudaStream_t.

#pragma once

#include "cuda_plugin_utils.h"

#include <mutex>
#include <unordered_map>
#include <vector>

namespace onnxruntime {
namespace cuda_plugin {

class CudaSyncNotification;
class CudaEpFactory;

/// CUDA stream implementation for the plugin EP.
/// Owns a cudaStream_t and associated CUDA library handles for owned streams,
/// or wraps an external stream for graph-mode registration/lifecycle tracking.
class CudaSyncStream : public OrtSyncStreamImpl {
 public:
  CudaSyncStream(CudaEpFactory& factory, int device_id,
                 const OrtEp* ep);
  ~CudaSyncStream();

  int GetDeviceId() const { return device_id_; }
  cudaStream_t GetCudaStream() const { return cuda_stream_; }
  cublasHandle_t GetCublasHandle() const { return cublas_handle_; }
  cudnnHandle_t GetCudnnHandle() const { return cudnn_handle_; }
  cublasLtHandle_t GetCublasLtHandle() const { return cublas_lt_handle_; }

  void EnqueueDeferredCPUBuffer(void* cpu_buffer);
  OrtStatus* InitHandles();

  /// Initialize with an external (non-owned) CUDA stream. The wrapper is
  /// registered for stream-aware lookup/cleanup, but CUDA library handles are
  /// resolved later from thread-local defaults when kernels dispatch.
  OrtStatus* InitHandlesWithExternalStream(cudaStream_t external_stream);

  /// Look up the CudaSyncStream wrapper from a raw cudaStream_t handle.
  /// Uses a thread-local TLS cache with a generation counter to avoid lock
  /// contention on this hot path (called on every kernel launch).
  static CudaSyncStream* FromCudaStream(cudaStream_t stream);

 private:
  static void RegisterStream(cudaStream_t stream, CudaSyncStream* sync_stream);
  static void UnregisterStream(cudaStream_t stream);
  static void* ORT_API_CALL GetHandleImpl(OrtSyncStreamImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL CreateNotificationImpl(
      OrtSyncStreamImpl* this_ptr, OrtSyncNotificationImpl** notification) noexcept;
  static OrtStatus* ORT_API_CALL FlushImpl(OrtSyncStreamImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL OnSessionRunEndImpl(OrtSyncStreamImpl* this_ptr) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtSyncStreamImpl* this_ptr) noexcept;

  OrtStatus* CleanupDeferredCPUBuffers() noexcept;

  CudaEpFactory& factory_;
  int device_id_;
  cudaStream_t cuda_stream_ = nullptr;
  bool owns_stream_ = true;  ///< False when wrapping an external stream (e.g., for CUDA graph).
  cublasHandle_t cublas_handle_ = nullptr;
  cudnnHandle_t cudnn_handle_ = nullptr;
  cublasLtHandle_t cublas_lt_handle_ = nullptr;

  // Tracks whether the stream was successfully registered in the global map.
  // Only registered streams should be unregistered in the destructor to avoid
  // unnecessarily bumping the TLS generation counter.
  bool registered_ = false;

  // CPU buffers whose deallocation is deferred to OnSessionRunEnd.
  // Pinned memory must remain valid until all async device operations that
  // reference it have completed, so we synchronize the stream first.
  mutable std::mutex deferred_cpu_buffers_mutex_;
  std::vector<void*> deferred_cpu_buffers_;
};

/// CUDA event-based notification for stream synchronization.
class CudaSyncNotification : public OrtSyncNotificationImpl {
 public:
  explicit CudaSyncNotification(CudaSyncStream& stream);
  ~CudaSyncNotification();

 private:
  static OrtStatus* ORT_API_CALL ActivateImpl(OrtSyncNotificationImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL WaitOnDeviceImpl(
      OrtSyncNotificationImpl* this_ptr, OrtSyncStream* stream) noexcept;
  static OrtStatus* ORT_API_CALL WaitOnHostImpl(OrtSyncNotificationImpl* this_ptr) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtSyncNotificationImpl* this_ptr) noexcept;

  CudaSyncStream& stream_;
  cudaEvent_t event_ = nullptr;
};

}  // namespace cuda_plugin
}  // namespace onnxruntime
