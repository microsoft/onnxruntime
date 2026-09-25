// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// CUDA stream and event-based synchronization primitives for the plugin EP.
// CudaSyncStream wraps a cudaStream_t and, for owned streams, cuBLAS/cuDNN/
// cuBLASLt handles. User compute streams borrow library handles owned by the EP.
// External graph streams are registered without library handles and migrated
// kernels fall back to thread-local defaults.
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

/// Owns cuBLAS/cuDNN/cuBLASLt handles bound to a single CUDA stream.
struct CudaLibraryHandles {
  CudaLibraryHandles() = default;
  ~CudaLibraryHandles() { Reset(); }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudaLibraryHandles);

  /// Create the handles on `device_id` and bind them to `stream`. The cuDNN handle is only
  /// created when `enable_cudnn` is set and cuDNN is available.
  OrtStatus* Init(int device_id, cudaStream_t stream, bool enable_cudnn);
  void Reset() noexcept;

  cublasHandle_t cublas = nullptr;
  cudnnHandle_t cudnn = nullptr;
  cublasLtHandle_t cublas_lt = nullptr;
};

/// CUDA stream implementation for the plugin EP.
/// Owns a cudaStream_t and associated CUDA library handles for owned streams,
/// or wraps an external stream for graph-mode registration/lifecycle tracking.
class CudaSyncStream : public OrtSyncStreamImpl {
 public:
  CudaSyncStream(CudaEpFactory& factory, int device_id, bool enable_cudnn,
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

  /// Initialize with a user-provided external CUDA stream and library handles already bound
  /// to it. Neither the stream nor the handles are owned; the EP keeps the handles alive for
  /// its whole lifetime so captured CUDA graphs never outlive their cuBLAS workspace.
  void InitHandlesWithUserStream(cudaStream_t user_stream, const CudaLibraryHandles& handles);

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

  /// Drain the stream via OnSessionRunEnd with the stream's own device selected.
  /// Returns false if the stream could not be drained.
  bool DrainOnOwningDevice() noexcept;

  CudaEpFactory& factory_;
  int device_id_;
  bool enable_cudnn_ = true;
  cudaStream_t cuda_stream_ = nullptr;
  bool owns_stream_ = true;  ///< False when wrapping an external stream (e.g., for CUDA graph).
  CudaLibraryHandles owned_handles_;
  // Views of either owned_handles_ or handles borrowed from the EP.
  cublasHandle_t cublas_handle_ = nullptr;
  cudnnHandle_t cudnn_handle_ = nullptr;
  cublasLtHandle_t cublas_lt_handle_ = nullptr;

  // Tracks whether the stream was successfully registered in the global map.
  // Only registered streams should be unregistered in the destructor to avoid
  // unnecessarily bumping the TLS generation counter.
  bool registered_ = false;
  bool initialized_ = false;

  // CPU buffers whose deallocation is deferred to OnSessionRunEnd.
  // Pinned memory must remain valid until all async device operations that
  // reference it have completed, so we synchronize the stream first.
  mutable std::mutex deferred_cpu_buffers_mutex_;
  std::vector<void*> deferred_cpu_buffers_;
  bool stream_synchronized_and_chunks_reset_ = false;
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
