// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_plugin_utils.h"

#include <vector>
#include <unordered_map>
#include <mutex>

namespace onnxruntime {
namespace cuda_plugin {

class CudaSyncNotification;
class CudaEpFactory;

/// CUDA stream implementation for the plugin EP.
/// Owns a cudaStream_t and associated cuBLAS/cuDNN handles.
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

  void CleanupDeferredCPUBuffers();

  CudaEpFactory& factory_;
  int device_id_;
  cudaStream_t cuda_stream_ = nullptr;
  cublasHandle_t cublas_handle_ = nullptr;
  cudnnHandle_t cudnn_handle_ = nullptr;
  cublasLtHandle_t cublas_lt_handle_ = nullptr;

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
