// Copyright (c) Microsoft Corporation. All rights reserved.

#pragma once

#include "core/session/onnxruntime_c_api.h"

#include "core/providers/cuda/cuda_pch.h"
#include "core/providers/cuda/plugin_ep/utils.h"

namespace cuda_plugin_ep {

struct CudaSyncNotificationImpl : OrtSyncNotificationImpl {
  static OrtStatus* Create(cudaStream_t stream, std::unique_ptr<CudaSyncNotificationImpl>& notification);

  static void ReleaseImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept;

  static OrtStatus* ActivateImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept;

  static OrtStatus* WaitOnDeviceImpl(_In_ OrtSyncNotificationImpl* this_ptr,
                                     _In_ OrtSyncStream* consumer_stream) noexcept;

  static OrtStatus* WaitOnHostImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept;

  ~CudaSyncNotificationImpl();

 private:
  CudaSyncNotificationImpl(cudaStream_t stream);

  cudaStream_t stream_;
  cudaEvent_t event_;
};

struct CudaSyncStreamImpl : OrtSyncStreamImpl {
  static OrtStatus* Create(cudaStream_t&& stream,
                           const OrtMemoryDevice& device,
                           std::unique_ptr<CudaSyncStreamImpl>& sync_stream);

  static void ReleaseImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
    delete static_cast<CudaSyncStreamImpl*>(this_ptr);
  }

  static void* GetHandleImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
    auto& impl = *static_cast<CudaSyncStreamImpl*>(this_ptr);
    return impl.stream_;
  }

  static OrtStatus* CreateNotificationImpl(_In_ OrtSyncStreamImpl* this_ptr,
                                           _Outptr_ OrtSyncNotificationImpl** notification_impl) noexcept;

  static OrtStatus* FlushImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept;

  static OrtStatus* OnSessionRunEndImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept;

 private:
  CudaSyncStreamImpl(cudaStream_t&& stream,
                     const OrtMemoryDevice& device);

  OrtStatus* CreateHandles();

  cudaStream_t stream_{};
  cudnnHandle_t cudnn_handle_{};
  cublasHandle_t cublas_handle_{};
};

}  // namespace cuda_plugin_ep
