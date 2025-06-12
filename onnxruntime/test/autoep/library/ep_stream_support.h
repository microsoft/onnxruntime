// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"
#include "utils.h"

//
// Class implementing Stream support for synchronization.
//
class StreamImpl : public OrtSyncStreamImpl, public ApiPtrs {
 public:
  StreamImpl(ApiPtrs apis) : ApiPtrs(apis) {
    version = ORT_API_VERSION;
    CreateNotification = CreateNotificationImpl;
    Flush = FlushImpl;
    OnSessionRunEnd = OnSessionRunEndImpl;
    Release = ReleaseImpl;
  }

  void* GetHandle() {
    return handle_;
  }

 private:
  static OrtStatus* ORT_API_CALL CreateNotificationImpl(_In_ void* this_ptr, _In_ struct OrtSyncStream* stream,
                                                        _In_ size_t num_consumers,
                                                        _Outptr_ OrtSyncNotification** sync_notification) noexcept;
  static OrtStatus* ORT_API_CALL FlushImpl(_In_ void* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL OnSessionRunEndImpl(_In_ void* this_ptr) noexcept;
  static void ORT_API_CALL ReleaseImpl(_In_ void* this_ptr) noexcept;

  void* handle_{nullptr};  // use the real stream type, like cudaStream_t or aclrtStream, etc.
};

//
// Class implementing synchronization notification support.
//
class NotificationImpl : public OrtSyncNotificationImpl, public ApiPtrs {
 public:
  NotificationImpl(ApiPtrs apis) : ApiPtrs(apis) {
    Activate = ActivateImpl;
    Release = ReleaseImpl;
    WaitOnDevice = WaitOnDeviceImpl;
    WaitOnHost = WaitOnHostImpl;
  }

 private:
  static void ORT_API_CALL ActivateImpl(_In_ void* this_ptr) noexcept;
  static void ORT_API_CALL WaitOnDeviceImpl(_In_ void* this_ptr, _In_ OrtSyncStream* stream) noexcept;
  static void ORT_API_CALL WaitOnHostImpl(_In_ void* this_ptr) noexcept;
  static void ORT_API_CALL ReleaseImpl(_In_ void* this_ptr) noexcept;

  void* event_{NULL};  // placeholder. e.g. CANN uses aclrtEvent, CUDA uses cudaEvent_t
};
