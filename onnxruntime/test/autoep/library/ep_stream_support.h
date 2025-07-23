// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"
#include "example_plugin_ep_utils.h"

//
// Class implementing Stream support for synchronization.
//
class StreamImpl : public OrtSyncStreamImpl, public ApiPtrs {
 public:
  StreamImpl(ApiPtrs apis, const OrtEp* ep, const OrtKeyValuePairs* /*stream_options*/)
      : ApiPtrs(apis), ep_{ep} {
    ort_version_supported = ORT_API_VERSION;
    CreateNotification = CreateNotificationImpl;
    GetHandle = GetHandleImpl;
    Flush = FlushImpl;
    OnSessionRunEnd = OnSessionRunEndImpl;
    Release = ReleaseImpl;
  }

 private:
  static OrtStatus* ORT_API_CALL CreateNotificationImpl(_In_ OrtSyncStreamImpl* this_ptr,
                                                        _Outptr_ OrtSyncNotificationImpl** sync_notification) noexcept;
  static void* ORT_API_CALL GetHandleImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL FlushImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL OnSessionRunEndImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept;
  static void ORT_API_CALL ReleaseImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept;

  void* handle_{nullptr};  // use the real stream type, like cudaStream_t or aclrtStream, etc.

  // EP instance if the stream is being created internally for inferencing.
  // nullptr when the stream is created outside of an inference session for data copies.
  const OrtEp* ep_;
};

//
// Class implementing synchronization notification support.
//
class NotificationImpl : public OrtSyncNotificationImpl, public ApiPtrs {
 public:
  NotificationImpl(ApiPtrs apis) : ApiPtrs(apis) {
    ort_version_supported = ORT_API_VERSION;
    Activate = ActivateImpl;
    Release = ReleaseImpl;
    WaitOnDevice = WaitOnDeviceImpl;
    WaitOnHost = WaitOnHostImpl;
  }

 private:
  static OrtStatus* ORT_API_CALL ActivateImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL WaitOnDeviceImpl(_In_ OrtSyncNotificationImpl* this_ptr,
                                                  _In_ OrtSyncStream* stream) noexcept;
  static OrtStatus* ORT_API_CALL WaitOnHostImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept;
  static void ORT_API_CALL ReleaseImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept;

  void* event_{NULL};  // placeholder. e.g. CANN uses aclrtEvent, CUDA uses cudaEvent_t
};
