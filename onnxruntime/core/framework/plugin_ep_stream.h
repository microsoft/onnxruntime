// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/framework/stream_handles.h"
#include "core/framework/error_code_helper.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"

// OrtSyncStream is an alias in the C API for onnxruntime::Stream
// OrtSyncNotification is an alias in the C API for onnxruntime::synchronize::Notification
struct OrtSyncStream : public onnxruntime::Stream {};
struct OrtSyncNotification : onnxruntime::synchronize::Notification {};

namespace onnxruntime {
namespace plugin_ep {

class Notification : public synchronize::Notification {
 public:
  Notification(Stream& stream, OrtSyncNotificationImpl& impl, const OrtLogger& logger)
      : synchronize::Notification(stream), impl_{impl}, logger_{logger} {
  }

  static void WaitNotificationOnDevice(onnxruntime::Stream* stream, synchronize::Notification& notification);

  static void WaitNotificationOnHost(onnxruntime::Stream* /*stream*/, synchronize::Notification& notification);

  void Activate() override;

  ~Notification() override {
    impl_.Release(&impl_);
  }

 private:
  OrtSyncNotificationImpl& impl_;
  const OrtLogger& logger_;
};

class Stream : public onnxruntime::Stream {
 public:
  Stream(const OrtDevice& memory_device, OrtSyncStreamImpl& impl, const OrtLogger& logger)
      : onnxruntime::Stream(impl.GetHandle(&impl), memory_device), impl_{impl}, logger_{logger} {
  }

  std::unique_ptr<synchronize::Notification> CreateNotification(size_t num_consumers) override {
    std::unique_ptr<Notification> plugin_notification;

    auto* ort_status = CreateNotificationImpl(num_consumers, plugin_notification);
    if (ort_status != nullptr) {
      ORT_THROW("Failed to create Notification: [", OrtApis::GetErrorCode(ort_status), "] ",
                OrtApis::GetErrorMessage(ort_status));
    }

    return plugin_notification;
  }

  void Flush() override;

  Status CleanUpOnRunEnd() override {
    auto* ort_status = impl_.OnSessionRunEnd(&impl_);
    return ToStatusAndRelease(ort_status);
  }

  const OrtSyncStreamImpl& GetImpl() const {
    return impl_;
  }

  ~Stream() override {
    impl_.Release(&impl_);
  }

 private:
  OrtSyncStream* ToApiStream() {
    return static_cast<OrtSyncStream*>(static_cast<onnxruntime::Stream*>(this));
  }

  OrtStatus* CreateNotificationImpl(size_t num_consumers, std::unique_ptr<Notification>& result);

  OrtSyncStreamImpl& impl_;
  const OrtLogger& logger_;
};
}  // namespace plugin_ep
}  // namespace onnxruntime

#undef LOG_AND_RETURN_IF_ORT_ERROR
