// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/common/logging/logging.h"
#include "core/framework/stream_handles.h"
#include "core/framework/error_code_helper.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"

// OrtSyncStream is an alias in the C API for onnxruntime::Stream
// OrtSyncNotification is an alias in the C API for onnxruntime::synchronize::Notification
struct OrtSyncStream : public onnxruntime::Stream {};
struct OrtSyncNotification : onnxruntime::synchronize::Notification {};

using onnxruntime::logging::Logger;

#define LOG_AND_RETURN_IF_ORT_ERROR(fn, logger)                                              \
  do {                                                                                       \
    OrtStatus* _status = (fn);                                                               \
    if (_status != nullptr) {                                                                \
      LOGS(logger, ERROR) << "Plug-in EP Error: [" << OrtApis::GetErrorCode(_status) << "] " \
                          << OrtApis::GetErrorMessage(_status);                              \
      OrtApis::ReleaseStatus(_status);                                                       \
      return;                                                                                \
    }                                                                                        \
  } while (0)

namespace onnxruntime {
namespace plugin_ep {

class Notification : public synchronize::Notification {
 public:
  Notification(Stream& stream, OrtSyncNotificationImpl& impl, const Logger& logger)
      : synchronize::Notification(stream), impl_{impl}, logger_{logger} {
  }

  static void WaitNotificationOnDevice(onnxruntime::Stream* stream, synchronize::Notification& notification) {
    auto* this_ptr = static_cast<Notification*>(&notification);

    LOG_AND_RETURN_IF_ORT_ERROR(this_ptr->impl_.WaitOnDevice(&this_ptr->impl_, static_cast<OrtSyncStream*>(stream)),
                                this_ptr->logger_);
  }

  static void WaitNotificationOnHost(onnxruntime::Stream* /*stream*/, synchronize::Notification& notification) {
    auto* this_ptr = static_cast<Notification*>(&notification);
    LOG_AND_RETURN_IF_ORT_ERROR(this_ptr->impl_.WaitOnHost(&this_ptr->impl_), this_ptr->logger_);
  }

  void Activate() override {
    LOG_AND_RETURN_IF_ORT_ERROR(impl_.Activate(&impl_), logger_);
  }

  ~Notification() override {
    impl_.Release(&impl_);
  }

 private:
  OrtSyncNotificationImpl& impl_;
  const Logger& logger_;
};

class Stream : public onnxruntime::Stream {
 public:
  Stream(const OrtDevice& memory_device, OrtSyncStreamImpl& impl, const logging::Logger& logger)
      : onnxruntime::Stream(impl.GetHandle(&impl), memory_device), impl_{impl}, logger_{logger} {
  }

  std::unique_ptr<synchronize::Notification> CreateNotification(size_t num_consumers) override {
    std::unique_ptr<Notification> plugin_notification;

    auto* ort_status = CreateNotificationImpl(num_consumers, plugin_notification);
    if (ort_status != nullptr) {
      ORT_THROW("Failed to create Notification: [", OrtApis::GetErrorCode(ort_status), "] ",
                OrtApis::GetErrorMessage(ort_status));
    }

    std::unique_ptr<synchronize::Notification> result(plugin_notification.release());
    return result;
  }

  void Flush() override {
    LOG_AND_RETURN_IF_ORT_ERROR(impl_.Flush(&impl_), logger_);
  }

  Status CleanUpOnRunEnd() override {
    auto* ort_status = impl_.OnSessionRunEnd(&impl_);
    return ToStatusAndRelease(ort_status);
  }

  WaitNotificationFn GetWaitNotificationFn() const override {
    return Notification::WaitNotificationOnDevice;
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
  const Logger& logger_;
};
}  // namespace plugin_ep
}  // namespace onnxruntime

#undef LOG_AND_RETURN_IF_ORT_ERROR
