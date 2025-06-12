// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/framework/stream_handles.h"
#include "core/framework/error_code_helper.h"
#include "core/session/onnxruntime_c_api.h"

// OrtSyncStream is an alias in the C API for onnxruntime::Stream
// OrtSyncNotification is an alias in the C API for onnxruntime::synchronize::Notification
struct OrtSyncStream : public onnxruntime::Stream {};
struct OrtSyncNotification : onnxruntime::synchronize::Notification {};

namespace onnxruntime {
namespace plugin_ep {

class Notification : public synchronize::Notification {
 public:
  Notification(Stream& stream, OrtSyncNotificationImpl& impl)
      : synchronize::Notification(stream), impl_{impl} {
  }

  static void WaitNotificationOnDevice(onnxruntime::Stream& stream, synchronize::Notification& notification) {
    auto* this_ptr = static_cast<Notification*>(&notification);
    this_ptr->impl_.WaitOnDevice(&this_ptr->impl_, static_cast<OrtSyncStream*>(&stream));
  }

  static void WaitNotificationOnHost(onnxruntime::Stream& /*stream*/, synchronize::Notification& notification) {
    auto* this_ptr = static_cast<Notification*>(&notification);
    this_ptr->impl_.WaitOnHost(&this_ptr->impl_);
  }

  void Activate() override {
    impl_.Activate(&impl_);
  }

  ~Notification() override {
    impl_.Release(&impl_);
  }

 private:
  OrtSyncNotificationImpl& impl_;
};

class Stream : public onnxruntime::Stream {
 public:
  Stream(const OrtDevice& memory_device, OrtSyncStreamImpl& impl)
      : onnxruntime::Stream(&impl, memory_device), impl_{impl} {
  }

  std::unique_ptr<synchronize::Notification> CreateNotification(size_t num_consumers) override {
    OrtSyncNotification* notification = nullptr;
    OrtSyncStream* stream = static_cast<OrtSyncStream*>(static_cast<onnxruntime::Stream*>(this));

    // call the implementation, which will use the ORT API to create a plugin_ep::Notification, which will be
    // returned as an OrtSyncNotification (the API alias for synchronize::Notification)
    auto* ort_status = impl_.CreateNotification(&impl_, stream, num_consumers, &notification);

    std::unique_ptr<synchronize::Notification> result;
    if (ort_status == nullptr) {
      result.reset(static_cast<synchronize::Notification*>(notification));
    }

    return result;
  }

  void Flush() override {
    // Implement the flush logic here if needed
    impl_.Flush(&impl_);
  }

  Status CleanUpOnRunEnd() override {
    auto* ort_status = impl_.OnSessionRunEnd(&impl_);
    return (ort_status != nullptr) ? ToStatus(ort_status) : Status::OK();
  }

  WaitNotificationFn GetWaitNotificationFn() const override {
    return Notification::WaitNotificationOnDevice;
  }

  OrtSyncStreamImpl& GetImplementation() {
    return impl_;
  }

  ~Stream() override {
    impl_.Release(&impl_);
  }

 private:
  OrtSyncStreamImpl& impl_;
};

}  // namespace plugin_ep
}  // namespace onnxruntime
