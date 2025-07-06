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

namespace onnxruntime {
namespace plugin_ep {

class Notification : public synchronize::Notification {
 public:
  Notification(Stream& stream, OrtSyncNotificationImpl& impl)
      : synchronize::Notification(stream), impl_{impl} {
  }

  static void WaitNotificationOnDevice(onnxruntime::Stream* stream, synchronize::Notification& notification) {
    auto* this_ptr = static_cast<Notification*>(&notification);
    this_ptr->impl_.WaitOnDevice(&this_ptr->impl_, static_cast<OrtSyncStream*>(stream));
  }

  static void WaitNotificationOnHost(onnxruntime::Stream* /*stream*/, synchronize::Notification& notification) {
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
  Stream(const OrtDevice& memory_device, OrtSyncStreamImpl& impl, const logging::Logger& logger)
      : onnxruntime::Stream(&impl, memory_device), impl_{impl}, logger_{logger} {
  }

  // add Notification for use with user provided input. ORT waits on the Notification before running steps in the
  // inferencing that requires the data associated with the stream.
  OrtStatus* EnableInputNotification();

  // add Notification to provide a user that output data is available. the user waits on the Notification and ORT
  // activates it when inferencing output is available.
  OrtStatus* EnableOutputNotification();

  void WaitOnInput();  // ORT will wait until inferencing until SignalInputAvailable is called

  void SignalInputAvailable();   // called by user to indicate that async input has been added to the stream
  void SignalOutputAvailable();  // called by ORT to indicate that inferencing output has been added to the stream

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
    // Implement the flush logic here if needed
    auto* status = impl_.Flush(&impl_);

    if (status != nullptr) {
      LOGS(logger_, ERROR) << "Failed to flush stream: [" << OrtApis::GetErrorCode(status) << "] "
                           << OrtApis::GetErrorMessage(status);
      OrtApis::ReleaseStatus(status);
    }
  }

  Status CleanUpOnRunEnd() override {
    auto* ort_status = impl_.OnSessionRunEnd(&impl_);
    return ToStatusAndRelease(ort_status);
  }

  WaitNotificationFn GetWaitNotificationFn() const override {
    return Notification::WaitNotificationOnDevice;
  }

  OrtSyncStreamImpl& GetImplementation() {
    return impl_;
  }

  // when being used for user provided input this will return a Notification pointer that we need to connect to the
  // internal Stream and wait on.
  // otherwise returns nullptr.
  Notification* GetInputNotification() {
    return input_notification_.get();
  }

  // when being used for a user provided OrtSyncStream for model output this will return a Notification pointer and
  // we activate the notification when the output data is added to the stream.
  // otherwise returns nullptr.
  Notification* GetOutputNotification() {
    return output_notification_.get();
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

  std::unique_ptr<Notification> input_notification_;
  std::unique_ptr<Notification> output_notification_;
};

}  // namespace plugin_ep
}  // namespace onnxruntime
