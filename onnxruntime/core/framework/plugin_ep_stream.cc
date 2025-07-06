// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/plugin_ep_stream.h"
#include "core/framework/error_code_helper.h"

namespace onnxruntime {
namespace plugin_ep {

// TODO: Is num_consumers meaningful? Unused everywhere currently.
OrtStatus* Stream::CreateNotificationImpl(size_t /*num_consumers*/, std::unique_ptr<Notification>& result) {
  OrtSyncNotificationImpl* notification_impl = nullptr;
  ORT_API_RETURN_IF_ERROR(impl_.CreateNotification(&impl_, &notification_impl));

  result = std::make_unique<Notification>(*this, *notification_impl);
  return nullptr;
}

OrtStatus* Stream::EnableInputNotification() {
  assert(input_notification_ == nullptr);
  ORT_API_RETURN_IF_ERROR(CreateNotificationImpl(0, input_notification_));

  return nullptr;
}

OrtStatus* Stream::EnableOutputNotification() {
  assert(output_notification_ == nullptr);
  ORT_API_RETURN_IF_ERROR(CreateNotificationImpl(0, output_notification_));

  return nullptr;
}

void Stream::WaitOnInput() {
  if (!input_notification_) {
    return;
  }

  input_notification_->WaitNotificationOnDevice(ToApiStream(), *input_notification_);
}

void Stream::SignalInputAvailable() {
  if (!input_notification_) {
    return;
  }

  input_notification_->Activate();  // TODO: do we need to call ActivateAndUpdate here or does it not matter?
}

void Stream::SignalOutputAvailable() {
  if (!output_notification_) {
    return;
  }

  output_notification_->Activate();  // TODO: do we need to call ActivateAndUpdate here or does it not matter?
}

}  // namespace plugin_ep
}  // namespace onnxruntime
