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

OrtStatus* Stream::CreateInputNotification() {
  if (input_notification_ != nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Cannot create a new notification when one already exists for this stream. "
                                 "Call ReleaseInputNotification first.");
  }

  return CreateNotificationImpl(0, input_notification_);
}

OrtStatus* Stream::ActivateInputNotification() {
  if (input_notification_ == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Cannot activate a notification that does not exist. "
                                 "Call CreateInputNotification first.");
  }

  input_notification_->ActivateAndUpdate();
  return nullptr;
}

void Stream::ReleaseInputNotification() {
  if (input_notification_) {
    input_notification_.reset();
  }
}
}  // namespace plugin_ep
}  // namespace onnxruntime
