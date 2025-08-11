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

  result = std::make_unique<Notification>(*this, *notification_impl, logger_);
  return nullptr;
}
}  // namespace plugin_ep
}  // namespace onnxruntime
