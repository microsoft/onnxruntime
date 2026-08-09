// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/plugin_ep_stream.h"
#include "core/framework/error_code_helper.h"
#include "core/session/abi_logger.h"

using namespace ::onnxruntime::logging;

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

void Notification::WaitNotificationOnDevice(onnxruntime::Stream* stream, synchronize::Notification& notification) {
  auto* this_ptr = static_cast<Notification*>(&notification);

  LOG_AND_RETURN_IF_ORT_ERROR(this_ptr->impl_.WaitOnDevice(&this_ptr->impl_, static_cast<OrtSyncStream*>(stream)),
                              *this_ptr->logger_.ToInternal());
}
void Notification::WaitNotificationOnHost(onnxruntime::Stream*, synchronize::Notification& notification) {
  auto* this_ptr = static_cast<Notification*>(&notification);
  LOG_AND_RETURN_IF_ORT_ERROR(this_ptr->impl_.WaitOnHost(&this_ptr->impl_), *this_ptr->logger_.ToInternal());
}
void Notification::Activate() {
  LOG_AND_RETURN_IF_ORT_ERROR(impl_.Activate(&impl_), *logger_.ToInternal());
}
void Stream::Flush() {
  LOG_AND_RETURN_IF_ORT_ERROR(impl_.Flush(&impl_), *logger_.ToInternal());
}
// TODO: Is num_consumers meaningful? Unused everywhere currently.
OrtStatus* Stream::CreateNotificationImpl(size_t /*num_consumers*/, std::unique_ptr<Notification>& result) {
  OrtSyncNotificationImpl* notification_impl = nullptr;
  ORT_API_RETURN_IF_ERROR(impl_.CreateNotification(&impl_, &notification_impl));

  result = std::make_unique<Notification>(*this, *notification_impl, logger_);
  return nullptr;
}
}  // namespace plugin_ep
}  // namespace onnxruntime
