// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep_stream_support.h"

//
// StreamImpl implementation
//

/*static*/
OrtStatus* ORT_API_CALL StreamImpl::CreateNotificationImpl(_In_ void* this_ptr, _In_ struct OrtSyncStream* stream,
                                                           _In_ size_t /*num_consumers*/,
                                                           _Outptr_ OrtSyncNotification** sync_notification) noexcept {
  auto& impl = *static_cast<StreamImpl*>(this_ptr);
  auto notification = std::make_unique<NotificationImpl>(impl);
  auto* status = impl.ep_api.CreateSyncNotification(stream, notification.get(), sync_notification);

  if (status != nullptr) {
    return status;  // error occurred
  }

  notification.release();
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL StreamImpl::FlushImpl(_In_ void* /*this_ptr*/) noexcept {
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL StreamImpl::OnSessionRunEndImpl(_In_ void* /*this_ptr*/) noexcept {
  return nullptr;
}

// callback for EP library to release any internal state
/*static*/
void ORT_API_CALL StreamImpl::ReleaseImpl(_In_ void* this_ptr) noexcept {
  delete static_cast<StreamImpl*>(this_ptr);
}

//
// Notification support
//

/*static*/
void ORT_API_CALL NotificationImpl::ActivateImpl(_In_ void* this_ptr) noexcept {
  auto& impl = *static_cast<NotificationImpl*>(this_ptr);
  static_cast<void>(impl);

  // e.g.
  // CUDA: cudaEventRecord
  // CANN: aclrtRecordEvent
}

/*static*/
void ORT_API_CALL NotificationImpl::WaitOnDeviceImpl(_In_ void* this_ptr, _In_ OrtSyncStream* stream) noexcept {
  auto& impl = *static_cast<NotificationImpl*>(this_ptr);
  StreamImpl& stream_impl = *static_cast<StreamImpl*>(impl.ep_api.SyncStream_GetStreamImpl(stream));
  static_cast<void>(stream_impl);

  // TODO: Setup the event or similar that will be activated on notification.
  // See CudaNotification or CannNotification for examples
  //
  // e.g.
  // CUDA: cudaStreamWaitEvent(static_cast<cudaStream_t>(device_stream.GetHandle()), event_)
  // CANN: aclrtStreamWaitEvent(static_cast<aclrtStream>(device_stream.GetHandle()), event_)
  //
  // `event_` should be a member that is created in the ctor.
  // The stream handle should come from the StreamImpl instance and can be the real type so no static_cast is needed.
}

/*static*/
void ORT_API_CALL NotificationImpl::WaitOnHostImpl(_In_ void* this_ptr) noexcept {
  auto& impl = *static_cast<NotificationImpl*>(this_ptr);
  static_cast<void>(impl);

  // e.g.
  // CUDA: cudaEventSynchronize(event_)
  // CANN: aclrtSynchronizeEvent(event_)
}

/*static*/
void ORT_API_CALL NotificationImpl::ReleaseImpl(_In_ void* this_ptr) noexcept {
  delete static_cast<NotificationImpl*>(this_ptr);
}
