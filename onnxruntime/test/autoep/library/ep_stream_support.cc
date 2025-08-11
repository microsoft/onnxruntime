// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep_stream_support.h"
#include "ep_factory.h"
//
// StreamImpl implementation
//

/*static*/
OrtStatus* ORT_API_CALL StreamImpl::CreateNotificationImpl(_In_ OrtSyncStreamImpl* this_ptr,
                                                           _Outptr_ OrtSyncNotificationImpl** notification) noexcept {
  auto& impl = *static_cast<StreamImpl*>(this_ptr);
  *notification = std::make_unique<NotificationImpl>(impl).release();
  return nullptr;
}

/*static*/
void* ORT_API_CALL StreamImpl::GetHandleImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
  auto& impl = *static_cast<StreamImpl*>(this_ptr);
  return impl.handle_;
}

/*static*/
OrtStatus* ORT_API_CALL StreamImpl::FlushImpl(_In_ OrtSyncStreamImpl* /*this_ptr*/) noexcept {
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL StreamImpl::OnSessionRunEndImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
  auto& impl = *static_cast<StreamImpl*>(this_ptr);
  auto* arena = impl.factory_->GetArenaAllocator();
  if (arena) {
    arena->ResetChunksUsingStream(this_ptr);
  }

  return nullptr;
}

// callback for EP library to release any internal state
/*static*/
void ORT_API_CALL StreamImpl::ReleaseImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
  delete static_cast<StreamImpl*>(this_ptr);
}

//
// Notification support
//

/*static*/
OrtStatus* ORT_API_CALL NotificationImpl::ActivateImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept {
  auto& impl = *static_cast<NotificationImpl*>(this_ptr);
  static_cast<void>(impl);

  // e.g.
  // CUDA: cudaEventRecord
  // CANN: aclrtRecordEvent
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL NotificationImpl::WaitOnDeviceImpl(_In_ OrtSyncNotificationImpl* this_ptr,
                                                           _In_ OrtSyncStream* stream) noexcept {
  auto& impl = *static_cast<NotificationImpl*>(this_ptr);
  void* handle = impl.ort_api.SyncStream_GetHandle(stream);
  static_cast<void>(handle);

  // Setup the event or similar that will be activated on notification.
  // See CudaNotification or CannNotification for examples.
  //
  // e.g.
  // CUDA: cudaStreamWaitEvent(static_cast<cudaStream_t>(device_stream.GetHandle()), event_)
  // CANN: aclrtStreamWaitEvent(static_cast<aclrtStream>(device_stream.GetHandle()), event_)
  //
  // `event_` should be a member that is created in the ctor.
  // The stream handle should come from the StreamImpl instance and can be the real type so no static_cast is needed.
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL NotificationImpl::WaitOnHostImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept {
  auto& impl = *static_cast<NotificationImpl*>(this_ptr);
  static_cast<void>(impl);

  // e.g.
  // CUDA: cudaEventSynchronize(event_)
  // CANN: aclrtSynchronizeEvent(event_)
  return nullptr;
}

/*static*/
void ORT_API_CALL NotificationImpl::ReleaseImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept {
  delete static_cast<NotificationImpl*>(this_ptr);
}
