// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cann/cann_stream_handle.h"
#include "core/common/spin_pause.h"

namespace onnxruntime {

struct CannNotification : public synchronize::Notification {
  explicit CannNotification(Stream& s) : Notification(s) {
    CANN_CALL_THROW(aclrtCreateEvent(&event_));
  }

  ~CannNotification() {
    if (event_)
      CANN_CALL_THROW(aclrtDestroyEvent(event_));
  }

  void Activate() override {
    CANN_CALL_THROW(aclrtRecordEvent(event_, static_cast<aclrtStream>(stream_.GetHandle())));
  }

  void wait_on_device(Stream& device_stream) {
    ORT_ENFORCE(device_stream.GetDevice().Type() == OrtDevice::NPU);
    CANN_CALL_THROW(aclrtStreamWaitEvent(static_cast<aclrtStream>(device_stream.GetHandle()), event_));
  }

  void wait_on_host() {
    CANN_CALL_THROW(aclrtSynchronizeEvent(event_));
  }

  aclrtEvent event_;
};

CannStream::CannStream(aclrtStream stream,
                       const OrtDevice& device,
                       bool own_flag) : Stream(stream, device),
                                        own_stream_(own_flag) {}

CannStream::~CannStream() {
  ORT_IGNORE_RETURN_VALUE(CleanUpOnRunEnd());
  if (own_stream_) {
    auto* handle = GetHandle();
    if (handle)
      aclrtDestroyStream(static_cast<aclrtStream>(handle));
  }
}

std::unique_ptr<synchronize::Notification> CannStream::CreateNotification(size_t /*num_consumers*/) {
  return std::make_unique<CannNotification>(*this);
}

void CannStream::Flush() {
  if (own_stream_)
    CANN_CALL_THROW(aclrtSynchronizeStream(static_cast<aclrtStream>(GetHandle())));
}

// CPU Stream command handles
void WaitCannNotificationOnDevice(Stream& stream, synchronize::Notification& notification) {
  static_cast<CannNotification*>(&notification)->wait_on_device(stream);
}

void WaitCannNotificationOnHost(Stream& /*stream*/, synchronize::Notification& notification) {
  static_cast<CannNotification*>(&notification)->wait_on_host();
}

void RegisterCannStreamHandles(IStreamCommandHandleRegistry& stream_handle_registry,
                               const OrtDevice::DeviceType device_type) {
  // wait cann notification on cann ep
  stream_handle_registry.RegisterWaitFn(device_type, device_type, WaitCannNotificationOnDevice);
  // wait cann notification on cpu ep
  stream_handle_registry.RegisterWaitFn(device_type, OrtDevice::CPU, WaitCannNotificationOnHost);
  stream_handle_registry.RegisterCreateStreamFn(device_type, [](const OrtDevice& device) {
    aclrtStream stream = nullptr;
    CANN_CALL_THROW(aclrtCreateStream(&stream));
    return std::make_unique<CannStream>(stream, device, true);
  });
}

}  // namespace onnxruntime
