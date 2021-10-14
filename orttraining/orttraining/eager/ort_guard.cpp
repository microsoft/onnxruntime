// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <torch/extension.h>

#include "ort_backends.h"
#include "ort_log.h"

namespace torch_ort {
namespace eager {

struct ORTGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  ORTGuardImpl() {
  }

  explicit ORTGuardImpl(at::DeviceType t) {
    AT_ASSERT(t == at::DeviceType::ORT);
  }

  at::DeviceType type() const override {
    ORT_LOG_FN();
    return at::DeviceType::ORT;
  }

  at::Device exchangeDevice(at::Device d) const override {
    ORT_LOG_FN(d);
    AT_ASSERT(d.type() == at::DeviceType::ORT);
    auto old_device = getDevice();
    if (old_device.index() != d.index()) {
      current_device_ = d.index();
    }
    return old_device;
  }

  at::Device getDevice() const override {
    ORT_LOG_FN();
    return at::Device(at::DeviceType::ORT, current_device_);
  }

  void setDevice(at::Device d) const override {
    ORT_LOG_FN(d);
    AT_ASSERT(d.type() == at::DeviceType::ORT);
    AT_ASSERT(d.index() >= 0);
    current_device_ = d.index();
  }

  void uncheckedSetDevice(at::Device d) const noexcept override {
    ORT_LOG_FN(d);
    current_device_ = d.index();
  }

  at::Stream getStream(at::Device d) const noexcept override {
    ORT_LOG_FN(d);
    return at::Stream(at::Stream::UNSAFE, d, current_streams_[d.index()]);
  }

  at::Stream exchangeStream(at::Stream s) const noexcept override {
    ORT_LOG_FN(s);
    auto old_id = current_streams_[s.device_index()];
    current_streams_[s.device_index()] = s.id();
    return at::Stream(at::Stream::UNSAFE, s.device(), old_id);
  }

  at::DeviceIndex deviceCount() const noexcept override {
    ORT_LOG_FN();
    return 1;
  }

//  #pragma region events

  #define EVENTS_NIEX TORCH_CHECK(false, "ORT backend doesn't support events.")

  void record(void** event,
    const at::Stream& stream,
    const at::DeviceIndex device_index,
    const at::EventFlag flag) const override {
    EVENTS_NIEX;
  }

  void block(
    void* event,
    const at::Stream& stream) const override {
    EVENTS_NIEX;
  }

  bool queryEvent(void* event) const override {
    EVENTS_NIEX;
  }

  void destroyEvent(
    void* event,
    const at::DeviceIndex device_index) const noexcept override {
  }

  #undef EVENTS_NIEX

  //#pragma endregion events

 private:
  thread_local static at::DeviceIndex current_device_;
  thread_local static std::map<at::DeviceIndex, at::StreamId> current_streams_;
};

thread_local at::DeviceIndex ORTGuardImpl::current_device_ = 0;
thread_local std::map<at::DeviceIndex, at::StreamId> ORTGuardImpl::current_streams_ = {};

C10_REGISTER_GUARD_IMPL(ORT, ORTGuardImpl);

} // namespace eager
} // namespace torch_ort