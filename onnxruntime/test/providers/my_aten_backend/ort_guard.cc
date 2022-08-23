// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace torch_my_kernel_lib {
namespace aten {

constexpr const char* kORTVirtualDeviceCount = "ORT_VIRTUAL_DEVICE_COUNT";

struct ORTGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  ORTGuardImpl() {
  }

  explicit ORTGuardImpl(at::DeviceType t) {
    AT_ASSERT(t == at::DeviceType::ORT);
  }

  at::DeviceType type() const override {
    return at::DeviceType::ORT;
  }

  at::Device exchangeDevice(at::Device d) const override {
    AT_ASSERT(d.type() == at::DeviceType::ORT);
    auto old_device = getDevice();
    if (old_device.index() != d.index()) {
      current_device_ = d.index();
    }
    return old_device;
  }

  at::Device getDevice() const override {
    return at::Device(at::DeviceType::ORT, current_device_);
  }

  void setDevice(at::Device d) const override {
    AT_ASSERT(d.type() == at::DeviceType::ORT);
    AT_ASSERT(d.index() >= 0);
    current_device_ = d.index();
  }

  void uncheckedSetDevice(at::Device d) const noexcept override {
    current_device_ = d.index();
  }

  at::Stream getStream(at::Device d) const noexcept override {
    return at::Stream(at::Stream::UNSAFE, d, current_streams_[d.index()]);
  }

  at::Stream exchangeStream(at::Stream s) const noexcept override {
    auto old_id = current_streams_[s.device_index()];
    current_streams_[s.device_index()] = s.id();
    return at::Stream(at::Stream::UNSAFE, s.device(), old_id);
  }

  at::DeviceIndex deviceCount() const noexcept override {
    // todo: fix it later by query the device count
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

}  // namespace aten
}  // namespace torch_my_kernel_lib
