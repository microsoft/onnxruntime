// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/arena.h"

namespace onnxruntime {

using DeviceAllocatorFactory = std::function<std::unique_ptr<IDeviceAllocator>(OrtDevice::DeviceId)>;

struct DeviceAllocatorRegistrationInfo {
  OrtMemType mem_type;
  DeviceAllocatorFactory factory;
  size_t max_mem;
};

AllocatorPtr CreateAllocator(DeviceAllocatorRegistrationInfo info, OrtDevice::DeviceId device_id = 0);

class DeviceAllocatorRegistry {
 public:
  void RegisterDeviceAllocator(std::string&& name, DeviceAllocatorFactory factory, size_t max_mem,
                               OrtMemType mem_type = OrtMemTypeDefault) {
    DeviceAllocatorRegistrationInfo info({mem_type, factory, max_mem});
    device_allocator_registrations_.emplace(std::move(name), std::move(info));
  }

  const std::map<std::string, DeviceAllocatorRegistrationInfo>& AllRegistrations() const {
    return device_allocator_registrations_;
  }

  static DeviceAllocatorRegistry& Instance();

 private:
  DeviceAllocatorRegistry() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DeviceAllocatorRegistry);

  std::map<std::string, DeviceAllocatorRegistrationInfo> device_allocator_registrations_;
};

}  // namespace onnxruntime
