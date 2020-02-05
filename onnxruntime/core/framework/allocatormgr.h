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

}  // namespace onnxruntime
