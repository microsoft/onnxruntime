// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/arena.h"
#include "core/framework/bfc_arena.h"

namespace onnxruntime {

using DeviceAllocatorFactory = std::function<std::unique_ptr<IDeviceAllocator>(OrtDevice::DeviceId)>;

struct DeviceAllocatorRegistrationInfo {
  DeviceAllocatorRegistrationInfo(OrtMemType ort_mem_type,
                                  DeviceAllocatorFactory alloc_factory,
                                  size_t mem,
                                  ArenaExtendStrategy strategy = ArenaExtendStrategy::kNextPowerOfTwo)
      : mem_type(ort_mem_type),
        factory(alloc_factory),
        max_mem(mem),
        arena_extend_strategy(strategy) {
  }

  OrtMemType mem_type;
  DeviceAllocatorFactory factory;
  size_t max_mem;
  ArenaExtendStrategy arena_extend_strategy;
};

AllocatorPtr CreateAllocator(const DeviceAllocatorRegistrationInfo& info, OrtDevice::DeviceId device_id = 0,
                             bool use_arena = true);

}  // namespace onnxruntime
