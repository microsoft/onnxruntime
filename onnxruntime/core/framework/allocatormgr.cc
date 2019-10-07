// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocatormgr.h"
#include "core/framework/bfc_arena.h"
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <limits>

namespace onnxruntime {

using namespace ::onnxruntime::common;

AllocatorPtr CreateAllocator(DeviceAllocatorRegistrationInfo info, int device_id) {
  auto device_allocator = std::unique_ptr<IDeviceAllocator>(info.factory(device_id));
  if (device_allocator->AllowsArena()) {
    return std::shared_ptr<IArenaAllocator>(
        onnxruntime::make_unique<BFCArena>(std::move(device_allocator), info.max_mem));
   }

  return AllocatorPtr(std::move(device_allocator));
}

DeviceAllocatorRegistry& DeviceAllocatorRegistry::Instance() {
  static DeviceAllocatorRegistry s_instance;
  return s_instance;
}

}  // namespace onnxruntime
