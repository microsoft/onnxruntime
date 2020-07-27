// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocatormgr.h"
#include "core/framework/bfc_arena.h"
#include "core/framework/mimalloc_arena.h"
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <limits>

namespace onnxruntime {
using namespace common;

AllocatorPtr CreateAllocator(const DeviceAllocatorRegistrationInfo& info,
                             OrtDevice::DeviceId device_id, bool use_arena) {
  auto device_allocator = std::unique_ptr<IDeviceAllocator>(info.factory(device_id));

  if (use_arena) {
#ifdef USE_MIMALLOC
    return std::shared_ptr<IArenaAllocator>(
        onnxruntime::make_unique<MiMallocArena>(std::move(device_allocator), info.max_mem));
#else
    return std::shared_ptr<IArenaAllocator>(
        onnxruntime::make_unique<BFCArena>(std::move(device_allocator), info.max_mem, info.arena_extend_strategy));
#endif
  }

  return AllocatorPtr(std::move(device_allocator));
}

}  // namespace onnxruntime
