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
  if (device_allocator->AllowsArena())
    return std::shared_ptr<IArenaAllocator>(
        std::make_unique<BFCArena>(std::move(device_allocator), info.max_mem));

  return device_allocator;
}

const std::vector<gsl::not_null<const IAllocator*>>& AllocatorManager::GetAllocators() const {
  return allocator_list_;
}

AllocatorPtr AllocatorManager::GetAllocator(const OrtDevice& device) const {
  auto iter = allocators_.find(OrtDevice::MakeKey(device));
  if (iter != allocators_.end()) {
    return iter->second;
  }
  return nullptr;
}

void AllocatorManager::InsertAllocator(AllocatorPtr allocator) {
  const OrtAllocatorInfo& info = allocator->Info();
  const int key = OrtDevice::MakeKey(info.device);
  auto iter = allocators_.find(key);
  if (iter != allocators_.end()) {
    ORT_THROW("duplicated allocator");
  }
  allocators_.insert(iter, {key, allocator});
  allocator_list_.emplace_back(gsl::not_null<IAllocator*>(allocator.get()));
}

}  // namespace onnxruntime
