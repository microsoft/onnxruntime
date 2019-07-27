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

namespace {
//It assumes max(OrtMemType) <= 1, min(OrtMemType) = -2
inline int MakeKey(int id, OrtMemType mem_type) {
  return id << 2 | (mem_type + 2);
}
}  // namespace

AllocatorPtr AllocatorManager::GetAllocator(int id, OrtMemType mem_type) const {
  auto iter = allocators_.find(MakeKey(id, mem_type));
  if (iter != allocators_.end()) {
    return iter->second;
  }
  return nullptr;
}

void AllocatorManager::InsertAllocator(AllocatorPtr allocator) {
  const OrtAllocatorInfo& info = allocator->Info();
  const int key = MakeKey(info.id, info.mem_type);
  auto iter = allocators_.find(key);
  if (iter != allocators_.end()) {
    ORT_THROW("duplicated allocator");
  }
  allocators_.insert(iter, {key, allocator});
  allocator_list_.emplace_back(gsl::not_null<IAllocator*>(allocator.get()));
}

}  // namespace onnxruntime
