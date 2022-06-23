// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocatormgr.h"
#include "core/framework/bfc_arena.h"
#include "core/common/logging/logging.h"
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <limits>

namespace onnxruntime {
using namespace common;

namespace {
int32_t MakeKey(OrtMemType mem_type, OrtDevice device) {
  // shorten device id so we can fit everything
  uint8_t short_device = gsl::narrow<uint8_t>(device.Id());
  // and convert mem_type. OrtMemType weirdly uses -2 as the first value so we offset by that before narrowing
  uint8_t ort_mem_type = gsl::narrow<uint8_t>(mem_type + 2);

  // NOTE: OrtMemType is the type of memory for a kernel's input/output
  //       OrtDevice.MemType is the device memory type.
  return int32_t(device.Type()) << 24 | int32_t(device.MemType()) << 16 | short_device << 8 | ort_mem_type;
}
}  // namespace

AllocatorPtr CreateAllocator(const AllocatorCreationInfo& info) {
  auto device_allocator = info.device_alloc_factory(info.device_id);

  if (info.use_arena) {
    size_t max_mem = info.arena_cfg.max_mem == 0 ? BFCArena::DEFAULT_MAX_MEM : info.arena_cfg.max_mem;
    int initial_chunk_size_bytes = info.arena_cfg.initial_chunk_size_bytes == -1
                                       ? BFCArena::DEFAULT_INITIAL_CHUNK_SIZE_BYTES
                                       : info.arena_cfg.initial_chunk_size_bytes;
    int max_dead_bytes_per_chunk = info.arena_cfg.max_dead_bytes_per_chunk == -1
                                       ? BFCArena::DEFAULT_MAX_DEAD_BYTES_PER_CHUNK
                                       : info.arena_cfg.max_dead_bytes_per_chunk;
    int initial_growth_chunk_size_bytes = info.arena_cfg.initial_growth_chunk_size_bytes == -1
                                              ? BFCArena::DEFAULT_INITIAL_GROWTH_CHUNK_SIZE_BYTES
                                              : info.arena_cfg.initial_growth_chunk_size_bytes;
    ArenaExtendStrategy arena_extend_str;
    switch (info.arena_cfg.arena_extend_strategy) {
      case static_cast<int>(ArenaExtendStrategy::kSameAsRequested):
        arena_extend_str = ArenaExtendStrategy::kSameAsRequested;
        break;
      case -1:  // default value supplied by user
      case static_cast<int>(ArenaExtendStrategy::kNextPowerOfTwo):
        arena_extend_str = ArenaExtendStrategy::kNextPowerOfTwo;
        break;
      default:
        LOGS_DEFAULT(ERROR) << "Received invalid value of arena_extend_strategy "
                            << info.arena_cfg.arena_extend_strategy;
        return nullptr;
    }

    return AllocatorPtr(std::make_unique<BFCArena>(std::move(device_allocator),
                                                   max_mem,
                                                   arena_extend_str,
                                                   initial_chunk_size_bytes,
                                                   max_dead_bytes_per_chunk,
                                                   initial_growth_chunk_size_bytes));
  } else {
    return device_allocator;
  }
}

// Update allocator in the provider if already present; ignore if not.
void AllocatorManager::ReplaceAllocator(AllocatorPtr allocator) {
  const auto& info = allocator->Info();
  auto iter = allocators_.find(MakeKey(info.mem_type, info.device));
  if (iter != allocators_.end()) {
    iter->second = allocator;
  }
}

void AllocatorManager::InsertAllocator(AllocatorPtr allocator) {
  const OrtMemoryInfo& info = allocator->Info();
  int32_t key = MakeKey(info.mem_type, info.device);
  auto iter = allocators_.find(key);
  if (iter != allocators_.end()) {
    ORT_THROW("Duplicate allocator for OrtMemType:", info.mem_type, " device:", info.device.ToString(),
              " Existing allocator: ", iter->second->Info().name,
              " New allocator: ", allocator->Info().name);
  }

  allocators_[key] = allocator;
}

AllocatorPtr AllocatorManager::GetAllocator(OrtMemType mem_type, OrtDevice device) const {
  auto iter = allocators_.find(MakeKey(mem_type, device));
  return iter != allocators_.end() ? iter->second : nullptr;
}
}  // namespace onnxruntime
