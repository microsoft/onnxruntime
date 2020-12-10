// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocatormgr.h"
#include "core/framework/bfc_arena.h"
#include "core/framework/mimalloc_arena.h"
#include "core/common/logging/logging.h"
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <limits>

namespace onnxruntime {
using namespace common;

AllocatorPtr CreateAllocator(const AllocatorCreationInfo& info) {
  auto device_allocator = std::unique_ptr<IAllocator>(info.device_alloc_factory(info.device_id));

  if (info.use_arena) {
    size_t max_mem = info.arena_cfg.max_mem == 0 ? BFCArena::DEFAULT_MAX_MEM : info.arena_cfg.max_mem;
    int initial_chunk_size_bytes = info.arena_cfg.initial_chunk_size_bytes == -1
                                       ? BFCArena::DEFAULT_INITIAL_CHUNK_SIZE_BYTES
                                       : info.arena_cfg.initial_chunk_size_bytes;
    int max_dead_bytes_per_chunk = info.arena_cfg.max_dead_bytes_per_chunk == -1
                                       ? BFCArena::DEFAULT_MAX_DEAD_BYTES_PER_CHUNK
                                       : info.arena_cfg.max_dead_bytes_per_chunk;
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
        LOGS_DEFAULT(ERROR) << "Received invalid value of arena_extend_strategy " << info.arena_cfg.arena_extend_strategy;
        return nullptr;
    }

#ifdef USE_MIMALLOC
    return std::shared_ptr<IArenaAllocator>(
        onnxruntime::make_unique<MiMallocArena>(std::move(device_allocator), max_mem));
#else
    return std::shared_ptr<IArenaAllocator>(
        onnxruntime::make_unique<BFCArena>(std::move(device_allocator),
                                           max_mem,
                                           arena_extend_str,
                                           initial_chunk_size_bytes,
                                           max_dead_bytes_per_chunk));
#endif
  }

  return AllocatorPtr(std::move(device_allocator));
}

}  // namespace onnxruntime
