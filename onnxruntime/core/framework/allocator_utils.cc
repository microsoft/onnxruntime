// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocator_utils.h"

#include <limits>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include <absl/base/config.h>

#include "core/common/logging/logging.h"
#include "core/common/narrow.h"
#include "core/framework/bfc_arena.h"

namespace onnxruntime {
using namespace common;

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
    int64_t max_power_of_two_extend_bytes = info.arena_cfg.max_power_of_two_extend_bytes == -1
                                                ? BFCArena::DEFAULT_MAX_POWER_OF_TWO_EXTEND_BYTES
                                                : info.arena_cfg.max_power_of_two_extend_bytes;
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

    if (info.use_stream_aware_arena) {
#ifdef ORT_ENABLE_STREAM
      return AllocatorPtr(
          std::make_unique<StreamAwareArena>(std::move(device_allocator),
                                             max_mem,
                                             info.enable_cross_stream_reusing,
                                             arena_extend_str,
                                             initial_chunk_size_bytes,
                                             max_dead_bytes_per_chunk,
                                             initial_growth_chunk_size_bytes));
#else
      ORT_THROW("StreamAwareArena should be transparent to minimal build.");
#endif
    } else {
      return AllocatorPtr(
          std::make_unique<BFCArena>(std::move(device_allocator),
                                     max_mem,
                                     arena_extend_str,
                                     initial_chunk_size_bytes,
                                     max_dead_bytes_per_chunk,
                                     initial_growth_chunk_size_bytes,
                                     max_power_of_two_extend_bytes));
    }
  } else {
    return device_allocator;
  }
}

bool DoesCpuAllocatorSupportArenaUsage() {
#if defined(USE_JEMALLOC) || defined(USE_MIMALLOC)
  // We use these allocators instead of the arena.
  return false;
#elif defined(ABSL_HAVE_ADDRESS_SANITIZER)
  // Using the arena may hide memory issues. Disable it in an ASan build.
  return false;
#else
  // Disable the arena for 32-bit builds because it may run into an infinite loop on integer overflow.
  if constexpr (sizeof(void*) == 4) {
    return false;
  } else {
    return true;
  }
#endif
}

}  // namespace onnxruntime
