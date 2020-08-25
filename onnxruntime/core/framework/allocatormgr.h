// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/arena.h"
#include "core/framework/bfc_arena.h"

namespace onnxruntime {

using DeviceAllocatorFactory = std::function<std::unique_ptr<IDeviceAllocator>(OrtDevice::DeviceId)>;

// TODO why does DeviceAllocatorRegistrationInfo have arena related configs?
// TODO even if it should, they should be inside their own struct (OrtArenaCfg) as opposed to
// littering them as individual members of DeviceAllocatorRegistrationInfo
struct DeviceAllocatorRegistrationInfo {
  DeviceAllocatorRegistrationInfo(OrtMemType ort_mem_type,
                                  DeviceAllocatorFactory alloc_factory,
                                  size_t mem,
                                  ArenaExtendStrategy strategy = BFCArena::DEFAULT_ARENA_EXTEND_STRATEGY,
                                  int initial_chunk_size_bytes0 = BFCArena::DEFAULT_INITIAL_CHUNK_SIZE_BYTES,
                                  int max_dead_bytes_per_chunk0 = BFCArena::DEFAULT_MAX_DEAD_BYTES_PER_CHUNK)
      : mem_type(ort_mem_type),
        factory(alloc_factory),
        max_mem(mem),
        arena_extend_strategy(strategy),
        initial_chunk_size_bytes(initial_chunk_size_bytes0),
        max_dead_bytes_per_chunk(max_dead_bytes_per_chunk0) {
  }

  OrtMemType mem_type;
  DeviceAllocatorFactory factory;
  size_t max_mem;
  ArenaExtendStrategy arena_extend_strategy;
  int initial_chunk_size_bytes;
  int max_dead_bytes_per_chunk;
};

AllocatorPtr CreateAllocator(const DeviceAllocatorRegistrationInfo& info, OrtDevice::DeviceId device_id = 0,
                             bool use_arena = true);

}  // namespace onnxruntime
