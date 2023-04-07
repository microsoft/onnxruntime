// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "core/session/onnxruntime_c_api.h"
#include <unordered_map>

namespace onnxruntime {

using AllocatorFactory = std::function<std::unique_ptr<IAllocator>(OrtDevice::DeviceId)>;

constexpr int DEFAULT_CPU_ALLOCATOR_DEVICE_ID = 0;

struct AllocatorCreationInfo {
  AllocatorCreationInfo(AllocatorFactory device_alloc_factory,
                        OrtDevice::DeviceId device_id = 0,
                        bool use_arena = true,
                        OrtArenaCfg arena_cfg = {0, -1, -1, -1, -1},
                        bool stream_aware_arena = false,
                        bool cross_stream_reusing = false)
      : device_alloc_factory(device_alloc_factory),
        device_id(device_id),
        use_arena(use_arena),
        arena_cfg(arena_cfg),
        use_stream_aware_arena(stream_aware_arena),
        enable_cross_stream_reusing(cross_stream_reusing) {
  }

  AllocatorFactory device_alloc_factory;
  OrtDevice::DeviceId device_id;
  bool use_arena;
  OrtArenaCfg arena_cfg;
  bool use_stream_aware_arena;
  bool enable_cross_stream_reusing;
};

// Returns an allocator (an instance of IAllocator) based on the creation info provided.
// Returns nullptr if an invalid value of info.arena_cfg.arena_extend_strategy is supplied.
// Valid values can be found in onnxruntime_c_api.h.
AllocatorPtr CreateAllocator(const AllocatorCreationInfo& info);

// Used for sharing allocators across EPs. e.g. CUDA with TensorRT, CPU with XNNPACK
// NOTE: This isn't managing the lifetime of the allocators (they're all in shared_ptr instances anyway).
// It really just provides a way to collect and pass around allocators between the calls to
// IExecutionProvider::RegisterAllocator for each registered EP. Once those calls complete it is no longer needed.
class AllocatorManager {
  //
 public:
  AllocatorManager() = default;
  void InsertAllocator(AllocatorPtr allocator);
  void ReplaceAllocator(AllocatorPtr allocator);
  // Get an allocator for the device. Return nullptr if it doesn't exist
  AllocatorPtr GetAllocator(OrtMemType mem_type, OrtDevice device) const;

 private:
  // key from OrtMemType+OrtDevice mapped to AllocatorPtr
  using AllocatorMap = std::unordered_map<int32_t, AllocatorPtr>;
  AllocatorMap allocators_;
};

}  // namespace onnxruntime
