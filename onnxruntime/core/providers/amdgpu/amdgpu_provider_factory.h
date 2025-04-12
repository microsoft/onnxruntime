// Copyright 2019 AMD AMDAMDGPU

#pragma once

#include <core/providers/amdgpu/amdgpu_execution_provider_info.h>

namespace onnxruntime {

class IAllocator;

struct ProviderInfo_AMDGPU {
  virtual std::unique_ptr<IAllocator> CreateAMDGPUAllocator(OrtDevice::DeviceId device_id, const char* name) = 0;
  virtual std::unique_ptr<IAllocator> CreateAMDGPUPinnedAllocator(OrtDevice::DeviceId device_id, const char* name) = 0;
  virtual void AMDGPUMemcpy_HostToDevice(void* dst, const void* src, size_t count) = 0;
  virtual void AMDGPUMemcpy_DeviceToHost(void* dst, const void* src, size_t count) = 0;
  virtual std::shared_ptr<IAllocator> CreateAMDGPUAllocator(OrtDevice::DeviceId device_id, size_t migx_mem_limit,
    ArenaExtendStrategy arena_extend_strategy, AMDGPUExecutionProviderExternalAllocatorInfo& external_allocator_info,
    const OrtArenaCfg* default_memory_arena_cfg) = 0;

 protected:
  ~ProviderInfo_AMDGPU() = default;  // Can only be destroyed through a subclass instance
};

}  // namespace onnxruntime
