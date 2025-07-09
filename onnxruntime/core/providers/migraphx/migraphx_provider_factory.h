// Copyright 2019 AMD AMDMIGraphX

#include "core/framework/ortdevice.h"
#include "onnxruntime_c_api.h"

namespace onnxruntime {
class IAllocator;
class IDataTransfer;
struct IExecutionProviderFactory;
struct MIGraphXExecutionProviderInfo;
enum class ArenaExtendStrategy : int32_t;
struct MIGraphXExecutionProviderExternalAllocatorInfo;

struct ProviderInfo_MIGraphX {
  virtual std::unique_ptr<onnxruntime::IAllocator> CreateMIGraphXAllocator(OrtDevice::DeviceId device_id, const char* name) = 0;
  virtual std::unique_ptr<onnxruntime::IAllocator> CreateMIGraphXPinnedAllocator(OrtDevice::DeviceId device_id, const char* name) = 0;
  virtual void MIGraphXMemcpy_HostToDevice(void* dst, const void* src, size_t count) = 0;
  virtual void MIGraphXMemcpy_DeviceToHost(void* dst, const void* src, size_t count) = 0;
  virtual std::shared_ptr<onnxruntime::IAllocator> CreateMIGraphXAllocator(OrtDevice::DeviceId device_id, size_t migx_mem_limit, onnxruntime::ArenaExtendStrategy arena_extend_strategy, onnxruntime::MIGraphXExecutionProviderExternalAllocatorInfo& external_allocator_info, const OrtArenaCfg* default_memory_arena_cfg) = 0;

 protected:
  ~ProviderInfo_MIGraphX() = default;  // Can only be destroyed through a subclass instance
};

}  // namespace onnxruntime
