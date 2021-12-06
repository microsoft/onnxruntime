// Copyright 2019 AMD AMDMIGraphX

#include "core/framework/provider_options.h"
#include "onnxruntime_c_api.h"

namespace onnxruntime {
class IAllocator;
class IDataTransfer;
struct IExecutionProviderFactory;
struct MIGraphXExecutionProviderInfo;
enum class ArenaExtendStrategy : int32_t;
struct MIGraphXExecutionProviderExternalAllocatorInfo;

struct ProviderInfo_MIGRAPHX {
  virtual std::unique_ptr<onnxruntime::IAllocator> CreateHIPAllocator(int16_t device_id, const char* name) = 0;
  virtual std::unique_ptr<onnxruntime::IAllocator> CreateHIPPinnedAllocator(int16_t device_id, const char* name) = 0;
  virtual std::unique_ptr<onnxruntime::IDataTransfer> CreateGPUDataTransfer(void* stream) = 0;
};
}

