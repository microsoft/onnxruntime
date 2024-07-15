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

struct ProviderInfo_MIGraphX {
  virtual std::unique_ptr<onnxruntime::IAllocator> CreateMIGraphXAllocator(int16_t device_id, const char* name) = 0;
  virtual std::unique_ptr<onnxruntime::IAllocator> CreateMIGraphXPinnedAllocator(int16_t device_id, const char* name) = 0;

 protected:
  ~ProviderInfo_MIGraphX() = default;  // Can only be destroyed through a subclass instance
};

}  // namespace onnxruntime
