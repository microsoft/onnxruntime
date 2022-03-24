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
}

