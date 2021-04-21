// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/get_execution_providers.h"

#include "core/graph/constants.h"

namespace onnxruntime {

namespace {
struct ProviderInfo {
  const char* name;
  bool available;
};

// all providers ordered by default priority from highest to lowest
// kCpuExecutionProvider should always be last
constexpr ProviderInfo kProvidersInPriorityOrder[] =
    {
        {
            kTensorrtExecutionProvider,
#ifdef USE_TENSORRT
            true,
#else
            false,
#endif
        },
        {
            kCudaExecutionProvider,
#ifdef USE_CUDA
            true,
#else
            false,
#endif
        },
        {
            kMIGraphXExecutionProvider,
#ifdef USE_MIGRAPHX
            true,
#else
            false,
#endif
        },
        {
            kRocmExecutionProvider,
#ifdef USE_ROCM
            true,
#else
            false,
#endif
        },
        {
            kOpenVINOExecutionProvider,
#ifdef USE_OPENVINO
            true,
#else
            false,
#endif
        },
        {
            kDnnlExecutionProvider,
#ifdef USE_DNNL
            true,
#else
            false,
#endif
        },
        {
            kNupharExecutionProvider,
#ifdef USE_NUPHAR
            true,
#else
            false,
#endif
        },
        {
            kVitisAIExecutionProvider,
#ifdef USE_VITISAI
            true,
#else
            false,
#endif
        },
        {
            kNnapiExecutionProvider,
#ifdef USE_NNAPI
            true,
#else
            false,
#endif
        },
        {
            kArmNNExecutionProvider,
#ifdef USE_ARMNN
            true,
#else
            false,
#endif
        },
        {
            kAclExecutionProvider,
#ifdef USE_ACL
            true,
#else
            false,
#endif
        },
        {
            kDmlExecutionProvider,
#ifdef USE_DML
            true,
#else
            false,
#endif
        },
        {
            kRknpuExecutionProvider,
#ifdef USE_RKNPU
            true,
#else
            false,
#endif
        },
        {
            kCoreMLExecutionProvider,
#ifdef USE_COREML
            true,
#else
            false,
#endif
        },
        {kCpuExecutionProvider, true},  // kCpuExecutionProvider is always last
};
}  // namespace

const std::vector<std::string>& GetAllExecutionProviderNames() {
  static const auto all_execution_providers = []() {
    std::vector<std::string> result{};
    for (const auto& provider : kProvidersInPriorityOrder) {
      result.push_back(provider.name);
    }
    return result;
  }();

  return all_execution_providers;
}

const std::vector<std::string>& GetAvailableExecutionProviderNames() {
  static const auto available_execution_providers = []() {
    std::vector<std::string> result{};
    for (const auto& provider : kProvidersInPriorityOrder) {
      if (provider.available) {
        result.push_back(provider.name);
      }
    }
    return result;
  }();

  return available_execution_providers;
}

}  // namespace onnxruntime
