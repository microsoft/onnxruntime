// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include <atomic>

#include "core/providers/hip/hip_provider_factory.h"
#include "core/session/abi_session_options_impl.h"

#include "core/providers/hip/hip_execution_provider.h"

using namespace onnxruntime;

namespace onnxruntime {

struct HIPProviderFactory : IExecutionProviderFactory {
  HIPProviderFactory(int device_id) : device_id_(device_id) {}
  ~HIPProviderFactory() = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    HIPExecutionProviderInfo info;
    info.device_id = device_id_;
    return std::make_unique<HIPExecutionProvider>(info);
  }

private:
  int device_id_;
};

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_HIP(int device_id) {
  return std::make_shared<onnxruntime::HIPProviderFactory>(device_id);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_HIP, _In_ OrtSessionOptions* options, int device_id) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_HIP(device_id));
  return nullptr;
}
