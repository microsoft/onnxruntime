// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/providers/migraphx/migraphx_provider_factory.h"
#include <atomic>
#include "migraphx_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {
struct MIGraphXProviderFactory : IExecutionProviderFactory {
  MIGraphXProviderFactory(int device_id) : device_id_(device_id) {}
  ~MIGraphXProviderFactory() = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    MIGraphXExecutionProviderInfo info;
    info.device_id = device_id_;
    info.target_device = "gpu";
    return std::make_unique<MIGraphXExecutionProvider>(info);
  }

private:
  int device_id_;
};

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_MIGraphX(int device_id) {
  return std::make_shared<onnxruntime::MIGraphXProviderFactory>(device_id);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_MIGraphX, _In_ OrtSessionOptions* options, int device_id) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_MIGraphX(device_id));
  return nullptr;
}
