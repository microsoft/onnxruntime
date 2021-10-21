// Copyright 2020 rock-chips.com Inc.

#include "core/providers/rknpu/rknpu_provider_factory.h"
#include "rknpu_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {

struct RknpuProviderFactory : IExecutionProviderFactory {
  RknpuProviderFactory() {}
  ~RknpuProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
};

std::unique_ptr<IExecutionProvider> RknpuProviderFactory::CreateProvider() {
  return std::make_unique<RknpuExecutionProvider>();
}

std::shared_ptr<IExecutionProviderFactory>
CreateExecutionProviderFactory_Rknpu() {
  return std::make_shared<onnxruntime::RknpuProviderFactory>();
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Rknpu,
                    _In_ OrtSessionOptions* options) {
  options->provider_factories.push_back(
      onnxruntime::CreateExecutionProviderFactory_Rknpu());
  return nullptr;
}
