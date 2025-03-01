// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/opencl/opencl_provider_factory.h"
#include "core/framework/execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "opencl_execution_provider.h"

#include <atomic>

namespace onnxruntime {
struct OpenCLExecutionProviderFactory final : IExecutionProviderFactory {
  OpenCLExecutionProviderInfo info;

  OpenCLExecutionProviderFactory() = default;
  ~OpenCLExecutionProviderFactory() final = default;
  std::unique_ptr<IExecutionProvider> CreateProvider() override;
};

std::unique_ptr<IExecutionProvider> OpenCLExecutionProviderFactory::CreateProvider() {
  return std::make_unique<OpenCLExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenCL(bool use_fp16, int auto_runing_level) {
  auto factory = std::make_shared<onnxruntime::OpenCLExecutionProviderFactory>();
  factory->info.use_fp16 = use_fp16;
  factory->info.auto_runing_level = auto_runing_level;
  return factory;
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_OpenCL, _In_ OrtSessionOptions* options, int use_fp16, int auto_runing_level) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_OpenCL(bool(use_fp16), auto_runing_level));
  return nullptr;
}
