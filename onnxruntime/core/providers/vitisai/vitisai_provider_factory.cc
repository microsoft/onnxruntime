// Copyright (c) Xilinx Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vitisai/vitisai_provider_factory.h"
#include <atomic>
#include "vitisai_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {

struct VitisAIProviderFactory : IExecutionProviderFactory {
  VitisAIProviderFactory(std::string&& backend_type, int device_id) 
  : backend_type_(std::move(backend_type)), device_id_(device_id) {}
  ~VitisAIProviderFactory() = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  const std::string backend_type_;
  int device_id_;
};

std::unique_ptr<IExecutionProvider> VitisAIProviderFactory::CreateProvider() {
  VitisAIExecutionProviderInfo info;
  info.backend_type = backend_type_;
  info.device_id = device_id_;
  return std::make_unique<VitisAIExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_VITISAI(const char *backend_type, int device_id) {
  return std::make_shared<onnxruntime::VitisAIProviderFactory>(backend_type, device_id);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_VITISAI, _In_ OrtSessionOptions* options, _In_ const char* backend_type, int device_id) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_VITISAI(backend_type, device_id));
  return nullptr;
}

