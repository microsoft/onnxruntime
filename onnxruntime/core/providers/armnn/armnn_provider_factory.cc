// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/armnn/armnn_provider_factory.h"
#include "armnn_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {

struct ArmNNProviderFactory : IExecutionProviderFactory {
  ArmNNProviderFactory(bool create_arena) : create_arena_(create_arena) {}
  ~ArmNNProviderFactory() override {}
  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  bool create_arena_;
};

std::unique_ptr<IExecutionProvider> ArmNNProviderFactory::CreateProvider() {
  ArmNNExecutionProviderInfo info;
  info.create_arena = create_arena_;
  return std::make_unique<ArmNNExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ArmNN(int use_arena) {
  return std::make_shared<onnxruntime::ArmNNProviderFactory>(use_arena != 0);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_ArmNN, _In_ OrtSessionOptions* options, int use_arena) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_ArmNN(use_arena));
  return nullptr;
}
