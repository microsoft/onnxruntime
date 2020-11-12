// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/abi_session_options_impl.h"
#include "internal_testing_execution_provider.h"

using namespace onnxruntime;

namespace onnxruntime {

struct InternalTestingProviderFactory : IExecutionProviderFactory {
  InternalTestingProviderFactory(const std::unordered_set<std::string>& ops) : ops_(ops) {}
  ~InternalTestingProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

  const std::unordered_set<std::string> ops_;
};

std::unique_ptr<IExecutionProvider> InternalTestingProviderFactory::CreateProvider() {
  return onnxruntime::make_unique<InternalTestingExecutionProvider>(ops_);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_InternalTesting(
    const std::unordered_set<std::string>& ops) {
  return std::make_shared<onnxruntime::InternalTestingProviderFactory>(ops);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_InternalTesting, _In_ OrtSessionOptions* options) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_InternalTesting({}));
  return nullptr;
}
