// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/acl/acl_provider_factory.h"
#include <atomic>
#include "acl_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {

struct ACLProviderFactory : IExecutionProviderFactory {
  ACLProviderFactory(bool create_arena) : create_arena_(create_arena) {}
  ~ACLProviderFactory() override {}
  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  bool create_arena_;
};

std::unique_ptr<IExecutionProvider> ACLProviderFactory::CreateProvider() {
  ACLExecutionProviderInfo info;
  info.create_arena = create_arena_;
  return std::make_unique<ACLExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ACL(int use_arena) {
  return std::make_shared<onnxruntime::ACLProviderFactory>(use_arena != 0);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_ACL, _In_ OrtSessionOptions* options, int use_arena) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_ACL(use_arena));
  return nullptr;
}
