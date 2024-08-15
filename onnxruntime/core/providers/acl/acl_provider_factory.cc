// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include "core/providers/acl/acl_provider_factory.h"
#include <atomic>
#include "acl_execution_provider.h"
#include "acl_provider_factory_creator.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {

struct ACLProviderFactory : IExecutionProviderFactory {
  ACLProviderFactory(bool enable_fast_math) : enable_fast_math_(enable_fast_math) {}
  ~ACLProviderFactory() override {}
  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  bool enable_fast_math_;
};

std::unique_ptr<IExecutionProvider> ACLProviderFactory::CreateProvider() {
  ACLExecutionProviderInfo info;
  info.enable_fast_math = enable_fast_math_;
  return std::make_unique<ACLExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> ACLProviderFactoryCreator::Create(bool enable_fast_math) {
  return std::make_shared<onnxruntime::ACLProviderFactory>(enable_fast_math);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_ACL, _In_ OrtSessionOptions* options,
      bool enable_fast_math) {
  options->provider_factories.push_back(onnxruntime::ACLProviderFactoryCreator::Create(enable_fast_math));
  return nullptr;
}
