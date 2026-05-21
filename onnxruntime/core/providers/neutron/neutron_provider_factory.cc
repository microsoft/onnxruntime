// Copyright 2024-2026 NXP
// SPDX-License-Identifier: MIT

// #include "core/providers/shared_library/provider_api.h"

#include "core/providers/neutron/neutron_provider_factory.h"
#include "core/providers/neutron/neutron_execution_provider.h"
#include "core/providers/neutron/neutron_provider_factory_creator.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {

struct NeutronProviderFactory : IExecutionProviderFactory {
  NeutronProviderFactory(NeutronProviderOptions neutron_options) : neutron_options_(neutron_options) {}
  ~NeutronProviderFactory() override = default;
  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  NeutronProviderOptions neutron_options_;
};

std::unique_ptr<IExecutionProvider> NeutronProviderFactory::CreateProvider() {
  return std::make_unique<NeutronExecutionProvider>(neutron_options_);
}

std::shared_ptr<IExecutionProviderFactory> NeutronProviderFactoryCreator::Create(
    NeutronProviderOptions neutron_options) {
  return std::make_shared<onnxruntime::NeutronProviderFactory>(neutron_options);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Neutron,
                    _In_ OrtSessionOptions* options, NeutronProviderOptions neutron_options) {
  options->provider_factories.push_back(onnxruntime::NeutronProviderFactoryCreator::Create(neutron_options));
  return nullptr;
}
