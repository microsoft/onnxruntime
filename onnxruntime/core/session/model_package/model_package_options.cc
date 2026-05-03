// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "core/session/model_package/model_package_options.h"

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/providers/providers.h"
#include "core/session/provider_policy_context.h"
#include "core/session/utils.h"

namespace onnxruntime {

ModelPackageOptions::ModelPackageOptions(const Environment& env,
                                         const OrtSessionOptions& session_options)
    : session_options_(session_options) {
  ResolveEpSelection(env);
}

void ModelPackageOptions::ResolveEpSelection(const Environment& env) {
  const bool has_provider_factories = !session_options_.provider_factories.empty();
  from_policy_ = !has_provider_factories && session_options_.value.ep_selection_policy.enable;

  provider_list_.clear();
  execution_devices_.clear();
  devices_selected_.clear();
  ep_infos_.clear();

  if (has_provider_factories) {
    const auto& logger = *logging::LoggingManager::DefaultLogger().ToExternal();
    for (auto& factory : session_options_.provider_factories) {
      provider_list_.push_back(factory->CreateProvider(session_options_, logger));
    }
  } else if (from_policy_) {
    OrtKeyValuePairs model_metadata;
    ProviderPolicyContext provider_policy_context;
    OrtSessionOptions mutable_session_options = session_options_;
    ORT_THROW_IF_ERROR(provider_policy_context.SelectEpsForModelPackage(
        env, mutable_session_options, model_metadata,
        execution_devices_, devices_selected_, provider_list_));
  }

  ORT_THROW_IF_ERROR(GetVariantSelectionEpInfo(provider_list_, ep_infos_));
  ORT_THROW_IF_ERROR(PrintAvailableAndSelectedEpInfos(env, ep_infos_));
}

// Rebuild the provider list based on the selected EP devices.
// The model package context instance (contains the model package options) can be used
// with multiple session creation calls, and the provider list is consumed (moved) when
// registering providers to each session, so we need to rebuild it for each session creation.
Status ModelPackageOptions::RebuildProviderListForSession(const Environment& env) const {
  provider_list_.clear();

  const bool has_provider_factories = !session_options_.provider_factories.empty();

  if (has_provider_factories) {
    const auto& logger = *logging::LoggingManager::DefaultLogger().ToExternal();
    for (auto& factory : session_options_.provider_factories) {
      provider_list_.push_back(factory->CreateProvider(session_options_, logger));
    }
    return Status::OK();
  }

  // Policy path: reconstruct providers from the already-selected EP devices.
  if (from_policy_ && !devices_selected_.empty()) {
    std::unique_ptr<IExecutionProviderFactory> provider_factory;
    ORT_RETURN_IF_ERROR(CreateIExecutionProviderFactoryForEpDevices(
        env,
        gsl::span<const OrtEpDevice* const>(devices_selected_.data(), devices_selected_.size()),
        provider_factory));

    const auto& logger = *logging::LoggingManager::DefaultLogger().ToExternal();
    provider_list_.push_back(provider_factory->CreateProvider(session_options_, logger));
  }

  return Status::OK();
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)