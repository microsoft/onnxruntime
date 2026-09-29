// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "core/session/model_package/model_package_options.h"

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/graph/constants.h"
#include "core/providers/providers.h"
#include "core/session/provider_policy_context.h"
#include "core/session/utils.h"

namespace onnxruntime {

ModelPackageOptions::ModelPackageOptions(const Environment& env,
                                         const OrtSessionOptions& session_options) {
  ResolveEpSelection(env, session_options);
}

void ModelPackageOptions::ResolveEpSelection(const Environment& env,
                                             const OrtSessionOptions& session_options) {
  const bool has_provider_factories = !session_options.provider_factories.empty();
  from_policy_ = !has_provider_factories && session_options.value.ep_selection_policy.enable;

  provider_list_.clear();
  execution_devices_.clear();
  devices_selected_.clear();
  ep_infos_.clear();

  if (has_provider_factories) {
    const auto& logger = *logging::LoggingManager::DefaultLogger().ToExternal();
    for (auto& factory : session_options.provider_factories) {
      provider_list_.push_back(factory->CreateProvider(session_options, logger));
    }
    ORT_THROW_IF_ERROR(GetVariantSelectionEpInfo(provider_list_, ep_infos_));
  } else if (from_policy_) {
    OrtKeyValuePairs model_metadata;
    ProviderPolicyContext provider_policy_context;
    OrtSessionOptions mutable_session_options = session_options;
    ORT_THROW_IF_ERROR(provider_policy_context.SelectEpsForModelPackage(
        env, mutable_session_options, model_metadata,
        execution_devices_, devices_selected_, provider_list_));
    ORT_THROW_IF_ERROR(GetVariantSelectionEpInfo(provider_list_, ep_infos_));
  } else {
    // No explicit providers and no policy: default to CPU for variant selection.
    ep_infos_.push_back(VariantSelectionEpInfo{});
    ep_infos_.back().ep_name = kCpuExecutionProvider;
  }

  ORT_THROW_IF_ERROR(PrintAvailableAndSelectedEpInfos(env, ep_infos_));
}

Status ModelPackageOptions::RebuildProviderListForSession(const Environment& env,
                                                          const OrtSessionOptions& effective_options) const {
  provider_list_.clear();

  if (ep_infos_.empty()) {
    return Status::OK();
  }

  const auto& ep_info = ep_infos_[0];
  if (ep_info.ep_name == kCpuExecutionProvider || ep_info.ep_devices.empty()) {
    // CPU is built-in; no provider to register.
    return Status::OK();
  }

  std::unique_ptr<IExecutionProviderFactory> provider_factory;
  ORT_RETURN_IF_ERROR(CreateIExecutionProviderFactoryForEpDevices(
      env,
      gsl::span<const OrtEpDevice* const>(ep_info.ep_devices.data(), ep_info.ep_devices.size()),
      provider_factory));

  const auto& logger = *logging::LoggingManager::DefaultLogger().ToExternal();
  provider_list_.push_back(provider_factory->CreateProvider(effective_options, logger));

  return Status::OK();
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
