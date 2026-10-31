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

  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  execution_devices_.clear();
  devices_selected_.clear();
  ep_infos_.clear();

  if (has_provider_factories) {
    const auto& logger = *logging::LoggingManager::DefaultLogger().ToExternal();
    for (auto& factory : session_options.provider_factories) {
      providers.push_back(factory->CreateProvider(session_options, logger));
    }
    ORT_THROW_IF_ERROR(GetVariantSelectionEpInfo(providers, ep_infos_));
    if (!ep_infos_.empty()) {
      ep_infos_.front().provider_factory = session_options.provider_factories.front();
    }
  } else if (from_policy_) {
    OrtKeyValuePairs model_metadata;
    ProviderPolicyContext provider_policy_context;
    OrtSessionOptions mutable_session_options = session_options;
    ORT_THROW_IF_ERROR(provider_policy_context.SelectEpsForModelPackage(
        env, mutable_session_options, model_metadata,
        execution_devices_, devices_selected_, providers));
    ORT_THROW_IF_ERROR(GetVariantSelectionEpInfo(providers, ep_infos_, devices_selected_));
  } else {
    // No explicit providers and no policy: default to CPU for variant selection.
    ep_infos_.push_back(VariantSelectionEpInfo{});
    ep_infos_.back().ep_name = kCpuExecutionProvider;
  }

  ORT_THROW_IF_ERROR(PrintAvailableAndSelectedEpInfos(env, ep_infos_));
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
