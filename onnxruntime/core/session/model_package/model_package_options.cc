// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "core/session/model_package/model_package_options.h"

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
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
    size_t ep_index = 0;
    for (size_t i = 0; i < providers.size(); ++i) {
      if (providers[i]) {
        ep_infos_[ep_index++].provider_factory = session_options.provider_factories[i];
      }
    }
  } else if (session_options.value.ep_selection_policy.enable) {
    selection_policy_ = session_options.value.ep_selection_policy;
    OrtKeyValuePairs model_metadata;
    ProviderPolicyContext provider_policy_context;
    OrtSessionOptions mutable_session_options = session_options;
    InlinedVector<SelectionInfo> provider_selections;
    ORT_THROW_IF_ERROR(provider_policy_context.SelectEpsForModelPackage(
        env, mutable_session_options, model_metadata,
        execution_devices_, devices_selected_, providers, provider_selections));
    ORT_THROW_IF_ERROR(GetVariantSelectionEpInfo(providers, ep_infos_));

    // Save the devices chosen by the policy for each EP and the factory used to create it.
    // Internal and provider-bridge EPs do not expose their OrtEpDevice instances through GetEpDevices.
    // ORT checks each variant's required device type against the captured devices.
    // It also uses these devices to create the EP for a session.
    // The factory validates each variant's compatibility string.
    // Match providers to policy selections by index, not by name. An OrtEpDevice's EP name can differ from the provider
    // type, as with OpenVINO's meta-device factories ("OpenVINOExecutionProvider.AUTO").
    ORT_ENFORCE(ep_infos_.size() == provider_selections.size());
    for (size_t i = 0; i < ep_infos_.size(); ++i) {
      ep_infos_[i].ep_factory = provider_selections[i].ep_factory;
      ep_infos_[i].ep_devices = std::move(provider_selections[i].devices);
      ep_infos_[i].hardware_devices = std::move(provider_selections[i].hardware_devices);
      ep_infos_[i].ep_metadata = std::move(provider_selections[i].ep_metadata);
    }
  } else {
    // No explicit providers and no policy: default to CPU for variant selection.
    ep_infos_.push_back(VariantSelectionEpInfo{});
    ep_infos_.back().ep_name = kCpuExecutionProvider;
  }

  ORT_THROW_IF_ERROR(PrintAvailableAndSelectedEpInfos(env, ep_infos_));
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
