// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>

#include "core/common/common.h"
#include "core/framework/error_code_helper.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/inference_session.h"
#include "core/session/model_package/model_package_context.h"
#include "core/session/model_package_api.h"
#include "core/session/ort_apis.h"
#include "stateful/bridge/stateful_bridge.h"
#include "stateful/model_package/model_package_chain.h"
#include "stateful/session/abi_stateful_impl.h"

using namespace onnxruntime;

namespace OrtModelPackageAPI {

ORT_API_STATUS_IMPL(CreateSessionChainImpl, const OrtEnv* env, const OrtModelPackageContext* context,
                    const OrtModelPackageChainConfig* config, OrtSessionChain** out) {
  API_IMPL_BEGIN
  constexpr size_t prefix = offsetof(OrtModelPackageChainConfig, num_step_session_options) +
                            sizeof(decltype(OrtModelPackageChainConfig::num_step_session_options));
  ORT_API_RETURN_IF(!env || !context || !out || !config || config->struct_size < prefix, ORT_INVALID_ARGUMENT,
                    "Invalid environment, package, output or chain config size");
  ORT_API_RETURN_IF(!config->chain_name || !*config->chain_name, ORT_INVALID_ARGUMENT,
                    "An explicit nonempty chain name is required");
  ORT_API_RETURN_IF(config->num_step_session_options && !config->step_session_options, ORT_INVALID_ARGUMENT,
                    "Per-step options require a non-NULL array");
  InlinedVector<std::unique_ptr<InferenceSession>> sessions;
  ModelPackageChainDefinition definition;
  std::string json;
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      reinterpret_cast<const ModelPackageContext*>(context)->GetChainJson(config->chain_name, json));
  const auto parsed = ParseModelPackageChain(json, definition);
  if (!parsed.IsOK()) {
    return ToOrtStatus(Status(parsed.Category(), parsed.Code(),
                              MakeString("chain '", config->chain_name, "': ", parsed.ErrorMessage())));
  }
  ORT_API_RETURN_IF(config->num_step_session_options > definition.options.components.size(), ORT_INVALID_ARGUMENT,
                    "Too many per-step options");
  InlinedHashMap<std::string, const OrtSessionOptions*> overrides;
  for (size_t i = 0; i < config->num_step_session_options; ++i) {
    const auto& item = config->step_session_options[i];
    ORT_API_RETURN_IF(!item.step_id || !*item.step_id || !item.session_options, ORT_INVALID_ARGUMENT,
                      "Step options require a nonempty ID and SessionOptions");
    const auto found = std::find_if(definition.options.components.begin(), definition.options.components.end(),
                                    [&](const auto& step) { return step.name == item.step_id; });
    ORT_API_RETURN_IF(
        found == definition.options.components.end() || !overrides.emplace(item.step_id, item.session_options).second,
        ORT_INVALID_ARGUMENT, "Unknown or duplicate step options ID");
  }
  OrtSessionOptions defaults;
  for (size_t i = 0; i < definition.options.components.size(); ++i) {
    auto& component = definition.options.components[i];
    const auto found = overrides.find(component.name);
    const auto* options = found == overrides.end() ? config->default_session_options : found->second;
    auto create_step = [&]() -> OrtStatus* {
      std::unique_ptr<OrtModelPackageOptions, decltype(&ReleaseModelPackageOptions)> selection(
          nullptr, ReleaseModelPackageOptions);
      OrtModelPackageOptions* raw_selection = nullptr;
      ORT_API_RETURN_IF_ERROR(
          CreateModelPackageOptionsFromSessionOptions(env, options ? options : &defaults, &raw_selection));
      selection.reset(raw_selection);
      std::unique_ptr<OrtModelPackageComponentContext, decltype(&ReleaseModelPackageComponentContext)> selected(
          nullptr, ReleaseModelPackageComponentContext);
      OrtModelPackageComponentContext* raw_component = nullptr;
      ORT_API_RETURN_IF_ERROR(
          SelectComponent(context, definition.component_names[i].c_str(), selection.get(), &raw_component));
      selected.reset(raw_component);
      OrtSession* raw_session = nullptr;
      ORT_API_RETURN_IF_ERROR(CreateSession(env, selected.get(), options, &raw_session));
      auto session = std::unique_ptr<InferenceSession>(reinterpret_cast<InferenceSession*>(raw_session));
      sessions.push_back(std::move(session));
      if (definition.loops[i]) {
        auto& loop_options = *definition.loops[i];
        const auto [status, outputs] = sessions.back()->GetModelOutputs();
        ORT_API_RETURN_IF_STATUS_NOT_OK(status);
        if (loop_options.ddim) {
          loop_options.output_names.push_back(loop_options.ddim->latent_output);
        } else {
          for (const auto* output : *outputs) loop_options.output_names.push_back(output->Name());
        }
        std::shared_ptr<const SessionLoop> loop;
        ORT_API_RETURN_IF_STATUS_NOT_OK(SessionLoop::Create(*sessions.back(), std::move(loop_options), loop));
        component.execution = std::move(loop);
      } else {
        component.execution = sessions.back().get();
      }
      return nullptr;
    };
    if (auto* error = create_step()) {
      std::unique_ptr<OrtStatus, decltype(&OrtApis::ReleaseStatus)> status(error, OrtApis::ReleaseStatus);
      return OrtApis::CreateStatus(
          OrtApis::GetErrorCode(error),
          MakeString("chain '", config->chain_name, "' step '", component.name, "': ", OrtApis::GetErrorMessage(error))
              .c_str());
    }
  }
  auto result = std::make_unique<OrtSessionChain>();
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      SessionChain::CreateOwned(std::move(definition.options), std::move(sessions), result->value));
  *out = result.release();
  return nullptr;
  API_IMPL_END
}

}  // namespace OrtModelPackageAPI
