// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <memory>
#include <vector>

#include "core/common/status.h"
#include "core/framework/execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/model_package/model_package_context.h"  // VariantSelectionEpInfo
#include "core/session/environment.h"

namespace onnxruntime {

class ModelPackageOptions {
 public:
  ModelPackageOptions(const Environment& env, const OrtSessionOptions& session_options);

  const OrtSessionOptions& SessionOptions() const noexcept { return session_options_; }

  Status RebuildProviderListForSession(const Environment& env) const;

  // Resolved state accessors
  std::vector<std::unique_ptr<IExecutionProvider>>& MutableProviderList() const noexcept { return provider_list_; }
  const std::vector<std::unique_ptr<IExecutionProvider>>& ProviderList() const noexcept { return provider_list_; }
  const std::vector<VariantSelectionEpInfo>& EpInfos() const noexcept { return ep_infos_; }
  const std::vector<const OrtEpDevice*>& ExecutionDevices() const noexcept { return execution_devices_; }
  const std::vector<const OrtEpDevice*>& DevicesSelected() const noexcept { return devices_selected_; }
  bool FromPolicy() const noexcept { return from_policy_; }

 private:
  void ResolveEpSelection(const Environment& env);

  OrtSessionOptions session_options_;

  // might needs to be rebuilt per session creation, as it becomes empty
  // after consumed by RegisterExecutionProvider(std::move(...)).
  mutable std::vector<std::unique_ptr<IExecutionProvider>> provider_list_{};

  std::vector<VariantSelectionEpInfo> ep_infos_{};
  std::vector<const OrtEpDevice*> execution_devices_{};
  std::vector<const OrtEpDevice*> devices_selected_{};
  bool from_policy_{false};
};

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)