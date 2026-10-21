// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <memory>
#include <optional>
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
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ModelPackageOptions);

  // Resolved state accessors
  const std::vector<VariantSelectionEpInfo>& EpInfos() const noexcept { return ep_infos_; }
  const std::vector<const OrtEpDevice*>& ExecutionDevices() const noexcept { return execution_devices_; }
  const std::vector<const OrtEpDevice*>& DevicesSelected() const noexcept { return devices_selected_; }
  // Returns the policy used to select the EPs, if any.
  const std::optional<EpSelectionPolicy>& SelectionPolicy() const noexcept { return selection_policy_; }

 private:
  void ResolveEpSelection(const Environment& env, const OrtSessionOptions& session_options);

  std::vector<VariantSelectionEpInfo> ep_infos_{};
  std::vector<const OrtEpDevice*> execution_devices_{};
  std::vector<const OrtEpDevice*> devices_selected_{};
  std::optional<EpSelectionPolicy> selection_policy_{};
};

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
