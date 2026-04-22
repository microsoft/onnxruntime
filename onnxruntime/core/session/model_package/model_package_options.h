// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <vector>

#include "core/common/status.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/model_package/model_package_context.h"  // VariantSelectionEpInfo

struct OrtEnv;

namespace onnxruntime {

// Backing object for the opaque OrtModelPackageOptions C type.
// Snapshots the relevant session-level configuration (including EP selection)
// needed to open a model package and later create a session from it.
class ModelPackageOptions {
 public:
  ModelPackageOptions(const OrtEnv& env, const OrtSessionOptions& session_options);

  // Resolves EP selection once. Idempotent. Safe to call eagerly in the C API entry.
  Status ResolveEpSelection();

  const OrtEnv& Env() const noexcept { return env_; }
  const OrtSessionOptions& SessionOptions() const noexcept { return session_options_; }
  const std::vector<VariantSelectionEpInfo>& EpInfos() const noexcept { return ep_infos_; }
  const std::vector<const OrtEpDevice*>& EpDevices() const noexcept { return ep_devices_; }

 private:
  const OrtEnv& env_;
  OrtSessionOptions session_options_;  // value-copied snapshot
  std::vector<VariantSelectionEpInfo> ep_infos_{};
  std::vector<const OrtEpDevice*> ep_devices_{};
  bool resolved_{false};
};

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)