// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "core/session/model_package/model_package_options.h"

#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_env.h"

namespace onnxruntime {

ModelPackageOptions::ModelPackageOptions(const OrtEnv& env, const OrtSessionOptions& session_options)
    : env_(env), session_options_(session_options) {}

Status ModelPackageOptions::ResolveEpSelection() {
  if (resolved_) {
    return Status::OK();
  }

  // TODO: reuse the existing session-options -> EP-devices resolution helper
  // that CreateSession already runs (same path used by InferenceSession when
  // session_options has V2-appended EP devices or a selection policy set).
  //
  // Expected shape:
  //   if (session_options_ has V2-appended EP devices) {
  //     ep_devices_ = those devices;
  //     ep_infos_   = BuildVariantSelectionEpInfos(ep_devices_, session_options_);
  //   } else if (session_options_ has EP selection policy) {
  //     ep_devices_ = EvaluatePolicyOverEnvDevices(env_, session_options_, /*graph=*/nullptr);
  //     ep_infos_   = BuildVariantSelectionEpInfos(ep_devices_, session_options_);
  //   } else {
  //     ep_devices_.clear();
  //     ep_infos_.clear();
  //   }
  //
  // Return a clear error if the policy requires graph inspection.

  resolved_ = true;
  return Status::OK();
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)