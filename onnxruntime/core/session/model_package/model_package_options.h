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
#include "core/session/provider_policy_context.h"

struct OrtEnv;

namespace onnxruntime {

// Snapshots the relevant session-level configuration (including EP selection)
// needed to open a model package and later create a session from it.
class ModelPackageOptions {
 public:
  ModelPackageOptions(const OrtSessionOptions& session_options);

  const OrtSessionOptions& SessionOptions() const noexcept { return session_options_; }

 private:
  OrtSessionOptions session_options_; // owned snapshot of caller-provided session options
};

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)