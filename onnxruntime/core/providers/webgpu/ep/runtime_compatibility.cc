// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "runtime_compatibility.h"

#include "ep/api.h"

#include "core/common/common.h"
#include "core/platform/env_var.h"

namespace onnxruntime {
namespace webgpu {
namespace ep {

bool UseSerializedExecutionMode() {
  static const bool serialized_execution = [] {
    // Retain the existing override name for compatibility with callers and CI.
    const auto force_serialized_execution = onnxruntime::detail::GetEnvironmentVar("ORT_WEBGPU_EP_FORCE_LEGACY");
    ORT_ENFORCE(force_serialized_execution.empty() || force_serialized_execution == "0" || force_serialized_execution == "1",
                "ORT_WEBGPU_EP_FORCE_LEGACY must be 0 or 1.");
    return ShouldUseSerializedExecutionMode(onnxruntime::ep::CurrentOrtApiVersion(),
                                            onnxruntime::ep::CurrentOrtPatchVersion(), force_serialized_execution == "1");
  }();
  return serialized_execution;
}

}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime
