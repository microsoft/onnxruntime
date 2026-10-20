// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace onnxruntime {
namespace webgpu {
namespace ep {

constexpr bool ShouldUseSerializedExecutionMode(uint32_t ort_api_version, uint32_t ort_patch_version,
                                                bool force_serialized_execution) {
  // Patch-level fixes in the 1.28 and 1.30 release lines do not imply support in 1.29.
  const bool supports_concurrent_sessions = ort_api_version >= 31 ||
                                            (ort_api_version == 30 && ort_patch_version >= 1) ||
                                            (ort_api_version == 28 && ort_patch_version >= 3);
  return force_serialized_execution || !supports_concurrent_sessions;
}

// Call after ApiInit(). This selects the caller's serialization contract, not a fixed CPU thread or a lock.
bool UseSerializedExecutionMode();

}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime
