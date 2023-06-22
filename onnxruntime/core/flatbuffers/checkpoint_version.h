// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

// The versions below highlight the checkpoint format version history.
// Everytime the checkpoint format changes, the version should be incremented
// and the changes should be documented here and in the
// onnxruntime/core/flatbuffers/schema/README.md file.
// Version 1: Introduces the On-Device Training Checkpoint format
//            The format includes support for the ModuleState (stores the module parameters), OptimizerGroups
//            (stores the optimizer states), and PropertyBag
//            (stores custom user properties with support for int64, float and strings).
constexpr const int kCheckpointVersion = 1;

/**
 * @brief Check if the given checkpoint version is supported in this build
 * @param checkpoint_version The checkpoint version to check
 * @return true if the checkpoint version is supported, false otherwise
 */
inline constexpr bool IsCheckpointVersionSupported(const int checkpoint_version) {
  return kCheckpointVersion == checkpoint_version;
}

}  // namespace onnxruntime
