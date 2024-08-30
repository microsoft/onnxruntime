// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <array>

namespace onnxruntime {
namespace lora {

// The current model versions for saving lora parameters in flatbuffers
// Once this version is updated, the kSupportedLoraFormatVersions in IsGenAiLoraFormatModelBytes
// below will also need to be updated.
// See src/flatbuffers/schema/README.md for more details on versioning.
// Version 1 - history begins
constexpr const int kLoraFormatVersion = 1;

// Check if the given lora format version is supported in this build
inline bool IsLoraFormatVersionSupported(const int lora_format_version) {
  // The lora format versions we will support in this build
  // This may contain more versions than the kLoraFormatVersion, based on the compatibilities
  static constexpr std::array<int, 1U> kSupportedLoraFormatVersions{
      kLoraFormatVersion,
  };

  const auto it =
      std::find(kSupportedLoraFormatVersions.begin(), kSupportedLoraFormatVersions.end(), lora_format_version);
  return it != kSupportedLoraFormatVersions.cend();
}

}  // namespace lora
}  // namespace onnxruntime
