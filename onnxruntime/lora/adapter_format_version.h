// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <array>

namespace onnxruntime {
namespace adapters {

// The current model versions for saving lora parameters in flatbuffers format.
// Once this version is updated, the kSupportedAdapterFormatVersions in IsAdapterFormatVersionSupported
// below will also need to be updated.
// See onnxruntime/lora/adapter_format/README.md for more details on versioning.
// Version 1 - history begins
constexpr const int kAdapterFormatVersion = 1;

// Check if the given lora format version is supported in this build
inline bool IsAdapterFormatVersionSupported(const int lora_format_version) {
  // The lora format versions we will support in this build
  // This may contain more versions than the kAdapterFormatVersion, based on the compatibilities
  static constexpr std::array<int, 1U> kSupportedAdapterFormatVersions{
      kAdapterFormatVersion,
  };

  const auto it =
      std::find(kSupportedAdapterFormatVersions.begin(), kSupportedAdapterFormatVersions.end(), lora_format_version);
  return it != kSupportedAdapterFormatVersions.cend();
}

}  // namespace adapters
}  // namespace onnxruntime
