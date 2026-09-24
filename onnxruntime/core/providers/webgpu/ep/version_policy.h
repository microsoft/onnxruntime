// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string_view>

namespace onnxruntime::webgpu::ep {

inline bool IsAdditionalOrtVersion(std::string_view runtime_version, std::string_view additional_versions) {
  if (runtime_version.empty()) {
    return false;
  }

  while (!additional_versions.empty()) {
    const auto separator = additional_versions.find(',');
    if (runtime_version == additional_versions.substr(0, separator)) {
      return true;
    }
    if (separator == std::string_view::npos) {
      break;
    }
    additional_versions.remove_prefix(separator + 1);
  }
  return false;
}

}  // namespace onnxruntime::webgpu::ep
