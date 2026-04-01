// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <climits>
#include <cstdint>
#include <string_view>

#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime::version_check {

// Parse a non-negative integer from a string_view without leading zeros.
// Returns -1 on failure (empty, leading zero, non-digit, or overflow).
consteval int64_t ParseUint(std::string_view str) {
  if (str.empty()) return -1;
  // Leading zeros are not allowed (except "0" itself).
  if (str.size() > 1 && str[0] == '0') return -1;
  int64_t result = 0;
  for (char c : str) {
    if (c < '0' || c > '9') return -1;
    result = result * 10 + (c - '0');
    if (result > UINT32_MAX) return -1;
  }
  return result;
}

// Validates a version string at compile time.
// It must be in the format "1.Y.Z" where:
//   - Major version is 1
//   - Y and Z are non-negative integers without leading zeros
//   - Y (minor version) must equal expected_api_version (defaults to ORT_API_VERSION)
consteval bool IsOrtVersionValid(std::string_view version, uint32_t expected_api_version = ORT_API_VERSION) {
  size_t first_dot = version.find('.');
  if (first_dot == std::string_view::npos) return false;
  size_t second_dot = version.find('.', first_dot + 1);
  if (second_dot == std::string_view::npos) return false;
  if (version.find('.', second_dot + 1) != std::string_view::npos) return false;  // Exactly two dots
  std::string_view major = version.substr(0, first_dot);
  std::string_view minor = version.substr(first_dot + 1, second_dot - first_dot - 1);
  std::string_view patch = version.substr(second_dot + 1);
  if (major != "1") {
    return false;
  }
  int64_t minor_val = ParseUint(minor);
  int64_t patch_val = ParseUint(patch);
  if (minor_val < 0 || patch_val < 0) {
    return false;
  }
  if (static_cast<uint32_t>(minor_val) != expected_api_version) {
    return false;
  }
  return true;
}

}  // namespace onnxruntime::version_check
