// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string_view>

#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime::version_check {

// A simple constexpr-friendly result type for ParseUint.
struct ParseUintResult {
  uint32_t value;
  bool has_value;

  constexpr bool operator==(uint32_t other) const { return has_value && value == other; }
  constexpr bool operator!=(uint32_t other) const { return !(*this == other); }
};

inline constexpr ParseUintResult ParseUintNone() { return {0, false}; }

// Parse a non-negative integer from a string_view without leading zeros.
// Returns a result with has_value == false on failure (empty, leading zero, non-digit, or overflow).
constexpr ParseUintResult ParseUint(std::string_view str) {
  if (str.empty()) return ParseUintNone();
  // Leading zeros are not allowed (except "0" itself).
  if (str.size() > 1 && str[0] == '0') return ParseUintNone();
  uint64_t result = 0;
  for (char c : str) {
    if (c < '0' || c > '9') return ParseUintNone();
    result = result * 10 + static_cast<uint64_t>(c - '0');
    if (result > UINT32_MAX) return ParseUintNone();
  }
  return {static_cast<uint32_t>(result), true};
}

// Validates a version string at compile time.
// It must be in the format "1.Y.Z" where:
//   - Major version is 1
//   - Y and Z are non-negative integers without leading zeros
//   - Y (minor version) must equal expected_api_version (defaults to ORT_API_VERSION)
constexpr bool IsOrtVersionValid(std::string_view version, uint32_t expected_api_version = ORT_API_VERSION) {
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
  auto minor_val = ParseUint(minor);
  auto patch_val = ParseUint(patch);
  if (!minor_val.has_value || !patch_val.has_value) {
    return false;
  }
  if (minor_val.value != expected_api_version) {
    return false;
  }
  return true;
}

}  // namespace onnxruntime::version_check
