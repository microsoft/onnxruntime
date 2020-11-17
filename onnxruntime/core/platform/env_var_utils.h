// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <sstream>

#include "core/common/common.h"
#include "core/common/optional.h"
#include "core/platform/env.h"

namespace onnxruntime {
/**
 * Parses an environment variable value if available (defined and not empty).
 */
template <typename T>
optional<T> ParseEnvironmentVariable(const std::string& name) {
  const std::string value_str = Env::Default().GetEnvironmentVar(name);
  if (value_str.empty()) {
    return {};
  }

  std::istringstream is{value_str};
  T parsed_value;
  ORT_ENFORCE(
      is >> std::noskipws >> parsed_value && is.eof(),
      "Failed to parse environment variable - name: \"", name, "\", value: \"", value_str, "\"");

  return parsed_value;
}

/**
 * Parses an environment variable value or returns the given default if unavailable.
 */
template <typename T>
T ParseEnvironmentVariable(const std::string& name, const T& default_value) {
  const auto parsed = ParseEnvironmentVariable<T>(name);
  if (parsed.has_value()) {
    return parsed.value();
  }

  return default_value;
}
}  // namespace onnxruntime
