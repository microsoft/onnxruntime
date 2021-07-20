// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/common/optional.h"
#include "core/common/parse_string.h"

#ifndef SHARED_PROVIDER
#include "core/platform/get_env_var.h"
#endif

namespace onnxruntime {
/**
 * Parses an environment variable value if available (defined and not empty).
 */
template <typename T>
optional<T> ParseEnvironmentVariable(const std::string& name) {
  const auto value_str = GetEnvironmentVar(name);
  if (!value_str.has_value()) {
    return nullopt;
  }

  T parsed_value;
  ORT_ENFORCE(
      TryParseStringWithClassicLocale(value_str.value(), parsed_value),
      "Failed to parse environment variable - name: \"", name, "\", value: \"", value_str.value(), "\"");

  return parsed_value;
}

/**
 * Parses an environment variable value or returns the given default if unavailable.
 */
template <typename T>
T ParseEnvironmentVariableWithDefault(const std::string& name, const T& default_value) {
  return ParseEnvironmentVariable<T>(name).value_or(default_value);
}
}  // namespace onnxruntime
