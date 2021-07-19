// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/common/optional.h"
#include "core/common/parse_string.h"
#include "core/platform/env.h"

namespace onnxruntime {
/**
 * Parses an environment variable value if available (defined and not empty).
 */
template <typename T>
optional<T> ParseEnvironmentVariable(const std::string& name) {
#ifndef SHARED_PROVIDER
  const std::string value_str = Env::Default().GetEnvironmentVar(name);
#else
  const std::string value_str = GetEnvironmentVar(name);
#endif
  if (value_str.empty()) {
    return {};
  }

  T parsed_value;
  ORT_ENFORCE(
      TryParseStringWithClassicLocale(value_str, parsed_value),
      "Failed to parse environment variable - name: \"", name, "\", value: \"", value_str, "\"");

  return parsed_value;
}

/**
 * Parses an environment variable value or returns the given default if unavailable.
 */
template <typename T>
T ParseEnvironmentVariableWithDefault(const std::string& name, const T& default_value) {
  const auto parsed = ParseEnvironmentVariable<T>(name);
  if (parsed.has_value()) {
    return parsed.value();
  }

  return default_value;
}
}  // namespace onnxruntime
