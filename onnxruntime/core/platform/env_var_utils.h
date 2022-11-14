// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_set>

#include "core/common/common.h"
#ifndef SHARED_PROVIDER
#include "core/common/logging/logging.h"
#endif
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
    return *parsed;
  }

  return default_value;
}

/**
 * Parses an environment variable value for testing convenience.
 *
 * This function ensures the value is valid and also produces a warning on its usage.
 */
template <typename T>
std::optional<T> ParseTestOnlyEnvironmentVariable(const std::string& name,
                                                  const std::unordered_set<std::string>& valid_values,
                                                  const std::string& hint = "") {
  ORT_ENFORCE(!valid_values.empty());

#ifndef SHARED_PROVIDER
  const std::string raw_env = Env::Default().GetEnvironmentVar(name);
#else
  const std::string raw_env = GetEnvironmentVar(name);
#endif
  if (raw_env.empty()) {
    return std::nullopt;
  }
  if (valid_values.find(raw_env) == valid_values.cend()) {
    std::ostringstream oss;
    auto it = valid_values.cbegin();
    oss << *it++;
    while(it != valid_values.cend()) {
      oss << ", " << *it++;
    }
    ORT_THROW("Value of environment variable ", name," must be ", oss.str(), ", but got ", raw_env);
  }

  auto env = onnxruntime::ParseEnvironmentVariable<T>(name);

  std::string default_hint = "End users should opt for provider options or session options.";
  const std::string& logged_hint = hint.empty() ? default_hint : hint;

  LOGS_DEFAULT(WARNING) << "Environment variable " << name << " is used. It is reserved for internal testing prupose. "
                        << logged_hint;

  return env;
}
}  // namespace onnxruntime
