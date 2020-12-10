// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/string_utils.h"
#include "core/framework/provider_options.h"

namespace onnxruntime {
/**
 * Reads the named provider option.
 * Returns true if the option is present, false otherwise.
 */
template <typename T>
bool ReadProviderOption(const ProviderOptions& options, const std::string& key, T& value) {
  auto it = options.find(key);
  if (it != options.end()) {
    ORT_ENFORCE(
        TryParse(it->second, value),
        "Failed to parse provider option \"", key, "\" with value \"", it->second, "\".");
    return true;
  }
  return false;
}

/**
 * Reads the named provider option.
 * Returns the value if the option is present or the specified default value otherwise.
 */
template <typename T>
T ReadProviderOptionOrDefault(
    const ProviderOptions& options, const std::string& key, const T& default_value) {
  T value{};
  if (ReadProviderOption(options, key, value)) {
    return value;
  }
  return default_value;
}
}  // namespace onnxruntime
