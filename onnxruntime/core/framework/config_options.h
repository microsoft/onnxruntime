// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include "core/common/common.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {

/**
  * Configuration options that can be used by any struct by inheriting this class.
  * Provides infrastructure to add/get config entries
  */
struct ConfigOptions {
  std::unordered_map<std::string, std::string> configurations;

  // Check if this instance of ConfigOptions has a config using the given config_key.
  // Returns true if found and copies the value into config_value.
  // Returns false if not found and clears config_value.
  bool TryGetConfigEntry(const std::string& config_key, std::string& config_value) const noexcept;

  // Get the config string in this instance of ConfigOptions using the given config_key
  // If there is no such config, the given default string will be returned
  const std::string GetConfigOrDefault(const std::string& config_key, const std::string& default_value) const noexcept;

  // Add a config pair (config_key, config_value) to this instance of ConfigOptions
  Status AddConfigEntry(_In_z_ const char* config_key, _In_z_ const char* config_value) noexcept;
};

}  // namespace onnxruntime
