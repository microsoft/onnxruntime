// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/config_options.h"
#include "core/common/common.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {

optional<std::string> ConfigOptions::GetConfigEntry(const std::string& config_key) const noexcept {
  if (auto it = configurations.find(config_key); it != configurations.end()) {
    return it->second;
  }
  return nullopt;
}

bool ConfigOptions::TryGetConfigEntry(const std::string& config_key, std::string& config_value) const noexcept {
  config_value.clear();
  auto entry = GetConfigEntry(config_key);
  const bool found = entry.has_value();
  if (found) {
    config_value = std::move(*entry);
  }
  return found;
}

std::string ConfigOptions::GetConfigOrDefault(const std::string& config_key,
                                              const std::string& default_value) const noexcept {
  return GetConfigEntry(config_key).value_or(default_value);
}

Status ConfigOptions::AddConfigEntry(const char* config_key, const char* config_value) noexcept {
  std::string key(config_key);
  if (key.empty() || key.length() > 128)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Config key is empty or longer than maximum length 128");

  std::string val(config_value);
  if (val.length() > onnxruntime::kMaxStrLen)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Config value is longer than maximum length: ",
                           onnxruntime::kMaxStrLen);

  auto iter = configurations.find(config_key);
  if (iter != configurations.cend()) {
    LOGS_DEFAULT(WARNING) << "Config with key [" << key << "] already exists with value ["
                          << iter->second << "]. It will be overwritten";
    iter->second = std::move(val);
  } else {
    configurations[std::move(key)] = std::move(val);
  }

  return Status::OK();
}

std::ostream& operator<<(std::ostream& os, const ConfigOptions& config_options) {
  for (const auto& [key, value] : config_options.configurations) {
    os << "  " << key << ": " << value;
  }
  return os;
}

}  // namespace onnxruntime
