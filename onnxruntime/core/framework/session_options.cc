// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/session_options.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {

bool SessionOptions::TryGetConfigEntry(const std::string& config_key, std::string& config_value) const noexcept {
  bool found = false;
  config_value.clear();

  auto iter = session_configurations.find(config_key);
  if (iter != session_configurations.cend()) {
    found = true;
    config_value = iter->second;
  }

  return found;
}

const std::string SessionOptions::GetConfigOrDefault(const std::string& config_key,
                                                     const std::string& default_value) const noexcept {
  auto iter = session_configurations.find(config_key);
  return iter == session_configurations.cend() ? default_value : iter->second;
}

Status SessionOptions::AddConfigEntry(const char* config_key, const char* config_value) noexcept {
  std::string key(config_key);
  if (key.empty() || key.length() > 128)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "config_key is empty or longer than maximum length 128");

  std::string val(config_value);
  if (val.length() > 1024)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "config_value is longer than maximum length 1024");

  auto iter = session_configurations.find(config_key);
  if (iter != session_configurations.cend()) {
    LOGS_DEFAULT(WARNING) << "Session Config with key [" << key << "] already exists with value ["
                          << iter->second << "]. It will be overwritten";
    iter->second = std::move(val);
  } else {
    session_configurations[std::move(key)] = std::move(val);
  }

  return Status::OK();
}
}  // namespace onnxruntime
