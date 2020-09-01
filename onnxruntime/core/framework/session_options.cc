// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/session_options.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {

bool SessionOptions::HasConfigEntry(const std::string& config_key) const noexcept {
  return session_configurations.find(config_key) != session_configurations.cend();
}

const std::string SessionOptions::GetConfigOrDefault(const std::string& config_key,
                                                     const std::string& default_value) const noexcept {
  if (!HasConfigEntry(config_key))
    return default_value;

  return session_configurations.at(config_key);
}

Status SessionOptions::AddConfigEntry(const char* config_key, const char* config_value) noexcept {
  std::string key(config_key);
  if (key.empty() || key.length() > 128)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "config_key is empty or longer than maximum length 128");

  std::string val(config_value);
  if (val.length() > 1024)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "config_value is longer than maximum length 1024");

  if (HasConfigEntry(config_key))
    LOGS_DEFAULT(WARNING) << "Session Config with key [" << key << "] already exists, it will be overwritten";

  session_configurations[std::move(key)] = std::move(val);
  return Status::OK();
}
}  // namespace onnxruntime
