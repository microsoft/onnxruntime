// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/session_options.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {

bool HasSessionConfigEntry(const SessionOptions& options, const std::string& config_key) {
  return options.session_configurations.find(config_key) != options.session_configurations.cend();
}

bool AddSessionConfigEntryImpl(SessionOptions& options, const char* config_key, const char* config_value) {
  std::string key(config_key);
  if (key.empty() || key.length() > 128) {
    LOGS_DEFAULT(ERROR) << "config_key is empty or longer than maximum length 128";
    return false;
  }

  std::string val(config_value);
  if (val.length() > 1024) {
    LOGS_DEFAULT(ERROR) << "config_value is longer than maximum length 1024";
    return false;
  }

  auto& configs = options.session_configurations;
  if (configs.find(key) != configs.end())
    LOGS_DEFAULT(WARNING) << "Session Config with key [" << key << "] already exists, it will be overwritten";

  configs[std::move(key)] = std::move(val);
  return true;
}
}  // namespace onnxruntime
