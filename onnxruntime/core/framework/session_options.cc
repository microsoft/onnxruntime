// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/session_options.h"
#include "core/common/logging/logging.h"
#include "core/framework/ml_value.h"

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

Status SessionOptions::AddConfigEntry(_In_z_ const char* config_key, _In_z_ const char* config_value) noexcept {
  std::string key(config_key);
  if (key.empty() || key.length() > 128)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Config key is empty or longer than maximum length 128");

  std::string val(config_value);
  if (val.length() > 1024)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Config value is longer than maximum length 1024");

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

Status SessionOptions::AddInitializer(_In_z_ const char* name, _In_ const OrtValue* val) noexcept {
  // input validation
  if (name == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Received nullptr for name.");
  }

  if (val == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Received nullptr for OrtValue.");
  }

  if (!val->IsTensor()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Received OrtValue is not a tensor. Only tensors are supported.");
  }

  if (val->Get<Tensor>().OwnsBuffer()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Buffer containing the initializer must be owned by the user.");
  }

  // now do the actual work
  auto rc = initializers_to_share_map.insert({name, val});
  if (!rc.second) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "An OrtValue for this name has already been added.");
  }

  return Status::OK();
}
}  // namespace onnxruntime
