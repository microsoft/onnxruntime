// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <charconv>
#include <optional>
#include <string>
#include <system_error>
#include "core/providers/cuda/cuda_kernel.h"
#include "core/platform/env_var_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime::contrib::cuda {

// An explicit configured value takes precedence over the environment.
inline std::string ResolveCudaSessionOptionText(const std::optional<std::string>& configured, const char* env,
                                                const std::string& default_text) {
  return configured.has_value() ? *configured : ParseEnvironmentVariableWithDefault<std::string>(env, default_text);
}

// Integer threshold: throws unless the result is an integer in [min_value, max_value].
inline int ResolveCudaSessionOption(const std::optional<std::string>& configured, const char* key, const char* env,
                                    int default_value, int min_value, int max_value) {
  const std::string value = ResolveCudaSessionOptionText(configured, env, std::to_string(default_value));
  int parsed = 0;
  const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  ORT_ENFORCE(result.ec == std::errc{} && result.ptr == value.data() + value.size() &&
                  parsed >= min_value && parsed <= max_value,
              key, " (", env, ") must be an integer in [", min_value, ", ", max_value, "], got '", value, "'.");
  return parsed;
}

// 0/1 switch: throws unless the result is exactly "0" or "1" (an integer parse would also take "00", "01" or "-0").
inline bool ResolveCudaSessionSwitch(const std::optional<std::string>& configured, const char* key, const char* env,
                                     bool default_value) {
  const std::string value = ResolveCudaSessionOptionText(configured, env, default_value ? "1" : "0");
  ORT_ENFORCE(value == "0" || value == "1", key, " (", env, ") must be exactly \"0\" or \"1\", got '", value, "'.");
  return value == "1";
}

// Call only from kernel construction. Explicit session configuration takes precedence over the environment.
inline int ReadCudaSessionOption(const OpKernelInfo& info, const char* key, const char* env,
                                 int default_value, int min_value, int max_value) {
  return ResolveCudaSessionOption(info.GetConfigOptions().GetConfigEntry(key), key, env, default_value, min_value,
                                  max_value);
}

inline bool ReadCudaSessionSwitch(const OpKernelInfo& info, const char* key, const char* env, bool default_value) {
  return ResolveCudaSessionSwitch(info.GetConfigOptions().GetConfigEntry(key), key, env, default_value);
}

}  // namespace onnxruntime::contrib::cuda
