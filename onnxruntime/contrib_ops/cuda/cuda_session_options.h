// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <charconv>
#include <string>
#include <system_error>
#include "core/providers/cuda/cuda_kernel.h"
#include "core/platform/env_var_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime::contrib::cuda {

// Call only from kernel construction. Explicit session configuration takes precedence over the environment.
inline int ReadCudaSessionOption(const OpKernelInfo& info, const char* key, const char* env,
                                 int default_value, int min_value, int max_value) {
  const auto configured = info.GetConfigOptions().GetConfigEntry(key);
  const std::string value = configured.has_value()
                                ? *configured
                                : ParseEnvironmentVariableWithDefault<std::string>(env, std::to_string(default_value));
  int parsed = 0;
  const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  ORT_ENFORCE(result.ec == std::errc{} && result.ptr == value.data() + value.size() &&
                  parsed >= min_value && parsed <= max_value,
              key, " (", env, ") must be an integer in [", min_value, ", ", max_value, "], got '", value, "'.");
  return parsed;
}

}  // namespace onnxruntime::contrib::cuda
