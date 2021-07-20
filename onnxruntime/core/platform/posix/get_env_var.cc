// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/get_env_var.h"

#include <stdlib.h>

namespace onnxruntime {
optional<std::string> GetEnvironmentVar(const std::string& var_name) {
  const char* val = ::getenv(var_name.c_str());
  return val != nullptr ? optional<std::string>{val} : nullopt;
}
}  // namespace onnxruntime
