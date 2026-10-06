// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include "core/platform/env_var.h"

namespace onnxruntime {

std::string GetEnvironmentVar(const std::string& var_name) {
  return detail::GetEnvironmentVar(var_name);
}

}  // namespace onnxruntime