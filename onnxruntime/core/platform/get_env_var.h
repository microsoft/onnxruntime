// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/common/optional.h"

namespace onnxruntime {

/**
 * Gets the environment variable value if the variable is defined.
 */
optional<std::string> GetEnvironmentVar(const std::string& var_name);

/**
 * Gets the environment variable value or an empty string if the variable is not defined.
 */
inline std::string GetEnvironmentVarOrEmpty(const std::string& var_name) {
  return GetEnvironmentVar(var_name).value_or("");
}

}  // namespace onnxruntime
