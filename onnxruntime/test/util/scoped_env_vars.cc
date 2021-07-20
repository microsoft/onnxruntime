// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/util/include/scoped_env_vars.h"

#ifndef _WIN32
#include <stdlib.h>
#else  // _WIN32
#include <Windows.h>
#endif

#include "core/platform/get_env_var.h"

namespace onnxruntime {
namespace test {

namespace {
#ifndef _WIN32

Status SetEnvironmentVar(const std::string& name, const optional<std::string>& value) {
  if (value.has_value()) {
    ORT_RETURN_IF_NOT(
        setenv(name.c_str(), value.value().c_str(), 1) == 0,
        "setenv() failed: ", errno);
  } else {
    ORT_RETURN_IF_NOT(
        unsetenv(name.c_str()) == 0,
        "unsetenv() failed: ", errno);
  }
  return Status::OK();
}

#else  // _WIN32

Status SetEnvironmentVar(const std::string& name, const optional<std::string>& value) {
  ORT_RETURN_IF_NOT(
      SetEnvironmentVariableA(name.c_str(), value.has_value() ? value.value().c_str() : nullptr) != 0,
      "SetEnvironmentVariableA() failed: ", GetLastError());
  return Status::OK();
}

#endif

void SetEnvironmentVars(const EnvVarMap& env_vars) {
  for (const auto& env_var : env_vars) {
    ORT_THROW_IF_ERROR(SetEnvironmentVar(env_var.first, env_var.second));
  }
}

EnvVarMap GetEnvironmentVars(const std::vector<std::string>& env_var_names) {
  EnvVarMap result{};
  for (const auto& env_var_name : env_var_names) {
    result.insert({env_var_name, GetEnvironmentVar(env_var_name)});
  }
  return result;
}
}  // namespace

ScopedEnvironmentVariables::ScopedEnvironmentVariables(const EnvVarMap& new_env_vars) {
  std::vector<std::string> new_env_var_names{};
  std::transform(
      new_env_vars.begin(), new_env_vars.end(), std::back_inserter(new_env_var_names),
      [](const EnvVarMap::value_type& new_env_var) { return new_env_var.first; });

  original_environment_variables_ = GetEnvironmentVars(new_env_var_names);

  SetEnvironmentVars(new_env_vars);
}

ScopedEnvironmentVariables::~ScopedEnvironmentVariables() {
  SetEnvironmentVars(original_environment_variables_);
}

}  // namespace test
}  // namespace onnxruntime
