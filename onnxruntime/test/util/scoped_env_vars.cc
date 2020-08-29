// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/util/include/scoped_env_vars.h"

#ifndef WIN32
#include <stdlib.h>
#else  // WIN32
#endif

namespace onnxruntime {
namespace test {

namespace {
#ifndef WIN32
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

Status GetEnvironmentVar(const std::string& name, optional<std::string>& value) {
  const char* val = getenv(name.c_str());
  value = val == nullptr ? optional<std::string>{} : optional<std::string>{std::string{val}};
  return Status::OK();
}
#else  // WIN32
Status SetEnvironmentVar(const std::string& name, const optional<std::string>& value) {
  ORT_NOT_IMPLEMENTED();
}

Status GetEnvironmentVar(const std::string& name, optional<std::string>& value) {
  ORT_NOT_IMPLEMENTED();
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
    optional<std::string> env_var_value{};
    ORT_THROW_IF_ERROR(GetEnvironmentVar(env_var_name, env_var_value));
    result.insert({env_var_name, env_var_value});
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
