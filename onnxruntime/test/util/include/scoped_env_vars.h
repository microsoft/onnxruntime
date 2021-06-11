// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unordered_map>
#include "core/common/optional.h"

#include "core/common/common.h"

namespace onnxruntime {
namespace test {

// map of environment variable name to optional value
// no value means the variable is not defined
using EnvVarMap = std::unordered_map<std::string, optional<std::string>>;

// Sets the given environment variables to their given values while in scope.
// The original values are reset on destruction.
class ScopedEnvironmentVariables {
 public:
  explicit ScopedEnvironmentVariables(const EnvVarMap& new_environment_variables);

  ~ScopedEnvironmentVariables();

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ScopedEnvironmentVariables);

  EnvVarMap original_environment_variables_;
};

}  // namespace test
}  // namespace onnxruntime
