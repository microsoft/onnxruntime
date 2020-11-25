// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <memory>

namespace onnxruntime {

// CodeGenTarget holds meta info for backend code generation
// and will be lowered to a target of corresponding backend
// code generation, e.g. TVM's Target.
class CodeGenTarget {
 public:
  CodeGenTarget() {}
  CodeGenTarget(const std::string& target_name)
      : target_name_(target_name) {}

  virtual int NaturalVectorWidth(int /*bits*/) const {
    return 1;
  }

  const std::string& GetTargetName() const {
    return target_name_;
  }

  virtual ~CodeGenTarget() = default;

 private:
  std::string target_name_{"unknown"};  // default name is unknown
};

}  // namespace onnxruntime
