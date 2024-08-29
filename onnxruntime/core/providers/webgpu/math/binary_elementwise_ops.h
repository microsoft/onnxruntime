// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

class BinaryElementwiseProgram final : public Program<BinaryElementwiseProgram> {
 public:
  BinaryElementwiseProgram(const std::string& kernel_name, const std::string& expression, const std::string& additional_impl = "")
      : Program{kernel_name}, expression_{expression}, additional_impl_{additional_impl} {}

  BinaryElementwiseProgram(const std::string& kernel_name, const std::string& expression, const std::string& expression_vec4, const std::string& additional_impl)
      : Program{kernel_name}, expression_{expression}, expression_vec4_{expression_vec4}, additional_impl_{additional_impl} {}

  BinaryElementwiseProgram(const std::string& kernel_name, const std::string& expression, std::function<std::string(int a_type)> func)
      : Program{kernel_name}, expression_{expression}, additional_impl_func_{func} {}

  BinaryElementwiseProgram(const std::string& kernel_name, const std::string& expression, const std::string& expression_vec4, std::function<std::string(int a_type)> func)
      : Program{kernel_name}, expression_{expression}, expression_vec4_{expression_vec4}, additional_impl_func_{func} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"vec_size", ProgramUniformVariableDataType::Uint32});

 private:
  std::string expression_;
  std::string expression_vec4_;
  std::string additional_impl_;
  std::function<std::string(int a_type)> additional_impl_func_;
};

}  // namespace webgpu
}  // namespace onnxruntime
