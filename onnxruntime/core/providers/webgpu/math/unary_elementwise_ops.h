// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

class UnaryElementwiseProgram final : public Program<UnaryElementwiseProgram> {
 public:
  UnaryElementwiseProgram(const std::string& kernel_name, std::string_view expression, std::string_view additional_impl, ShaderVariable::Usage usage)
      : Program{kernel_name}, expression_{expression}, additional_impl_{additional_impl}, additional_usage_{usage} {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"vec_size", ProgramUniformVariableDataType::Uint32},  // output size
      {"attr", ProgramUniformVariableDataType::Float32});    // float type attribute(s)
                                                             // TODO: add u32/i32 attribute(s) if needed

 private:
  std::string_view expression_;
  std::string_view additional_impl_;
  ShaderVariable::Usage additional_usage_;
};

// TODO: after upgrading to C++20, use consteval to make a compile-time constructor so that it will be safe to switch
//       the std::string to std::string_view. This will avoid the cost of copying the string.

class UnaryElementwise : public WebGpuKernel {
 public:
  UnaryElementwise(const OpKernelInfo& info,
                   const std::string& kernel_name,
                   const std::string& expression,
                   const std::string& additional_impl = "",
                   ShaderVariable::Usage usage = ShaderVariable::None) : WebGpuKernel{info},
                                                                         kernel_name_{kernel_name},
                                                                         expression_{expression},
                                                                         additional_impl_{additional_impl},
                                                                         additional_usage_{usage} {}

 protected:
  std::string cache_hint;

  Status ComputeInternal(ComputeContext& context) const final;
  virtual Status ConfigureProgram(const ComputeContext& /*context*/, UnaryElementwiseProgram& program) const {
    program.UniformVariables({{}});  // empty for attribute(s)
    return Status::OK();
  }

 private:
  std::string kernel_name_;
  std::string expression_;
  std::string additional_impl_;
  ShaderVariable::Usage additional_usage_;
};

}  // namespace webgpu
}  // namespace onnxruntime
