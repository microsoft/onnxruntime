// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

class BinaryElementwiseProgram final : public Program<BinaryElementwiseProgram> {
 public:
  BinaryElementwiseProgram(const std::string& kernel_name,
                           const std::string& expression,
                           const std::string& additional_impl,
                           const bool is_broadcast,
                           const bool is_lhs_scalar,
                           const bool is_rhs_scalar,
                           const bool vectorize) : Program{kernel_name},
                                                   expression_{expression},
                                                   additional_impl_{additional_impl},
                                                   is_broadcast_{is_broadcast},
                                                   is_lhs_scalar_{is_lhs_scalar},
                                                   is_rhs_scalar_{is_rhs_scalar},
                                                   vectorize_{vectorize} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"vec_size", ProgramUniformVariableDataType::Uint32});

 private:
  std::string_view expression_;
  std::string_view additional_impl_;
  bool is_broadcast_;
  bool is_lhs_scalar_;
  bool is_rhs_scalar_;
  bool vectorize_;
};

class BinaryElementwise : public WebGpuKernel {
 public:
  using GetAdditionalImplementationFunction = std::string (*)(int lhs_element_type, int rhs_element_type);

  BinaryElementwise(const OpKernelInfo& info,
                    const std::string& kernel_name,
                    const std::string& expression,
                    const GetAdditionalImplementationFunction get_additional_impl = nullptr) : WebGpuKernel{info},
                                                                                               kernel_name_{kernel_name},
                                                                                               expression_{expression},
                                                                                               get_additional_impl_{get_additional_impl} {}

 protected:
  Status ComputeInternal(ComputeContext& context) const final;

 private:
  std::string kernel_name_;
  std::string expression_;
  const GetAdditionalImplementationFunction get_additional_impl_;
};

}  // namespace webgpu
}  // namespace onnxruntime
