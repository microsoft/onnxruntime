// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/kernel_registry.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/cpu/generator/constant_of_shape_base.h"

namespace onnxruntime {
namespace webgpu {

class ConstantOfShapeProgram final : public Program<ConstantOfShapeProgram> {
 public:
  ConstantOfShapeProgram() : Program{"ConstantOfShape"} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"value", ProgramUniformVariableDataType::Uint32});
};

class ConstantOfShape final : public WebGpuKernel, public ConstantOfShapeBase<> {
 public:
  explicit ConstantOfShape(const OpKernelInfo& info) : WebGpuKernel(info), ConstantOfShapeBase<>(info) {}

  Status ComputeInternal(ComputeContext& context) const override;
};

void RegisterConstantOfShapeKernels(KernelRegistry& kernel_registry, bool enable_int64);

}  // namespace webgpu
}  // namespace onnxruntime
