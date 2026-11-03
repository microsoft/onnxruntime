// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime::contrib::webgpu {
using namespace onnxruntime::webgpu;

class QuickGeluMulF16Program final : public Program<QuickGeluMulF16Program> {
 public:
  QuickGeluMulF16Program() : Program{"QuickGeluMulF16"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"count", ProgramUniformVariableDataType::Uint32});
};

class QuickGeluMulF16 final : public WebGpuKernel {
 public:
  explicit QuickGeluMulF16(const OpKernelInfo& info) : WebGpuKernel(info) {}
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;
};
}  // namespace onnxruntime::contrib::webgpu
