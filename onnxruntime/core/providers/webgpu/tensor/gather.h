// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/cpu/tensor/gatherbase.h"

namespace onnxruntime {
namespace webgpu {

class GatherProgram final : public Program<GatherProgram> {
 public:
  GatherProgram(const uint32_t axis) : Program{"Gather"}, axis_{axis} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"data_size", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t axis_;
};

class Gather final : public WebGpuKernel, public GatherBase {
 public:
  Gather(const OpKernelInfo& info) : WebGpuKernel(info), GatherBase(info) {}

 protected:
  Status ComputeInternal(ComputeContext& context) const override;
};

}  // namespace webgpu
}  // namespace onnxruntime
