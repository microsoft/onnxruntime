// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

class ExpandProgram final : public Program<ExpandProgram> {
 public:
  ExpandProgram() : Program{"Expand"} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"data_size", ProgramUniformVariableDataType::Uint32});
};

class Expand final : public WebGpuKernel {
 public:
  Expand(const OpKernelInfo& info) : WebGpuKernel(info) {}

  Status ComputeInternal(ComputeContext& context) const override;
};

// Create Expand kernel info with appropriate type constraints based on int64 support
template <int StartVersion, int EndVersion>
KernelCreateInfo CreateExpandVersionedKernelInfo(bool enable_int64);
template <int SinceVersion>
KernelCreateInfo CreateExpandKernelInfo(bool enable_int64);

}  // namespace webgpu
}  // namespace onnxruntime
