// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

class WhereProgram final : public Program<WhereProgram> {
 public:
  WhereProgram(bool is_broadcast) : Program{"Where"}, is_broadcast_{is_broadcast} {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"vec_size", ProgramUniformVariableDataType::Uint32});

 private:
  const bool is_broadcast_;
};

class Where final : public WebGpuKernel {
 public:
  Where(const OpKernelInfo& info) : WebGpuKernel{info} {
  }

  Status ComputeInternal(ComputeContext& context) const override;
};

// Factory functions for conditional int64 support (registered via RegisterKernels).
template <int StartVersion, int EndVersion>
KernelCreateInfo CreateWhereVersionedKernelInfo(bool enable_int64);
template <int SinceVersion>
KernelCreateInfo CreateWhereKernelInfo(bool enable_int64);

}  // namespace webgpu
}  // namespace onnxruntime
