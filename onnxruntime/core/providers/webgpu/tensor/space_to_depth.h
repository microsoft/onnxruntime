// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

class SpaceToDepthProgram final : public Program<SpaceToDepthProgram> {
 public:
  SpaceToDepthProgram(int64_t* perm) : Program{"SpaceToDepth"}, perm_{perm} {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32});

 private:
  int64_t* perm_;
};

template <bool is_nhwc>
class SpaceToDepth final : public WebGpuKernel {
 public:
  SpaceToDepth(const OpKernelInfo& info) : WebGpuKernel(info) {
    blocksize_ = info.GetAttr<int64_t>("blocksize");
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t blocksize_;
};

}  // namespace webgpu
}  // namespace onnxruntime
