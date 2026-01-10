// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

class DepthToSpaceProgram final : public Program<DepthToSpaceProgram> {
 public:
  DepthToSpaceProgram(int64_t* perm) : Program{"DepthToSpace"}, perm_{perm} {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32});

 private:
  int64_t* perm_;
};

template <bool is_nhwc>
class DepthToSpace final : public WebGpuKernel {
 public:
  DepthToSpace(const OpKernelInfo& info) : WebGpuKernel(info) {
    blocksize_ = info.GetAttr<int64_t>("blocksize");
    std::string mode = info.GetAttrOrDefault<std::string>("mode", "DCR");
    is_dcr_ = (mode == "DCR");
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t blocksize_;
  bool is_dcr_;
};

}  // namespace webgpu
}  // namespace onnxruntime