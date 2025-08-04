// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

class GatherNDProgram final : public Program<GatherNDProgram> {
 public:
  GatherNDProgram(const uint32_t batch_dims, const uint32_t indices_innerest_dim) : Program{"GatherND"},
                                                                                    batch_dims_{batch_dims},
                                                                                    indices_innerest_dim_{indices_innerest_dim} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"data_size", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t batch_dims_;
  uint32_t indices_innerest_dim_;
};

class GatherNDBase : public WebGpuKernel {
 public:
  GatherNDBase(const OpKernelInfo& info) : WebGpuKernel(info) {
    info.GetAttrOrDefault("batch_dims", &batch_dims_, static_cast<int64_t>(0));
    ORT_ENFORCE(batch_dims_ >= 0);
  }

 protected:
  int64_t batch_dims_;
};

class GatherND final : public GatherNDBase {
 public:
  GatherND(const OpKernelInfo& info) : GatherNDBase(info) {}

 protected:
  Status ComputeInternal(ComputeContext& context) const override;
};

}  // namespace webgpu
}  // namespace onnxruntime
