// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

class QuickGeluProgram final : public Program<QuickGeluProgram> {
 public:
  QuickGeluProgram(float32_t alpha) : Program{"QuickGelu"}, alpha_(alpha) {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"vec_size", ProgramUniformVariableDataType::Uint32});

 private:
  float32_t alpha_;
};

class QuickGelu final : public WebGpuKernel {
 public:
  QuickGelu(const OpKernelInfo& info) : WebGpuKernel(info) {
    alpha_ = info.GetAttr<float32_t>("alpha");
  }
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  float32_t alpha_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
