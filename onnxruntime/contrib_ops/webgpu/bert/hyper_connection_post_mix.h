// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime::contrib::webgpu {

using namespace onnxruntime::webgpu;

class HyperConnectionPostMixProgram final : public Program<HyperConnectionPostMixProgram> {
 public:
  HyperConnectionPostMixProgram(int gate_layout, bool has_stream_mix)
      : Program{"HyperConnectionPostMix"},
        gate_layout_(gate_layout),
        has_stream_mix_(has_stream_mix) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"count", ProgramUniformVariableDataType::Uint32},
      {"branches", ProgramUniformVariableDataType::Uint32},
      {"hidden", ProgramUniformVariableDataType::Uint32});

 private:
  int gate_layout_;
  bool has_stream_mix_;
};

class HyperConnectionPostMix final : public WebGpuKernel {
 public:
  explicit HyperConnectionPostMix(const OpKernelInfo& info);
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  int64_t num_branches_;
};

}  // namespace onnxruntime::contrib::webgpu
