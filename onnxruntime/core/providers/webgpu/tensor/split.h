// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/split.h"

namespace onnxruntime {
namespace webgpu {

class SplitProgram final : public Program<SplitProgram> {
 public:
  explicit SplitProgram(size_t output_count) : Program{"Split"}, output_count_{output_count} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"input_size", ProgramUniformVariableDataType::Uint32},
                                          {"total_segment_elements", ProgramUniformVariableDataType::Uint32},
                                          {"segment_sizes", ProgramUniformVariableDataType::Uint32});

 private:
  // The outputs are separate bindings, so the branch over them has to be unrolled at shader
  // generation time. Their sizes are uniforms, so shapes that differ only in split sizes share
  // one shader.
  size_t output_count_;
};

class Split : public WebGpuKernel, public SplitBase {
 public:
  Split(const OpKernelInfo& info, uint32_t opset) : WebGpuKernel(info), SplitBase(info, opset) {}

 protected:
  Status ComputeInternal(ComputeContext& context) const override;
};

class Split_1 final : public Split {
 public:
  Split_1(const OpKernelInfo& info) : Split(info, 1) {}
};

class Split_2_10 final : public Split {
 public:
  Split_2_10(const OpKernelInfo& info) : Split(info, 2) {}
};

class Split_11_12 final : public Split {
 public:
  Split_11_12(const OpKernelInfo& info) : Split(info, 11) {}
};

class Split_13_17 final : public Split {
 public:
  Split_13_17(const OpKernelInfo& info) : Split(info, 13) {}
};

class Split_18 final : public Split {
 public:
  Split_18(const OpKernelInfo& info) : Split(info, 18) {}
};

}  // namespace webgpu
}  // namespace onnxruntime
