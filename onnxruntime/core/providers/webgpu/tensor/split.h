// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/split.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_SPLIT_PROGRAM_CONFIG(F) F(size_t, output_count_)

struct SplitProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_SPLIT_PROGRAM_CONFIG);
  static Config MakeConfig(size_t output_count) {
    Config result{};
    result.output_count_ = size_t{output_count};

    return result;
  }
  static constexpr std::string_view name = "Split";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_SPLIT_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, input_size)                         \
  U(Uint32, total_segment_elements)             \
  U(Uint32, segment_sizes)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_SPLIT_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_SPLIT_PROGRAM_SHADER_UNIFORMS

  // The outputs are separate bindings, so the branch over them has to be unrolled at shader
  // generation time. Their sizes are uniforms, so shapes that differ only in split sizes share
  // one shader.
};
#undef WEBGPU_SPLIT_PROGRAM_CONFIG

using SplitProgram = ConfiguredProgram<SplitProgramShader>;

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
