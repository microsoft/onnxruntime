// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace webgpu {

class Softmax final : public WebGpuKernel {
 public:
  Softmax(const OpKernelInfo& info) : WebGpuKernel{info} {
    opset_ = info.node().SinceVersion();
    int64_t axis = 0;
    Status status = info.GetAttr<int64_t>("axis", &axis);

    if (status.IsOK()) {
      axis_ = axis;
    } else {
      if (opset_ < 13) {
        axis_ = 1;  // opset-12 and below, the default axis value is 1
      } else {
        axis_ = -1;  // opset-13, the default axis value is -1
      }
    }
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t axis_;
  int opset_;
};

#define WEBGPU_SOFTMAX_PROGRAM_CONFIG(F) \
  F(uint32_t, wg_)                       \
  F(bool, is_fp32_)

struct SoftmaxProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_SOFTMAX_PROGRAM_CONFIG);
  static Config MakeConfig(uint32_t wg, bool is_fp32) {
    Config result{};
    result.wg_ = uint32_t{wg};
    result.is_fp32_ = bool{is_fp32};

    return result;
  }
  static constexpr std::string_view name = "Softmax";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_SOFTMAX_PROGRAM_SHADER_UNIFORMS(U) \
  U(Int32, packedCols)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_SOFTMAX_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_SOFTMAX_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_SOFTMAX_PROGRAM_CONFIG

using SoftmaxProgram = ConfiguredProgram<SoftmaxProgramShader>;

}  // namespace webgpu
}  // namespace onnxruntime
