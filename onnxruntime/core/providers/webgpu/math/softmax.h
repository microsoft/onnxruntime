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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_SOFTMAX_PROGRAM_CONFIG);
    Config(uint32_t wg, bool is_fp32) : wg_{wg}, is_fp32_{is_fp32} {}
  };
  static constexpr std::string_view name = "Softmax";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"packedCols", ProgramUniformVariableDataType::Int32});
};
#undef WEBGPU_SOFTMAX_PROGRAM_CONFIG

using SoftmaxProgram = ConfiguredProgram<SoftmaxProgramShader>;

}  // namespace webgpu
}  // namespace onnxruntime
