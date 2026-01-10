// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace webgpu {

class Softmax final : public WebGpuKernel {
 public:
  Softmax(const OpKernelInfo& info) : WebGpuKernel{info} {
    opset_ = info.node().SinceVersion();
    int64_t axis;
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

class SoftmaxProgram final : public Program<SoftmaxProgram> {
 public:
  SoftmaxProgram(uint32_t wg, bool is_fp32)
      : Program{"Softmax"}, wg_{wg}, is_fp32_{is_fp32} {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"packedCols", ProgramUniformVariableDataType::Int32});

 private:
  uint32_t wg_;
  bool is_fp32_;
};

}  // namespace webgpu
}  // namespace onnxruntime
