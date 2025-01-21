// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/cpu/math/softmax.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

class Softmax final : public WebGpuKernel {
 public:
  Softmax(const OpKernelInfo& info) : WebGpuKernel{info} {
    int opset_ = info.node().SinceVersion();
    size_t axis;
    Status status = info.GetAttr<size_t>("axis", &axis);

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
  size_t axis_;
};

class SoftmaxProgram final : public Program<SoftmaxProgram> {
 public:
  SoftmaxProgram(size_t axis, int wg) : Program{"Softmax"}, axis_{axis}, WG_{wg} {
 }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"packedCols", ProgramUniformVariableDataType::Int32});

 private:
    int WG;
};

}  // namespace webgpu
}  // namespace onnxruntime
