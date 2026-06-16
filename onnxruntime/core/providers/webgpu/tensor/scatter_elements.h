// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

enum class ScatterElementsReduction : int {
  None = 0,
  Add = 1,
  Mul = 2,
  Min = 3,
  Max = 4,
};

class ScatterElementsProgram final : public Program<ScatterElementsProgram> {
 public:
  ScatterElementsProgram(int64_t axis, ScatterElementsReduction reduction, MLDataType data_type)
      : Program{"ScatterElements"}, axis_(axis), reduction_(reduction), data_type_(data_type) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"axis_dim_limit", ProgramUniformVariableDataType::Uint32});

 private:
  int64_t axis_;
  ScatterElementsReduction reduction_;
  MLDataType data_type_;
};

class ScatterElements : public WebGpuKernel {
 public:
  ScatterElements(const OpKernelInfo& info) : WebGpuKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK(),
                "Missing/Invalid 'axis' attribute value");

    std::string reduction = info.GetAttrOrDefault<std::string>("reduction", "none");
    if (reduction == "add") {
      reduction_ = ScatterElementsReduction::Add;
    } else if (reduction == "mul") {
      reduction_ = ScatterElementsReduction::Mul;
    } else if (reduction == "min") {
      reduction_ = ScatterElementsReduction::Min;
    } else if (reduction == "max") {
      reduction_ = ScatterElementsReduction::Max;
    } else if (reduction == "none") {
      reduction_ = ScatterElementsReduction::None;
    } else {
      ORT_THROW("Reduction '", reduction, "' is not supported.");
    }
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t axis_;
  ScatterElementsReduction reduction_{ScatterElementsReduction::None};
};

}  // namespace webgpu
}  // namespace onnxruntime
