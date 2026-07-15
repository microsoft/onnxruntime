// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/framework/data_transfer_manager.h"

namespace onnxruntime {
namespace webgpu {

enum class ScatterNDReduction : int {
  None = 0,
  Add = 1,
  Mul = 2,
  Min = 3,
  Max = 4,
};

class ScatterNDProgram final : public Program<ScatterNDProgram> {
 public:
  ScatterNDProgram(ScatterNDReduction reduction, MLDataType data_type) : Program{"ScatterND"}, reduction_(reduction), data_type_(data_type) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"last_index_dimension", ProgramUniformVariableDataType::Uint32},
                                          {"num_updates_elements", ProgramUniformVariableDataType::Uint32});
  ScatterNDReduction reduction_;
  MLDataType data_type_;
};

class ScatterND : public WebGpuKernel {
 public:
  ScatterND(const OpKernelInfo& info) : WebGpuKernel(info) {
    std::string reduction = info.GetAttrOrDefault<std::string>("reduction", "none");
    if (reduction == "add") {
      reduction_ = ScatterNDReduction::Add;
    } else if (reduction == "mul") {
      reduction_ = ScatterNDReduction::Mul;
    } else if (reduction == "min") {
      reduction_ = ScatterNDReduction::Min;
    } else if (reduction == "max") {
      reduction_ = ScatterNDReduction::Max;
    } else if (reduction == "none") {
      reduction_ = ScatterNDReduction::None;
    } else {
      ORT_THROW("Reduction '", reduction, "' is not supported on webgpu when opset <= 18.");
    }
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  ScatterNDReduction reduction_{ScatterNDReduction::None};
};

}  // namespace webgpu
}  // namespace onnxruntime
