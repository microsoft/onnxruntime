// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"

namespace onnxruntime {
namespace webgpu {

enum class ScatterElementsReduction : int {
  None = 0,
  Add = 1,
  Mul = 2,
  Min = 3,
  Max = 4,
};

#define WEBGPU_SCATTER_ELEMENTS_PROGRAM_CONFIG(F) \
  F(int64_t, axis_)                               \
  F(ScatterElementsReduction, reduction_)         \
  F(int32_t, data_type_)

struct ScatterElementsProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_SCATTER_ELEMENTS_PROGRAM_CONFIG);
  static Config MakeConfig(int64_t axis, ScatterElementsReduction reduction, MLDataType data_type) {
    Config result{};
    result.axis_ = int64_t(axis);
    result.reduction_ = ScatterElementsReduction(reduction);
    result.data_type_ = int32_t(data_type->AsPrimitiveDataType()->GetDataType());

    return result;
  }
  static constexpr std::string_view name = "ScatterElements";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_SCATTER_ELEMENTS_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)                                   \
  U(Uint32, axis_dim_limit)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_SCATTER_ELEMENTS_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_SCATTER_ELEMENTS_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_SCATTER_ELEMENTS_PROGRAM_CONFIG

using ScatterElementsProgram = ConfiguredProgram<ScatterElementsProgramShader>;

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
