// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/cum_sum.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    CumSum,
    kOnnxDomain,
    11, 13,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()),
    CumSum);

ONNX_OPERATOR_KERNEL_EX(
    CumSum,
    kOnnxDomain,
    14,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()),
    CumSum);

Status CumSumProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& input = shader.AddInput("input", ShaderUsage::UseUniform);
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform);

  std::string index = "i32(" + input.IndicesGet("uniforms.input_shape", "uniforms.axis", input.Rank()) + ")";
  std::string max = GetElementAt("uniforms.input_shape", "uniforms.axis", input.Rank());
  std::string lowerLimit = reverse_ ? index + (exclusive_ ? " + 1" : "") : "0";
  std::string upperLimit = reverse_ ? max : index + (exclusive_ ? "" : " + 1");
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "var input_indices = " << input.OffsetToIndices("global_idx") << ";\n"
                            << "var sum = output_indices_t(0);\n"
                            << "let first : i32 = " << lowerLimit << ";\n"
                            << "let last : i32 = " << upperLimit << ";\n"
                            << "for (var i : i32 = first; i < last; i++) {\n"
                            << "  " << input.IndicesSet("input_indices", "uniforms.axis", "u32(i)") << ";\n"
                            << "  sum = sum + " << input.GetByIndices("input_indices") << ";\n"
                            << "}\n"
                            << output.SetByOffset("global_idx", "sum") << ";\n";

  return Status::OK();
}

Status CumSum::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  const TensorShape& input_shape = input_tensor->Shape();
  int64_t input_rank = input_shape.NumDimensions();

  const auto* axis_tensor = context.Input(1);
  const auto* axis_data = axis_tensor->Data<int64_t>();

  // Handle negative axis
  int64_t axis = axis_data[0];
  if (axis < 0) {
    axis += input_rank;
  }

  auto* output_tensor = context.Output(0, input_shape);
  int64_t output_size = output_tensor->Shape().Size();

  if (output_size == 0) {
    return Status::OK();
  }

  CumSumProgram program{exclusive_, reverse_};
  program
      .AddInput({input_tensor})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::TypeAndRank})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(output_size)},
                            {static_cast<uint32_t>(axis)}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
