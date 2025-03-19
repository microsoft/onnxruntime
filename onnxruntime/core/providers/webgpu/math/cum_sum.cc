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
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()).TypeConstraint("T2", WebGpuSupportedNumberTypes()).InputMemoryType(OrtMemTypeCPU, 1),
    CumSum);

ONNX_OPERATOR_KERNEL_EX(
    CumSum,
    kOnnxDomain,
    14,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()).TypeConstraint("T2", WebGpuSupportedNumberTypes()).InputMemoryType(OrtMemTypeCPU, 1),
    CumSum);

Status CumSumProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& input = shader.AddInput("input", ShaderUsage::UseUniform);
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform);

  std::string index = "i32(" + input.IndicesGet("input_indices", "uniforms.axis") + ")";
  std::string max = GetElementAt("uniforms.input_shape", "uniforms.axis", input.Rank());
  std::stringstream lowerLimit;
  lowerLimit << "let first : i32 = 0;\n"
             << "if (uniforms.reverse == 1) {\n"
             << "  first = " << index << ";\n"
             << "  if (uniforms.exclusive == 1) { first += 1; }\n"
             << "}\n";
  std::stringstream upperLimit;
  upperLimit << "let last : i32 = 0;\n"
             << "if (uniforms.reverse == 1) {\n"
             << "  last = " << max << ";\n"
             << "} else {\n"
             << "  last = " << index << ";\n"
             << "  if (uniforms.exclusive == 0) { last += 1; }\n"
             << "}\n";
  // std::string lowerLimit = reverse_ ? index + (exclusive_ ? " + 1" : "") : "0";
  // std::string upperLimit = reverse_ ? max : index + (exclusive_ ? "" : " + 1");
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "var input_indices = " << input.OffsetToIndices("global_idx") << ";\n"
                            << "var sum = output_indices_t(0);\n"
                            << lowerLimit.str()
                            << upperLimit.str()
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
  int64_t axis = axis_data[0];

  bool valid_axis = true;
  if (axis < -input_rank || axis >= input_rank) {
    valid_axis = false;
  }
  ORT_ENFORCE(valid_axis, "Axes attribute must be within range -input_rank <= axis < input_rank.");
  // Handle negative axis
  if (axis < 0) {
    axis += input_rank;
  }

  auto* output_tensor = context.Output(0, input_shape);
  int64_t output_size = output_tensor->Shape().Size();

  if (output_size == 0) {
    return Status::OK();
  }

  CumSumProgram program{};
  program
      .AddInput({input_tensor})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::TypeAndRank})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(output_size)},
                            {static_cast<uint32_t>(axis)},
                            {static_cast<uint32_t>(exclusive_)},
                            {static_cast<uint32_t>(reverse_)}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
