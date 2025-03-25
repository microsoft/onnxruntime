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
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<int32_t>(),
                               DataTypeImpl::GetTensorType<int64_t>()})
        .InputMemoryType(OrtMemTypeCPU, 1),
    CumSum);

ONNX_OPERATOR_KERNEL_EX(
    CumSum,
    kOnnxDomain,
    14,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<int32_t>(),
                               DataTypeImpl::GetTensorType<int64_t>()})
        .InputMemoryType(OrtMemTypeCPU, 1),
    CumSum);

Status CumSumProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& input = shader.AddInput("input", ShaderUsage::UseUniform);
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "var input_indices = " << input.OffsetToIndices("global_idx") << ";\n"
                            << "var sum : output_value_t = 0;\n"
                            << "var first : i32 = 0;\n"
                            << "if (uniforms.reverse == 1) {\n"
                            << "  first = i32(" + input.IndicesGet("input_indices", "uniforms.axis") + ");\n"
                            << "  if (uniforms.exclusive == 1) { first += 1; }\n"
                            << "}\n\n"
                            << "var last : i32 = 0;\n"
                            << "if (uniforms.reverse == 1) {\n"
                            << "  last = i32(" << GetElementAt("uniforms.input_shape", "uniforms.axis", input.Rank()) << ");\n"
                            << "} else {\n"
                            << "  last = i32(" + input.IndicesGet("input_indices", "uniforms.axis") + ");\n"
                            << "  if (uniforms.exclusive == 0) { last += 1; }\n"
                            << "}\n\n"
                            << "for (var i : i32 = first; i < last; i++) {\n"
                            << "  " << input.IndicesSet("input_indices", "uniforms.axis", "u32(i)") << ";\n"
                            << "  sum = sum + " << input.GetByIndices("input_indices") << ";\n"
                            << "}\n"
                            << output.SetByOffset("global_idx", "sum");

  return Status::OK();
}

Status CumSum::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  const TensorShape& input_shape = input_tensor->Shape();
  int64_t input_rank = input_shape.NumDimensions();

  const auto* axis_tensor = context.Input(1);
  const auto* axis_data = axis_tensor->Data<int>();
  int64_t axis = static_cast<int64_t>(axis_data[0]);

  ORT_ENFORCE(-input_rank <= axis && axis < input_rank, "Axes attribute must be within range -input_rank <= axis < input_rank.");
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