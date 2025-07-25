// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/tensor/gather_elements.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    GatherElements,
    kOnnxDomain,
    11, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()),
    GatherElements);

ONNX_OPERATOR_KERNEL_EX(
    GatherElements,
    kOnnxDomain,
    13,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()),
    GatherElements);

Status GatherElementsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& input = shader.AddInput("input", ShaderUsage::UseUniform);
  const ShaderVariableHelper& indices = shader.AddInput("indices", ShaderUsage::UseUniform);
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform);

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "let output_indices = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "var idx = " << indices.GetByOffset("global_idx") << ";\n"
                            << "if (idx < 0) {\n"
                            << "  idx = idx + uniforms.axis_dim_limit;\n"
                            << "}\n"
                            << "var input_indices = output_indices;\n"
                            << input.IndicesSet("input_indices", "uniforms.axis", "u32(idx)") << ";\n"
                            << "let value = " << input.GetByIndices("input_indices") << ";\n"
                            << output.SetByOffset("global_idx", "value") << ";\n";

  return Status::OK();
}

Status GatherElements::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  const TensorShape& input_shape = input_tensor->Shape();
  int64_t input_rank = input_shape.NumDimensions();

  const auto* indices_tensor = context.Input(1);
  const TensorShape& indices_shape = indices_tensor->Shape();

  // Handle negative axis
  int64_t axis = axis_;
  if (axis < 0) {
    axis += input_rank;
  }

  auto axis_dim_limit = input_shape[static_cast<size_t>(axis)];

  auto output_dims = indices_shape.AsShapeVector();
  TensorShape output_shape(output_dims);
  auto* output_tensor = context.Output(0, output_shape);
  int64_t output_size = output_tensor->Shape().Size();

  if (output_size == 0) {
    return Status::OK();
  }

  GatherElementsProgram program{};
  program
      .AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddInputs({{indices_tensor, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutputs({output_tensor})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(output_size)},
                            {static_cast<int32_t>(axis_dim_limit)},
                            {static_cast<int32_t>(axis)}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
