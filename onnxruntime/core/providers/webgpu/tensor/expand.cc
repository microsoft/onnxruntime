// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"

#include "core/providers/webgpu/tensor/expand.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

Status ExpandProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderVariable::UseUniform);
  const auto& output = shader.AddOutput("output", ShaderVariable::UseUniform);

  shader.MainFunctionBody(shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.data_size"),
                          "let output_indices = ", output.OffsetToIndices("global_idx"), ";\n",
                          "let input_offset = ", input.BroadcastedIndicesToOffset("output_indices", output), ";\n",
                          output.SetByOffset("global_idx", input.GetByOffset("input_offset")));

  return Status::OK();
}

Status Expand::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  const auto* input_shape_tensor = context.Input(1);

  auto output_dims = input_shape_tensor->DataAsSpan<int64_t>();
  TensorShape output_shape{};
  ORT_RETURN_IF_ERROR(ComputeBroadcastOutputShape(Node().Name(), input_tensor->Shape(), output_dims, output_shape));

  auto* output_tensor = context.Output(0, output_shape);
  uint32_t data_size = SafeInt<uint32_t>(output_shape.Size());
  ExpandProgram program{};
  program
      .Inputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank}})
      .Outputs({{output_tensor, ProgramTensorMetadataDependency::Rank}})
      .DispatchGroupSize((data_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .UniformVariables({
          {data_size},
      });
  return context.RunProgram(program);
}

#define WEBGPU_EXPAND_KERNEL(OP_TYPE, VERSION, KERNEL_CLASS, TYPE)                    \
  ONNX_OPERATOR_KERNEL_EX(                                                            \
      OP_TYPE, kOnnxDomain, VERSION, kWebGpuExecutionProvider,                        \
      KernelDefBuilder().TypeConstraint("T", TYPE).InputMemoryType(OrtMemTypeCPU, 1), \
      KERNEL_CLASS);

#define WEBGPU_EXPAND_VERSIONED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, KERNEL_CLASS, TYPE) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                          \
      OP_TYPE, kOnnxDomain, VERSION_FROM, VERSION_TO, kWebGpuExecutionProvider,               \
      KernelDefBuilder().TypeConstraint("T", TYPE).InputMemoryType(OrtMemTypeCPU, 1),         \
      KERNEL_CLASS);

WEBGPU_EXPAND_VERSIONED_KERNEL(Expand, 8, 12, Expand, WebGpuSupportedFloatTypes())
WEBGPU_EXPAND_KERNEL(Expand, 13, Expand, WebGpuSupportedFloatTypes())

}  // namespace webgpu
}  // namespace onnxruntime
