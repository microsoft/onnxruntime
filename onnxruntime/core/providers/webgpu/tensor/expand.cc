// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"

#include "core/providers/webgpu/tensor/expand.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

Status ExpandProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input",
                                      ToProgramVariableDataType(Inputs()[0].tensor->GetElementType()),
                                      ShaderVariable::UseUniform);
  const auto& output = shader.AddOutput("output",
                                        ToProgramVariableDataType(Outputs()[0].tensor->GetElementType()),
                                        ShaderVariable::UseUniform);

  shader.MainFunctionBody(shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size"),
                          "let output_indices = ", output.OffsetToIndices("global_idx"), ";\n",
                          "let input_offset = ", input.BroadcastedIndicesToOffset("output_indices", output), ";\n",
                          output.SetByOffset("global_idx", input.GetByOffset("input_offset")));

  return Status::OK();
}

Status Expand::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  const auto* input_shape_tensor = context.Input(1);

  const auto* p_shape = input_shape_tensor->Data<int64_t>();
  TensorShapeVector output_dims{p_shape, p_shape + input_shape_tensor->Shape().Size()};
  TensorShape output_shape(output_dims);
  ORT_RETURN_IF_ERROR(ComputeBroadcastOutputShape(Node().Name(), input_tensor->Shape(), output_dims, output_shape));

  auto* output_tensor = context.Output(0, output_shape);
  SafeInt<uint32_t> vec_size = output_shape.Size();
  ExpandProgram program{"Expand"};
  program
      .Inputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank}})
      .Outputs({{output_tensor, ProgramTensorMetadataDependency::Rank}})
      .DispatchGroupSize((vec_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .UniformVariables({
          {static_cast<uint32_t>(vec_size)},
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
};  // namespace onnxruntime
