// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"

#include "core/providers/webgpu/tensor/expand.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

Status ExpandProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.data_size");
  if (input.NumComponents() != output.NumComponents()) {
    const auto& output_indices = shader.AddIndices("output_indices");
    shader.MainFunctionBody() << "  let output_indices = " << output_indices.OffsetToIndices("global_idx * 4") << ";\n"
                              << "  let input_offset = " << input.BroadcastedIndicesToOffset("output_indices", output_indices) << ";\n  "
                              << "  let value = vec4<input_value_t>(" << input.GetByOffset("input_offset") << ");\n"
                              << output.SetByOffset("global_idx", "value");
  } else {
    shader.MainFunctionBody() << "  let output_indices = " << output.OffsetToIndices("global_idx") << ";\n"
                              << "  let input_offset = " << input.BroadcastedIndicesToOffset("output_indices", output) << ";\n  "
                              << output.SetByOffset("global_idx", input.GetByOffset("input_offset"));
  }
  return Status::OK();
}

Status Expand::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  const auto* input_shape_tensor = context.Input(1);

  auto output_dims = input_shape_tensor->DataAsSpan<int64_t>();
  TensorShape output_shape{};
  TensorShape input_shape = input_tensor->Shape();
  ORT_RETURN_IF_ERROR(ComputeBroadcastOutputShape(Node().Name(), input_shape, output_dims, output_shape));

  auto* output_tensor = context.Output(0, output_shape);
  const int components_i = input_shape.IsScalar() ? 1 : input_shape[input_shape.NumDimensions() - 1] % 4 == 0 ? 4
                                                                                                              : 1;
  const int components_o = output_shape.IsScalar() ? 1 : output_shape[output_shape.NumDimensions() - 1] % 4 == 0 ? 4
                                                                                                                 : 1;
  uint32_t data_size = gsl::narrow<uint32_t>(output_shape.Size() / components_o);

  ExpandProgram program{};
  program
      .AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank, components_i}})
      .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::TypeAndRank, components_o}})
      .SetDispatchGroupSize((data_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({
          {data_size},
      });
  if (components_i != components_o) {
    program.AddIndices(output_shape);
  }
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

WEBGPU_EXPAND_VERSIONED_KERNEL(Expand, 8, 12, Expand, WebGpuSupportedNumberTypes())
WEBGPU_EXPAND_KERNEL(Expand, 13, Expand, WebGpuSupportedNumberTypes())

}  // namespace webgpu
}  // namespace onnxruntime
