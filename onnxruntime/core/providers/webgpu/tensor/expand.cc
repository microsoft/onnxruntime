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
  if (Inputs()[0].var_type == ProgramVariableDataType::Boolx4) {
    const auto& input_indices = shader.AddIndices("input_indices");
    const auto& output_indices = shader.AddIndices("output_indices");
    if (input_last_dim_divisible_by_4_) {
      // The last dims of input shape and output shape are all divisible by 4.
      shader.MainFunctionBody() << "  let output_indices = " << output_indices.OffsetToIndices("global_idx * 4") << ";\n"
                                << "  let input_offset = " << input_indices.BroadcastedIndicesToOffset("output_indices", output_indices) << ";\n  "
                                << output.SetByOffset("global_idx", input.GetByOffset("input_offset"));
    } else if (output_last_dim_divisible_by_4_) {
      // The last dim of output shape is divisible by 4, and the last dim of input shape is 1.
      shader.MainFunctionBody() << "  let output_indices = " << output_indices.OffsetToIndices("global_idx * 4") << ";\n"
                                << "  let input_offset = " << input_indices.BroadcastedIndicesToOffset("output_indices", output_indices) << ";\n  "
                                << "  let value = vec4<bool>(" << input.GetByOffset("input_offset / 4") << "[input_offset % 4]);\n"
                                << "  " << output.SetByOffset("global_idx", "value");
    } else {
      shader.MainFunctionBody() << "  var output_indices = " << output_indices.OffsetToIndices("global_idx * 4") << ";\n"
                                << "  let input_offset_0 = " << input_indices.BroadcastedIndicesToOffset("output_indices", output_indices) << ";\n"
                                << "  output_indices = " << output_indices.OffsetToIndices("global_idx * 4 + 1") << ";\n"
                                << "  let input_offset_1 = " << input_indices.BroadcastedIndicesToOffset("output_indices", output_indices) << ";\n"
                                << "  output_indices = " << output_indices.OffsetToIndices("global_idx * 4 + 2") << ";\n"
                                << "  let input_offset_2 = " << input_indices.BroadcastedIndicesToOffset("output_indices", output_indices) << ";\n"
                                << "  output_indices = " << output_indices.OffsetToIndices("global_idx * 4 + 3") << ";\n"
                                << "  let input_offset_3 = " << input_indices.BroadcastedIndicesToOffset("output_indices", output_indices) << ";\n"
                                << "  let value = vec4<bool>("
                                << input.GetByOffset("input_offset_0 / 4") << "[input_offset_0 % 4], "
                                << input.GetByOffset("input_offset_1 / 4") << "[input_offset_1 % 4], "
                                << input.GetByOffset("input_offset_2 / 4") << "[input_offset_2 % 4], "
                                << input.GetByOffset("input_offset_3 / 4") << "[input_offset_3 % 4]);\n"
                                << output.SetByOffset("global_idx", "value");
    }
    return Status::OK();
  }

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
  // Check if either input is boolean
  // For boolean inputs, we need to handle them differently in the shader. This is because `bool` is not a valid type in
  // storage buffer. We have to use a `u32` to represent 4 boolean values.
  bool is_bool = input_tensor->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  bool input_last_dim_divisible_by_4 = (!input_shape.IsScalar()) && (input_shape[input_shape.NumDimensions() - 1] % 4 == 0);
  bool output_last_dim_divisible_by_4 = (!output_shape.IsScalar()) && (output_shape[output_shape.NumDimensions() - 1] % 4 == 0);
  const int components_i = (is_bool || input_last_dim_divisible_by_4) ? 4 : 1;
  const int components_o = (is_bool || output_last_dim_divisible_by_4) ? 4 : 1;
  uint32_t data_size = onnxruntime::narrow<uint32_t>((output_shape.Size() + components_o - 1) / components_o);
  if (data_size == 0) {
    return Status::OK();
  }
  ExpandProgram program{input_last_dim_divisible_by_4, output_last_dim_divisible_by_4};
  program.SetDispatchGroupSize((data_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({
          {data_size},
      });
  if (is_bool) {
    program.AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank, ProgramInput::Flatten, components_i}})
        .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::TypeAndRank, {data_size}, components_o}})
        .AddIndices(std::move(input_shape));
  } else {
    program.AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank, components_i}})
        .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::TypeAndRank, components_o}});
  }
  if (is_bool || components_i != components_o) {
    program.AddIndices(std::move(output_shape));
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

WEBGPU_EXPAND_VERSIONED_KERNEL(Expand, 8, 12, Expand, WebGpuSupportedNumberAndBoolTypes())
WEBGPU_EXPAND_KERNEL(Expand, 13, Expand, WebGpuSupportedNumberAndBoolTypes())

}  // namespace webgpu
}  // namespace onnxruntime
