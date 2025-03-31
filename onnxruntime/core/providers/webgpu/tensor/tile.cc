// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/tensor/tile.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Tile,
    kOnnxDomain,
    6, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()).InputMemoryType(OrtMemTypeCPU, 1),
    Tile);

ONNX_OPERATOR_KERNEL_EX(
    Tile,
    kOnnxDomain,
    13,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()).InputMemoryType(OrtMemTypeCPU, 1),
    Tile);

Status TileProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "let output_indices = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "var input_indices: input_indices_t;\n";
  for (auto i = 0; i < input.Rank(); i++) {
    std::string input_dim_i = absl::StrCat("input_dim_", i);
    std::string input_dim_value = absl::StrCat("input_dim_", i, "_value");
    shader.MainFunctionBody() << "let " << input_dim_i << " = " << input.IndicesGet("uniforms.input_shape", i) << ";\n"
                              << "let " << input_dim_value << " = " << output.IndicesGet("output_indices", i) << " % " << input_dim_i << ";\n"
                              << input.IndicesSet("input_indices", i, input_dim_value) << ";\n";
  }

  shader.MainFunctionBody() << output.SetByOffset("global_idx", input.GetByIndices("input_indices"));

  return Status::OK();
}

Status Tile::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  const TensorShape& input_shape = input_tensor->Shape();
  size_t input_rank = input_shape.NumDimensions();

  const auto* repeats_tensor = context.Input(1);
  const auto* repeats_data = repeats_tensor->Data<int64_t>();
  std::vector<uint32_t> repeats;

  for (size_t i = 0; i < static_cast<uint32_t>(repeats_tensor->Shape().Size()); i++) {
    repeats.push_back(static_cast<uint32_t>(repeats_data[i]));
  }

  auto output_dims = input_shape.AsShapeVector();
  for (size_t axis = 0; axis < input_rank; axis++) {
    output_dims[axis] *= repeats[axis];
  }

  TensorShape output_shape(output_dims);
  auto* output_tensor = context.Output(0, output_shape);
  int64_t output_size = output_tensor->Shape().Size();

  if (output_size == 0) {
    return Status::OK();
  }

  TileProgram program{};
  program
      .AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutputs({output_tensor})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(output_size)},
                            {repeats}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime