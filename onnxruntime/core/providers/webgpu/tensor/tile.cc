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
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedNumberTypes()),
    Tile);

ONNX_OPERATOR_KERNEL_EX(
    Tile,
    kOnnxDomain,
    13,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedNumberTypes()),
    Tile);

const std::string AppendTileFunction(ShaderVariableHelper& input, ShaderVariableHelper& output) {
  std::ostringstream ss;
  ss.imbue(std::locale::classic());
  const TensorShape& input_shape = input.GetShape();
  int32_t input_rank = gsl::narrow_cast<int32_t>(input_shape.NumDimensions());
  ss << "fn tile(i: output_indices_t)->input_indices_t {\n"
        "  var input_indices;\n";
  for (auto i = 0; i < input_rank; i++) {
    ss << "  input_dim_i = input.GetDimensionByIndex(" << i << ");\n";
    ss << "  input_dim_value = output.GetDimensionByIndex(" << i << ") % input_dim_i;\n";
    ss << "  input.indicesSet('input_indices', '" << i << "', 'input_dim_value');\n";
  }
  ss << "  return input_indices;\n"
        "}\n";

  return ss.str();
}

Status TileProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& input = shader.AddInput("input", ShaderUsage::UseUniform);
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform);
  shader.AppendImplementation(AppendTileFunction(input, output));
  shader.SetMainFunctionBody(shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size"),
                             "  let output_indices = ", output.OffsetToIndices("global_idx"),
                             ";\n"
                             "  let input_indices = tile(input, output); \n"
                             "  ",
                             output.SetByOffset("global_idx", input.GetByIndices("input_indices")));

  return Status::OK();
}

Status Tile::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  const TensorShape& input_shape = input_tensor->Shape();
  int32_t input_rank = gsl::narrow_cast<int32_t>(input_shape.NumDimensions());

  const auto* repeats_tensor = context.Input(1);
  const auto* repeats = repeats_tensor->Data<int32_t>();

  auto output_dims = input_shape.AsShapeVector();
  for (size_t axis = 0; axis < input_rank; axis++) {
    output_dims[axis] *= repeats[axis];
  }

  TensorShape output_shape(output_dims);
  auto* output_tensor = context.Output(0, output_shape);
  uint32_t output_size = gsl::narrow_cast<int32_t>(output_tensor->Shape().Size());

  TileProgram program{};
  program
      .AddInputs({{input_tensor}, {repeats_tensor}})
      .AddOutputs({output_tensor})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime