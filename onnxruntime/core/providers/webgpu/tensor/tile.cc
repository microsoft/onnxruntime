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

  auto output_dims = input_shape.AsShapeVector();
  // Bound the total tiled byte count and detect overflow with division-based
  // checks so we return INVALID_ARGUMENT instead of throwing a SafeInt
  // overflow exception. Mirrors the CPU Tile implementation.
  constexpr int64_t kMaxTileOutputBytes = int64_t{4} * 1024 * 1024 * 1024;  // 4 GiB
  const int64_t element_size = static_cast<int64_t>(input_tensor->DataType()->Size());
  if (element_size <= 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Invalid element size for Tile input tensor type.");
  }
  const int64_t max_elements = kMaxTileOutputBytes / element_size;
  int64_t total_elements = 1;
  for (size_t axis = 0; axis < input_rank; axis++) {
    if (repeats_data[axis] < 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Tile repeat value must be non-negative, got: ", repeats_data[axis]);
    }
    const int64_t input_dim = output_dims[axis];
    const int64_t r = repeats_data[axis];
    int64_t dim;
    if (input_dim == 0 || r == 0) {
      dim = 0;
    } else if (input_dim > max_elements / r) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Tile output tensor would require more than ",
                             kMaxTileOutputBytes,
                             " bytes, which exceeds the supported maximum of ",
                             kMaxTileOutputBytes, " bytes.");
    } else {
      dim = input_dim * r;
    }
    output_dims[axis] = dim;
    if (dim > 0 && total_elements > max_elements / dim) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Tile output tensor would require more than ",
                             kMaxTileOutputBytes,
                             " bytes, which exceeds the supported maximum of ",
                             kMaxTileOutputBytes, " bytes.");
    }
    total_elements *= dim;
  }
  for (size_t i = 0; i < static_cast<size_t>(repeats_tensor->Shape().Size()); i++) {
    repeats.push_back(static_cast<uint32_t>(repeats_data[i]));
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
