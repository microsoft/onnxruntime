// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "core/providers/webgpu/tensor/gather_nd.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

Status GatherNDProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& data = shader.AddInput("data", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& indices = shader.AddInput("input_indices", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.data_size")
                            << "  let output_indices = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "  var data_indices: data_indices_t;\n"
                            << "  var indices_indices: input_indices_indices_t;\n";

  uint32_t data_dim = 0;
  for (uint32_t i = data_dim; i < batch_dims_; i++) {
    shader.MainFunctionBody() << "  " << data.IndicesSet("data_indices", i, output.IndicesGet("output_indices", i)) << "\n"
                              << "  " << indices.IndicesSet("indices_indices", i, output.IndicesGet("output_indices", i)) << "\n";
  }
  data_dim += batch_dims_;

  for (uint32_t i = data_dim; i < static_cast<uint32_t>(indices.Rank() - 1); i++) {
    shader.MainFunctionBody() << "  " << indices.IndicesSet("indices_indices", i, output.IndicesGet("output_indices", i)) << "\n";
  }

  shader.MainFunctionBody() << "  var indice_value = i32(0);\n";
  for (uint32_t i = 0; i < indices_innerest_dim_; i++) {
    shader.MainFunctionBody() << "  " << indices.IndicesSet("indices_indices", indices.Rank() - 1, std::to_string(i)) << "\n"
                              << "  indice_value = " << indices.GetByIndices("indices_indices") << ";\n"
                              << "  if (indice_value < 0) {\n"
                              << "    indice_value += i32(" << data.IndicesGet("uniforms.data_shape", data_dim + i) << ");\n"
                              << "  }\n"
                              << "  " << data.IndicesSet("data_indices", data_dim + i, "u32(indice_value)") << "\n";
  }
  data_dim += indices_innerest_dim_;

  for (uint32_t i = 0; i < static_cast<uint32_t>(data.Rank() - data_dim); i++) {
    shader.MainFunctionBody() << "  " << data.IndicesSet("data_indices", data_dim, output.IndicesGet("output_indices", indices.Rank() - 1 + i)) << "\n";
  }

  shader.MainFunctionBody() << "  " << output.SetByOffset("global_idx", data.GetByIndices("data_indices"));

  return Status::OK();
}

Status CheckBatchDimensionsMatch(size_t num_batch_dimensions, const TensorShape& input_shape,
                                 const TensorShape& indices_shape) {
  ORT_RETURN_IF_NOT(
      num_batch_dimensions <= input_shape.NumDimensions() && num_batch_dimensions <= indices_shape.NumDimensions(),
      "Number of batch dimensions exceeds tensor rank. ", "Batch dimension count: ", num_batch_dimensions,
      ", input tensor rank: ", input_shape.NumDimensions(), ", indices tensor rank: ", indices_shape.NumDimensions());

  for (size_t batch_dimension_idx = 0; batch_dimension_idx < num_batch_dimensions; ++batch_dimension_idx) {
    ORT_RETURN_IF_NOT(
        input_shape[batch_dimension_idx] == indices_shape[batch_dimension_idx],
        "Batch dimensions differ at index ", batch_dimension_idx, ": ",
        input_shape[batch_dimension_idx], " != ", indices_shape[batch_dimension_idx]);
  }

  return Status::OK();
}

Status GatherND::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  const TensorShape& input_shape = input_tensor->Shape();
  const auto* indices_tensor = context.Input(1);
  const TensorShape& indices_shape = indices_tensor->Shape();

  if (indices_shape.NumDimensions() == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "indices tensor must has rank larger than 0");
  }

  auto indices_innerest_dim = indices_shape[indices_shape.NumDimensions() - 1];
  auto last_indices_dimension = batch_dims_ + indices_innerest_dim;
  if (last_indices_dimension > static_cast<int64_t>(input_shape.NumDimensions())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "last dimension of indices must not be larger than rank of input tensor");
  }

  ORT_RETURN_IF_ERROR(CheckBatchDimensionsMatch(static_cast<size_t>(batch_dims_),
                                                input_shape, indices_shape));

  // Output shape
  std::vector<int64_t> shape(indices_shape.GetDims().begin(), indices_shape.GetDims().end() - 1);
  shape.insert(shape.end(), input_shape.GetDims().begin() + static_cast<size_t>(last_indices_dimension), input_shape.GetDims().end());
  auto output_tensor = context.Output(0, TensorShape(shape));
  uint32_t data_size = onnxruntime::narrow<uint32_t>(output_tensor->Shape().Size());
  if (data_size == 0) {
    return Status::OK();
  }

  GatherNDProgram program{static_cast<uint32_t>(batch_dims_), static_cast<uint32_t>(indices_innerest_dim)};
  program
      .AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank},
                  {indices_tensor, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::Rank})
      .SetDispatchGroupSize((data_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .CacheHint(std::to_string(batch_dims_), std::to_string(indices_innerest_dim))
      .AddUniformVariables({{data_size}});
  return context.RunProgram(program);
}

#define WEBGPU_GATHERND_KERNEL(OP_TYPE, VERSION, KERNEL_CLASS, TYPE)                                                  \
  ONNX_OPERATOR_KERNEL_EX(                                                                                            \
      OP_TYPE, kOnnxDomain, VERSION, kWebGpuExecutionProvider,                                                        \
      KernelDefBuilder().TypeConstraint("T", TYPE).TypeConstraint("indices", DataTypeImpl::GetTensorType<int64_t>()), \
      KERNEL_CLASS);

#define WEBGPU_GATHERND_VERSIONED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, KERNEL_CLASS, TYPE)                       \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                                                  \
      OP_TYPE, kOnnxDomain, VERSION_FROM, VERSION_TO, kWebGpuExecutionProvider,                                       \
      KernelDefBuilder().TypeConstraint("T", TYPE).TypeConstraint("indices", DataTypeImpl::GetTensorType<int64_t>()), \
      KERNEL_CLASS);

WEBGPU_GATHERND_VERSIONED_KERNEL(GatherND, 11, 11, GatherND, WebGpuSupportedNumberTypes())
WEBGPU_GATHERND_VERSIONED_KERNEL(GatherND, 12, 12, GatherND, WebGpuSupportedNumberTypes())
WEBGPU_GATHERND_KERNEL(GatherND, 13, GatherND, WebGpuSupportedNumberTypes())

}  // namespace webgpu
}  // namespace onnxruntime
