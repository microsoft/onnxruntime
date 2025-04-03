// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "contrib_ops/webgpu/quantization/gather_block_quantized.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

Status GatherBlockQuantizedProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& indices = shader.AddInput("indices", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& scales = shader.AddInput("scales", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride | ShaderUsage::UseValueTypeAlias);

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
      << "let output_indices = " << output.OffsetToIndices("global_idx") << ";\n"
      << "let indices_indices = " << indices.OffsetToIndices("global_idx") << ";\n";

  if (indices_rank_ > 1) {
    shader.MainFunctionBody()
        << "for (var i: u32 = 0; i < " << indices_rank_ << "; i++) {\n"
        << "  let index = " << output.IndicesGet("output_indices", "uniforms.gather_axis + i") << ";\n"
        << "  " << indices.IndicesSet("indices_indices", "i", "index") << ";\n};\n";
  } else {
    shader.MainFunctionBody()
        << "indices_indices = " << output.IndicesGet("output_indices", "uniforms.gather_axis") << ";\n";
  }

  return Status::OK();
}

Status GatherBlockQuantized::ComputeInternal(ComputeContext& context) const {
  const auto* x = context.Input(0);
  const auto* indices = context.Input(1);
  const auto* scales = context.Input(2);
  const auto* zero_points = context.Input(2);

  const auto x_shape = x->Shape();
  int64_t x_size = x_shape.Size();
  int64_t x_rank = x_shape.NumDimensions();
  int64_t x_dtype = x->GetElementType();
  auto* output_tensor = context.Output(0, x_shape);

  int indices_rank = indices->Shape().NumDimensions();
  bool is_signed = x_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || x_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4;
  int64_t gather_axis = (gather_axis_ >= 0) ? gather_axis_ : gather_axis_ + x_rank;
  int64_t quantize_axis = (quantize_axis_ >= 0) ? quantize_axis_ : quantize_axis_ + x_rank;

  GatherBlockQuantizedProgram program{is_signed, indices_rank, zero_points != nullptr};

  program
      .AddInputs({{x, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddInputs({{indices, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddInputs({{scales, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((x_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(x_size)}})
      .AddUniformVariables({{static_cast<uint32_t>(quantize_axis)}})
      .AddUniformVariables({{static_cast<uint32_t>(gather_axis)}})
      .AddUniformVariables({{static_cast<uint32_t>(block_size_)}})
      .CacheHint(std::to_string(is_signed), std::to_string(gather_axis), std::to_string(quantize_axis), std::to_string(block_size_));

  if (zero_points != nullptr) {
    program.AddInputs({{zero_points, ProgramTensorMetadataDependency::TypeAndRank}});
  }

  return context.RunProgram(program);
}

namespace {
const std::vector<MLDataType>& GatherBlockQuantizedT1Constraint() {
  static std::vector<MLDataType> types{
      DataTypeImpl::GetTensorType<Int4x2>(),
      DataTypeImpl::GetTensorType<UInt4x2>(),
      DataTypeImpl::GetTensorType<int8_t>(),
      DataTypeImpl::GetTensorType<uint8_t>()};
  return types;
}
const std::vector<MLDataType>& GatherBlockQuantizedTindConstraint() {
  static std::vector<MLDataType> types{
      DataTypeImpl::GetTensorType<int32_t>(),
      DataTypeImpl::GetTensorType<int64_t>()};
  return types;
}
}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    GatherBlockQuantized,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", GatherBlockQuantizedT1Constraint())
        .TypeConstraint("T2", WebGpuSupportedFloatTypes())
        .TypeConstraint("Tind", GatherBlockQuantizedTindConstraint()),
    GatherBlockQuantized);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
