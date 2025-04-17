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
      << "let output_indices = " << output.OffsetToIndices("global_idx") << ";\n";

  if (indices_rank_ > 1) {
    shader.MainFunctionBody()
        << "let indices_indices = input_indices_t(0);\n"
        << "for (var i: u32 = 0; i < " << indices_rank_ << "; i++) {\n"
        << "  let index = " << output.IndicesGet("output_indices", "uniforms.gather_axis + i") << ";\n"
        << "  " << indices.IndicesSet("indices_indices", "i", "index") << ";\n};\n";
  } else {
    shader.MainFunctionBody()
        << "let indices_indices = " << output.IndicesGet("output_indices", "uniforms.gather_axis") << ";\n";
  }
  shader.MainFunctionBody()
      << "var data_indices = input_indices_t(0);\n"
      << "for (var i: u32 = 0; i < uniforms.gather_axis; i++) {\n"
      << "  let index = " << output.IndicesGet("output_indices", "i") << ";\n"
      << x.IndicesSet("data_indices", "i", "index") << ";\n};\n";

  shader.MainFunctionBody()
      << "var index_from_indices = " << indices.GetByIndices("indices_indices") << ";\n"
      << "if (index_from_indices < 0) {\n"
      << "  index_from_indices += " << x_shape_[gather_axis_] << ";}\n"
      << x.IndicesSet("data_indices", "uniforms.gather_axis", "u32(index_from_indices)") << ";\n"
      << "for (var i = uniforms.gather_axis + 1; i < " << output_shape_.NumDimensions() << "; i++) {\n"
      << "  let index = " << output.IndicesGet("output_indices", "i + " + std::to_string(indices_rank_ - 1)) << ";\n"
      << x.IndicesSet("data_indices", "i", "index") << ";\n};\n";

  const std::string unpack = (is_signed_) ? "unpack4xI8" : "unpack4xU8";

  shader.MainFunctionBody()
      << "let data_offset = " << x.IndicesToOffset("data_indices") << ";\n"
      << "let data_index = data_offset % 8;\n"
      << "let packed_4bit_quantized_data = " << x.GetByOffset("data_offset / 8") << ";\n"
      << "let packed_8bit_quantized_data = (packed_4bit_quantized_data >> (4 * (data_index % 2))) & 0x0f0f0f0f;\n"
      << "let quantized_data_vec = " << unpack << "(u32(packed_8bit_quantized_data));\n"
      << "var quantized_data = quantized_data_vec[data_index / 2];\n";
  if (is_signed_) {
    shader.MainFunctionBody()
        << "if((quantized_data & 0x8) != 0) { quantized_data = quantized_data - 16 ;};\n";
  }
  shader.MainFunctionBody()
      << "var scale_indices = data_indices;\n"
      << "let quantize_axis_index = " << scales.IndicesGet("data_indices", "uniforms.quantize_axis") << "/ uniforms.block_size;\n"
      << scales.IndicesSet("scale_indices", "uniforms.quantize_axis", "quantize_axis_index") << ";\n"
      << "var scale = " << scales.GetByIndices("scale_indices") << ";\n";

  if (!has_zeropoint_) {
    shader.MainFunctionBody()
        << "let zero_point = 0;\n";
  } else {
    const auto& zero_point = shader.AddInput("zero_point", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
    shader.MainFunctionBody()
        << "let zero_point_indices = scale_indices;\n"
        << "let zero_point_offset = " << zero_point.IndicesToOffset("zero_point_indices") << ";\n"
        << "let zero_point_index = zero_point_offset % 8;\n"
        << "let packed_4bit_zero_points = " << zero_point.GetByOffset("zero_point_offset / 8") << ";\n"
        << "let packed_8bit_zero_points = (packed_4bit_zero_points >> (4 * (zero_point_index % 2))) & 0x0f0f0f0f;\n"
        << "let zero_point_vec = " << unpack << "(u32(packed_8bit_zero_points));\n"
        << "var zero_point = zero_point_vec[zero_point_index / 2];\n";
    if (is_signed_) {
      shader.MainFunctionBody()
          << "if((zero_point & 0x8) != 0) { zero_point = zero_point - 16 ;};\n";
    }
  }
  shader.MainFunctionBody()
      << "let dequantized_data = output_value_t(quantized_data - zero_point) * scale;\n"
      << output.SetByOffset("global_idx", "dequantized_data") << ";\n";

  return Status::OK();
}

TensorShapeVector splice(TensorShapeVector vec, size_t start, size_t deleteCount, const TensorShapeVector toInsert = {}) {
  TensorShapeVector new_vec;

  for (size_t i = 0; i < vec.size(); i++) {
    if (i < start) {
      new_vec.push_back(vec[i]);
    } else if (i == start) {
      new_vec.insert(new_vec.end(), toInsert.begin(), toInsert.end());
    } else if (i >= start + deleteCount) {
      new_vec.push_back(vec[i]);
    }
  }
  return new_vec;
}

Status GatherBlockQuantized::ComputeInternal(ComputeContext& context) const {
  const auto* x = context.Input(0);
  const auto* indices = context.Input(1);
  const auto* scales = context.Input(2);
  const auto* zero_points = context.Input(3);

  const auto x_shape = x->Shape();
  int64_t x_size = x_shape.Size();
  int64_t x_rank = x_shape.NumDimensions();
  int64_t x_dtype = x->GetElementType();

  size_t indices_rank = indices->Shape().NumDimensions();
  bool is_signed = x_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || x_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4;
  int64_t gather_axis = (gather_axis_ >= 0) ? gather_axis_ : gather_axis_ + x_rank;
  int64_t quantize_axis = (quantize_axis_ >= 0) ? quantize_axis_ : quantize_axis_ + x_rank;

  TensorShape output_shape = splice(x_shape.AsShapeVector(), gather_axis, 1, indices->Shape().AsShapeVector());
  size_t output_size = output_shape.Size();
  auto* output_tensor = context.Output(0, output_shape);

  GatherBlockQuantizedProgram program{is_signed, indices_rank, gather_axis, zero_points != nullptr, x_shape, output_shape};

  program
      .AddInputs({{x, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddInputs({{indices, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddInputs({{scales, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(x_size)}})
      .AddUniformVariables({{static_cast<uint32_t>(quantize_axis)}})
      .AddUniformVariables({{static_cast<uint32_t>(gather_axis)}})
      .AddUniformVariables({{static_cast<uint32_t>(block_size_)}})
      .CacheHint(std::to_string(gather_axis), std::to_string(quantize_axis), std::to_string(block_size_));

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
