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
  const auto& x = shader.AddInput("input", ShaderUsage::UseElementTypeAlias);
  const auto& x_shape = shader.AddIndices("input_shape", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& indices = shader.AddInput("indices", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseIndicesToOffset);
  const auto& scales = shader.AddInput("scales", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride | ShaderUsage::UseValueTypeAlias);

  bool is_4bit = bits_ == 4;
  const std::string unpack = (is_signed_) ? "unpack4xI8" : "unpack4xU8";

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
      << "let output_indices = " << output.OffsetToIndices("global_idx") << ";\n";

  if (indices_rank_ > 1) {
    shader.MainFunctionBody()
        << "var indices_indices = indices_indices_t(0);\n"
        << "for (var i: u32 = 0; i < " << indices_rank_ << "; i++) {\n"
        << "  let index = " << output.IndicesGet("output_indices", "uniforms.gather_axis + i") << ";\n"
        << "  " << indices.IndicesSet("indices_indices", "i", "index") << ";\n};\n";
  } else {
    shader.MainFunctionBody()
        << "let indices_indices = " << output.IndicesGet("output_indices", "uniforms.gather_axis") << ";\n";
  }
  shader.MainFunctionBody()
      << "var data_indices = input_shape_indices_t(0);\n"
      << "for (var i: u32 = 0; i < uniforms.gather_axis; i++) {\n"
      << "  let index = " << output.IndicesGet("output_indices", "i") << ";\n  "
      << x_shape.IndicesSet("data_indices", "i", "index") << ";\n};\n"
      << "var index_from_indices = " << indices.GetByIndices("indices_indices") << ";\n"
      << "if (index_from_indices < 0) { index_from_indices += " << x_shape_[gather_axis_] << ";}\n"
      << x_shape.IndicesSet("data_indices", "uniforms.gather_axis", "u32(index_from_indices)") << ";\n"
      << "for (var i = uniforms.gather_axis + 1; i < " << output_shape_.NumDimensions() << "; i++) {\n"
      << "  let index = " << output.IndicesGet("output_indices", "i + " + std::to_string(indices_rank_ - 1)) << ";\n  "
      << x_shape.IndicesSet("data_indices", "i", "index") << ";\n};\n"
      << "  let data_offset = " << x_shape.IndicesToOffset("data_indices") << ";\n";

  if (is_4bit) {
    shader.MainFunctionBody()
        << "  let data_index = data_offset % 8;\n"
        << "  let packed_4bit_quantized_data = " << x.GetByOffset("data_offset / 8") << ";\n"
        << "  let packed_8bit_quantized_data = (packed_4bit_quantized_data >> (4 * (data_index % 2))) & 0x0f0f0f0f;\n"
        << "  let quantized_data_vec = " << unpack << "(u32(packed_8bit_quantized_data));\n"
        << "  var quantized_data = quantized_data_vec[data_index / 2];\n";
  } else {
    shader.MainFunctionBody()
        << "  let data_index = data_offset % 4;\n"
        << "  let packed_8bit_quantized_data = " << x.GetByOffset("data_offset / 4") << ";\n"
        << "  let quantized_data_vec = " << unpack << "(u32(packed_8bit_quantized_data));\n"
        << "  var quantized_data = quantized_data_vec[data_index];\n";
  }

  if (is_signed_) {
    shader.MainFunctionBody()
        << "  if((quantized_data & 0x8) != 0) { quantized_data = quantized_data - 16 ;};\n";
  }
  shader.MainFunctionBody()
      << "  var scale_indices = data_indices;\n"
      << "  let quantize_axis_index = " << scales.IndicesGet("data_indices", "uniforms.quantize_axis") << "/ uniforms.block_size;\n  "
      << scales.IndicesSet("scale_indices", "uniforms.quantize_axis", "quantize_axis_index") << ";\n"
      << "  var scale = " << scales.GetByIndices("scale_indices") << ";\n";

  if (!has_zeropoint_) {
    const std::string default_zero_point = is_uint8_ ? is_4bit ? "input_element_t(8)" : "input_element_t(128)" : "input_element_t(0)";
    shader.MainFunctionBody()
        << "  let zero_point = " << default_zero_point << ";\n";
  } else {
    const auto& zero_point = shader.AddInput("zero_point", ShaderUsage::None);
    shader.MainFunctionBody()
        << "  let zero_point_indices = scale_indices;\n"
        << "  let zero_point_offset = " << scales.IndicesToOffset("zero_point_indices") << ";\n";
    if (is_4bit) {
      shader.MainFunctionBody()
        << "  let zero_point_index = zero_point_offset % 8;\n"
        << "  let packed_4bit_zero_points = " << zero_point.GetByOffset("zero_point_offset / 8") << ";\n"
        << "  let packed_8bit_zero_points = (packed_4bit_zero_points >> (4 * (zero_point_index % 2))) & 0x0f0f0f0f;\n"
        << "  let zero_point_vec = " << unpack << "(u32(packed_8bit_zero_points));\n"
        << "  var zero_point = zero_point_vec[zero_point_index / 2];\n";
    } else {
      shader.MainFunctionBody()
        << "  let zero_point_index = zero_point_offset % 4;\n"
        << "  let packed_8bit_zero_points = " << zero_point.GetByOffset("zero_point_offset / 4") << ";\n"
        << "  let zero_point_vec = " << unpack << "(u32(packed_8bit_zero_points));\n"
        << "  var zero_point = zero_point_vec[zero_point_index];\n";
    }
    if (is_signed_) {
      shader.MainFunctionBody()
          << "  if((zero_point & 0x8) != 0) { zero_point = zero_point - 16 ;};\n";
    }
  }
  shader.MainFunctionBody()
      << "  let dequantized_data = (output_value_t(quantized_data) - output_value_t(zero_point)) * scale;\n  "
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

  // auto x_shape = x->Shape();
  int64_t x_size = x->Shape().Size();
  int x_rank = static_cast<int>(x->Shape().NumDimensions());
  int64_t x_dtype = x->GetElementType();
  bool is_signed = x_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || x_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4;
  bool is_int8 = x_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || x_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;

  if (bits_ == 4 && is_int8) {
    std::optional<Tensor> data_representation_4bit;
    std::optional<Tensor> zero_points_representation_4bit;
    TensorShape data_representation_4bit_shape{x->Shape()};
    MLDataType new_dtype = (x_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) ?
      DataTypeImpl::GetType<UInt4x2>() : DataTypeImpl::GetType<Int4x2>();
    auto memory_info = OrtMemoryInfo{
            "WebGPU_Buffer",
            OrtDeviceAllocator,
            OrtDevice{OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE, 0}};

    data_representation_4bit_shape[x_rank - 1] = data_representation_4bit_shape[x_rank - 1] * 2;
    data_representation_4bit.emplace(
        new_dtype,
        data_representation_4bit_shape,
        const_cast<void*>(x->DataRaw()),
        memory_info);

    if (zero_points) {
      TensorShape zero_points_representation_4bit_shape{zero_points->Shape()};
      zero_points_representation_4bit_shape[zero_points->Shape().NumDimensions() - 1] =
          zero_points_representation_4bit_shape[zero_points->Shape().NumDimensions() - 1] * 2;
      zero_points_representation_4bit.emplace(
          new_dtype,
          zero_points_representation_4bit_shape,
          const_cast<void*>(zero_points->DataRaw()),
          memory_info);
    }
    x = data_representation_4bit.has_value() ? &data_representation_4bit.value() : x;
    zero_points = zero_points_representation_4bit.has_value() ? &zero_points_representation_4bit.value() : zero_points;
  }

  const auto& x_shape = x->Shape();

  size_t indices_rank = indices->Shape().NumDimensions();
  const auto scales_shape = scales->Shape();
  size_t scales_rank = scales_shape.NumDimensions();
  int gather_axis = (gather_axis_ >= 0) ? gather_axis_ : gather_axis_ + x_rank;
  int quantize_axis = (quantize_axis_ >= 0) ? quantize_axis_ : quantize_axis_ + x_rank;

  ORT_RETURN_IF_NOT(x_shape.NumDimensions() == scales_rank,
                    "data and scales must have the same rank.");
  for (size_t i = 0; i < x_shape.NumDimensions(); ++i) {
    ORT_RETURN_IF_NOT(i == static_cast<size_t>(quantize_axis)
                          ? (x_shape[i] * 1 + block_size_ - 1) / block_size_ == scales_shape[i]
                          : x_shape[i] == scales_shape[i],
                      "data and scales do not match shapes.");
  }

  TensorShape output_shape = splice(x_shape.AsShapeVector(), gather_axis, 1, indices->Shape().AsShapeVector());
  int64_t output_size = output_shape.Size();
  auto* output_tensor = context.Output(0, output_shape);

  GatherBlockQuantizedProgram program{is_signed, is_int8, indices_rank, gather_axis, bits_, zero_points != nullptr, x_shape, output_shape};

  program
      .AddInputs({{x, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, (bits_ == 4) ? 8 : 4}})
      .AddIndices(x_shape)
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
    ORT_RETURN_IF_NOT(scales_shape == zero_points->Shape(),
                      "scales and zero_points must have the same shape.");
    auto zero_points_shape = zero_points->Shape();
    program.AddInputs({{zero_points, ProgramTensorMetadataDependency::None, ProgramInput::Flatten, (bits_ == 4) ? 8 : 4}});
  }

  return context.RunProgram(program);
}

namespace {
const std::vector<MLDataType>& GatherBlockQuantizedT1Constraint() {
  static std::vector<MLDataType> types{
      DataTypeImpl::GetTensorType<Int4x2>(),
      DataTypeImpl::GetTensorType<UInt4x2>(),
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
