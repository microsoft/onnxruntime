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
  const auto& indices = shader.AddInput("indices", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseIndicesToOffset | ShaderUsage::UseValueTypeAlias);
  const auto& scales = shader.AddInput("scales", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride | ShaderUsage::UseValueTypeAlias);

  const bool is_2bit = bits_ == 2;
  const bool is_4bit = bits_ == 4;
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
      << "var index = " << indices.GetByIndices("indices_indices") << ";\n"
      << "let gather_axis_dim = indices_value_t(" << x_shape.IndicesGet("uniforms.input_shape_shape", gather_axis_) << ");\n"
      << "if (index < 0) { index += gather_axis_dim;}\n"
      << "if (index < 0 || index >= gather_axis_dim) {\n"
      << "  " << output.SetByOffset("global_idx", "output_value_t(0)") << ";\n"
      << "  return;\n"
      << "}\n"
      << "var data_indices = input_shape_indices_t(0);\n";

  for (int i = 0, j = 0; i < x_shape.Rank(); i++) {
    if (static_cast<int>(i) == gather_axis_) {
      shader.MainFunctionBody() << "  " << x_shape.IndicesSet("data_indices", i, "u32(index)") << ";\n";
      j += indices.Rank();
    } else {
      shader.MainFunctionBody() << "  " << x_shape.IndicesSet("data_indices", i, output.IndicesGet("output_indices", j)) << ";\n";
      j++;
    }
  }

  shader.MainFunctionBody()
      << "  let data_offset = " << x_shape.IndicesToOffset("data_indices") << ";\n";

  if (is_2bit) {
    // 2-bit values are packed 4 per byte (LSB first). x is the original uint8 tensor with
    // Flatten=4 (4 bytes per u32); the input_shape uniform here is the *dequantized* shape,
    // so data_offset is the dequantized 2-bit-element index.
    shader.MainFunctionBody()
        << "  let byte_idx_2b = data_offset / 4;\n"
        << "  let bit_shift_2b = (data_offset % 4) * 2;\n"
        << "  let packed_word_2b = " << x.GetByOffset("byte_idx_2b / 4") << ";\n"
        << "  let byte_in_word_2b = byte_idx_2b % 4;\n"
        << "  let unpacked_bytes_2b = " << unpack << "(u32(packed_word_2b));\n"
        << "  var quantized_data = (unpacked_bytes_2b[byte_in_word_2b] >> bit_shift_2b) & 0x3;\n";
    if (is_signed_) {
      shader.MainFunctionBody()
          << "  if((quantized_data & 0x2) != 0) { quantized_data = quantized_data - 4 ;};\n";
    }
  } else if (is_4bit) {
    shader.MainFunctionBody()
        << "  let data_index = data_offset % 8;\n"
        << "  let packed_4bit_quantized_data = " << x.GetByOffset("data_offset / 8") << ";\n"
        << "  let packed_8bit_quantized_data = (packed_4bit_quantized_data >> (4 * (data_index % 2))) & 0x0f0f0f0f;\n"
        << "  let quantized_data_vec = " << unpack << "(u32(packed_8bit_quantized_data));\n"
        << "  var quantized_data = quantized_data_vec[data_index / 2];\n";
    if (is_signed_) {
      shader.MainFunctionBody()
          << "  if((quantized_data & 0x8) != 0) { quantized_data = quantized_data - 16 ;};\n";
    }
  } else {
    shader.MainFunctionBody()
        << "  let data_index = data_offset % 4;\n"
        << "  let packed_8bit_quantized_data = " << x.GetByOffset("data_offset / 4") << ";\n"
        << "  let quantized_data_vec = " << unpack << "(u32(packed_8bit_quantized_data));\n"
        << "  var quantized_data = quantized_data_vec[data_index];\n";
  }

  shader.MainFunctionBody()
      << "  var scale_indices = data_indices;\n"
      << "  let quantize_axis_index = " << scales.IndicesGet("data_indices", "uniforms.quantize_axis") << "/ uniforms.block_size;\n  "
      << scales.IndicesSet("scale_indices", "uniforms.quantize_axis", "quantize_axis_index") << ";\n"
      << "  var scale = " << scales.GetByIndices("scale_indices") << ";\n";

  if (!has_zeropoint_) {
    std::string default_zero_point;
    if (is_uint8_) {
      if (is_2bit) {
        default_zero_point = "input_element_t(2)";
      } else if (is_4bit) {
        default_zero_point = "input_element_t(8)";
      } else {
        default_zero_point = "input_element_t(128)";
      }
    } else {
      default_zero_point = "input_element_t(0)";
    }
    shader.MainFunctionBody()
        << "  let zero_point = " << default_zero_point << ";\n";
  } else {
    const auto& zero_point = shader.AddInput("zero_point", ShaderUsage::None);
    shader.MainFunctionBody()
        << "  let zero_point_indices = scale_indices;\n"
        << "  let zero_point_offset = " << scales.IndicesToOffset("zero_point_indices") << ";\n";
    if (is_2bit) {
      // 2-bit zero points are packed 4-per-byte along the quantize axis only. The scales
      // tensor's flat offset cannot be used directly because dividing it by 4 crosses row
      // boundaries when scale_qaxis_dim is not a multiple of 4 (e.g. scales {2,3,1} has
      // packed zp shape {2,3,1} with one usable 2-bit value per byte per row). Derive the
      // packed byte index from the scale row index plus the within-row quantize-axis index.
      shader.MainFunctionBody()
          << "  let q_idx_2b = " << scales.IndicesGet("scale_indices", "uniforms.quantize_axis") << ";\n"
          << "  let scale_row_2b = zero_point_offset / uniforms.scale_qaxis_dim;\n"
          << "  let zp_byte_offset_2b = scale_row_2b * uniforms.zp_packed_qaxis_dim + q_idx_2b / 4u;\n"
          << "  let zp_bit_shift_2b = (q_idx_2b % 4u) * 2u;\n"
          << "  let packed_zp_word_2b = " << zero_point.GetByOffset("zp_byte_offset_2b / 4") << ";\n"
          << "  let zp_byte_in_word_2b = zp_byte_offset_2b % 4;\n"
          << "  let zp_unpacked_2b = " << unpack << "(u32(packed_zp_word_2b));\n"
          << "  var zero_point = (zp_unpacked_2b[zp_byte_in_word_2b] >> zp_bit_shift_2b) & 0x3;\n";
    } else if (is_4bit) {
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
      if (is_2bit) {
        shader.MainFunctionBody()
            << "  if((zero_point & 0x2) != 0) { zero_point = zero_point - 4 ;};\n";
      } else if (is_4bit) {
        shader.MainFunctionBody()
            << "  if((zero_point & 0x8) != 0) { zero_point = zero_point - 16 ;};\n";
      }
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

  int x_rank = static_cast<int>(x->Shape().NumDimensions());
  int64_t x_dtype = x->GetElementType();
  bool is_signed = x_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || x_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4;
  bool is_int8 = x_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || x_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  bool is_uint8 = x_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;

  // Only uint8 storage supports the full bits set {2, 4, 8}. The packed int4/uint4 types
  // can only carry bits==4, matching the CPU kernel's constraint.
  if (is_uint8) {
    ORT_RETURN_IF_NOT(bits_ == 2 || bits_ == 4 || bits_ == 8,
                      "'bits' must be 2, 4 or 8 for uint8 input.");
  } else {
    ORT_RETURN_IF_NOT(bits_ == 4, "'bits' must be 4 for non-uint8 input.");
  }

  std::optional<Tensor> data_representation_4bit;
  std::optional<Tensor> zero_points_representation_4bit;
  if (bits_ == 4 && is_int8) {
    TensorShape data_representation_4bit_shape{x->Shape()};
    MLDataType new_dtype = (x_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) ? DataTypeImpl::GetType<UInt4x2>() : DataTypeImpl::GetType<Int4x2>();
    auto memory_info = OrtMemoryInfo{
        WEBGPU_BUFFER,
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

  const auto& x_shape_intrinsic = x->Shape();
  // For bits == 2 with uint8 storage we don't construct a packed-type reinterpret (no UInt2x4 type
  // exists). Instead, build a logical "dequantized" shape (last dim x4) and feed that to the shader
  // as the input_shape uniform. The buffer remains the original uint8 storage with Flatten=4, and
  // the shader does explicit byte+bit-position extraction.
  TensorShape x_shape;
  if (bits_ == 2 && is_uint8) {
    TensorShapeVector v = x_shape_intrinsic.AsShapeVector();
    v.back() *= 4;
    x_shape = TensorShape(std::move(v));
  } else {
    x_shape = x_shape_intrinsic;
  }

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

  if (output_size == 0) {
    return Status::OK();
  }

  // For the 2-bit zero-point path we need to address the packed byte using the scale row index
  // and the within-row quantize-axis index (not the flat scales offset, which crosses row
  // boundaries when scale_qaxis_dim isn't a multiple of the packing factor). To keep the shader
  // simple we require quantize_axis to be the last dim for uint8 2-bit, matching the CPU kernel.
  if (bits_ == 2 && is_uint8) {
    ORT_RETURN_IF_NOT(quantize_axis == x_rank - 1,
                      "For uint8 2-bit data, quantize_axis must be the last dimension.");
  }
  const uint32_t scale_qaxis_dim = static_cast<uint32_t>(scales_shape[quantize_axis]);
  const uint32_t zp_packed_qaxis_dim = (scale_qaxis_dim + 3) / 4;

  GatherBlockQuantizedProgram program{is_signed, is_int8, indices_rank, gather_axis, bits_, zero_points != nullptr, x_shape, output_shape};

  program
      .AddInputs({{x, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, (bits_ == 4) ? 8 : 4}})
      .AddIndices(x_shape)
      .AddInputs({{indices, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddInputs({{scales, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(output_size)}})
      .AddUniformVariables({{static_cast<uint32_t>(quantize_axis)}})
      .AddUniformVariables({{static_cast<uint32_t>(gather_axis)}})
      .AddUniformVariables({{static_cast<uint32_t>(block_size_)}})
      .AddUniformVariables({{scale_qaxis_dim}})
      .AddUniformVariables({{zp_packed_qaxis_dim}})
      .CacheHint(std::to_string(bits_), std::to_string(gather_axis), std::to_string(quantize_axis), std::to_string(block_size_));

  if (zero_points != nullptr) {
    if (bits_ == 2 && is_uint8) {
      // 2-bit zero points are packed 4 per byte along the quantize axis.
      const auto& zp_shape = zero_points->Shape();
      ORT_RETURN_IF_NOT(zp_shape.NumDimensions() == scales_shape.NumDimensions(),
                        "scales and zero_points must have the same rank.");
      for (size_t i = 0; i < scales_shape.NumDimensions(); ++i) {
        int64_t expected = (i == static_cast<size_t>(quantize_axis))
                               ? (scales_shape[i] + 3) / 4
                               : scales_shape[i];
        ORT_RETURN_IF_NOT(zp_shape[i] == expected,
                          "zero_points shape does not match expected packed shape for 2-bit data.");
      }
    } else {
      ORT_RETURN_IF_NOT(scales_shape == zero_points->Shape(),
                        "scales and zero_points must have the same shape.");
    }
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
