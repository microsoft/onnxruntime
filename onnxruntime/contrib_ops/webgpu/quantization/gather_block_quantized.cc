// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstring>

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "contrib_ops/webgpu/quantization/gather_block_quantized.h"
#if !defined(DISABLE_FLOAT8_TYPES)
#include "core/common/float8.h"
#endif
#if !defined(DISABLE_FLOAT4_TYPES)
#include "core/framework/float4.h"
#endif

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

namespace {
// Builds the WGSL `const` dequantization lookup table for an FP8 or FP4 `data` type: table[code]
// is the float value of the code, computed once host-side via ORT's own (already-tested)
// Float8E*/Float4E2M1x2 -> float conversions, so the shader never needs to reproduce FP8/FP4 bit
// manipulation itself. FP8 has 256 possible byte codes; FP4 has 16 (one nibble).
std::string BuildFpDequantLutWgsl(int32_t fp_elem_type) {
  std::vector<float> table;
#if !defined(DISABLE_FLOAT8_TYPES)
  if (fp_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN ||
      fp_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ ||
      fp_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2 ||
      fp_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ) {
    table.reserve(256);
    for (int i = 0; i < 256; ++i) {
      const auto byte = static_cast<uint8_t>(i);
      switch (fp_elem_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
          table.push_back(Float8E4M3FN(byte, Float8E4M3FN::FromBits()).ToFloat());
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
          table.push_back(Float8E4M3FNUZ(byte, Float8E4M3FNUZ::FromBits()).ToFloat());
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
          table.push_back(Float8E5M2(byte, Float8E5M2::FromBits()).ToFloat());
          break;
        default:
          table.push_back(Float8E5M2FNUZ(byte, Float8E5M2FNUZ::FromBits()).ToFloat());
          break;
      }
    }
  }
#endif  // !defined(DISABLE_FLOAT8_TYPES)
#if !defined(DISABLE_FLOAT4_TYPES)
  if (fp_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1) {
    table.reserve(16);
    for (int i = 0; i < 16; ++i) {
      // Float4E2M1x2 packs element 0 in the low nibble (shift 0); build a code with that nibble
      // set to `i` and read element 0 back out, giving the decode for a raw 4-bit code `i`.
      table.push_back(Float4E2M1x2(static_cast<uint8_t>(i), Float4E2M1x2::FromBits()).GetElem(0));
    }
  }
#endif  // !defined(DISABLE_FLOAT4_TYPES)

  std::ostringstream oss;
  oss << std::setprecision(9);
  oss << "const kFpDequantLut = array<f32, " << table.size() << ">(";
  for (size_t i = 0; i < table.size(); ++i) {
    if (i > 0) oss << ", ";
    // NaN/Inf (reserved codes in some FP8 layouts, e.g. E5M2) have no valid WGSL float-literal
    // spelling ("nan"/"inf" text is not a WGSL token); encode them via a bit-pattern reinterpret
    // instead so the const array always parses, even though such codes are unlikely to appear in
    // real quantized data.
    if (std::isfinite(table[i])) {
      oss << table[i] << "f";
    } else {
      uint32_t bits;
      std::memcpy(&bits, &table[i], sizeof(bits));
      oss << "bitcast<f32>(" << bits << "u)";
    }
  }
  oss << ");\n";
  return oss.str();
}
}  // namespace

Status GatherBlockQuantizedProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("input", ShaderUsage::UseElementTypeAlias);
  const auto& x_shape = shader.AddIndices("input_shape", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& indices = shader.AddInput("indices", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseIndicesToOffset | ShaderUsage::UseValueTypeAlias);
  const auto& scales = shader.AddInput("scales", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride | ShaderUsage::UseValueTypeAlias);

  const bool is_2bit = bits_ == 2;
  const bool is_4bit = bits_ == 4;
  const std::string unpack = (is_signed_) ? "unpack4xI8" : "unpack4xU8";

  if (is_fp_quantized_) {
    shader.AdditionalImplementation() << BuildFpDequantLutWgsl(fp_elem_type_);
  }

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
      << "  var dequantized_data = output_value_t(0);\n";
  if (is_fp_quantized_) {
    shader.MainFunctionBody()
        << "  dequantized_data = output_value_t(kFpDequantLut[quantized_data]) * scale;\n";
  } else {
    shader.MainFunctionBody()
        << "  dequantized_data = (output_value_t(quantized_data) - output_value_t(zero_point)) * scale;\n";
  }
  shader.MainFunctionBody()
      << "  " << output.SetByOffset("global_idx", "dequantized_data") << ";\n";

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
  bool is_fp4 = false;
  bool is_fp8 = false;
#if !defined(DISABLE_FLOAT4_TYPES)
  is_fp4 = x_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1;
#endif  // !defined(DISABLE_FLOAT4_TYPES)
#if !defined(DISABLE_FLOAT8_TYPES)
  is_fp8 = x_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN ||
           x_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ ||
           x_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2 ||
           x_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ;
#endif  // !defined(DISABLE_FLOAT8_TYPES)
  bool is_fp_quantized = is_fp4 || is_fp8;

  // `bits_`/`block_size_` are the raw attribute values. FP8/FP4 data is not governed by `bits`
  // (a byte or nibble is dequantized wholesale via a lookup table), so use a fixed effective bit
  // width for shader/packing purposes instead of the (irrelevant) attribute value.
  const int bits = is_fp_quantized ? (is_fp4 ? 4 : 8) : bits_;

  if (is_fp_quantized) {
    ORT_RETURN_IF_NOT(zero_points == nullptr, "zero_points must not be provided when data is an FP8 or FP4 type.");
  } else {
    // Only uint8 storage supports the full bits set {2, 4, 8}. The packed int4/uint4 types
    // can only carry bits==4, matching the CPU kernel's constraint.
    if (is_uint8) {
      ORT_RETURN_IF_NOT(bits_ == 2 || bits_ == 4 || bits_ == 8,
                        "'bits' must be 2, 4 or 8 for uint8 input.");
    } else {
      ORT_RETURN_IF_NOT(bits_ == 4, "'bits' must be 4 for non-uint8 input.");
    }
  }

  std::optional<Tensor> data_representation_4bit;
  std::optional<Tensor> zero_points_representation_4bit;
  if (bits == 4 && is_int8) {
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

  // The WebGPU program layer only knows how to derive a WGSL storage type for a fixed set of
  // ONNX element types (see ToProgramVariableDataType in core/providers/webgpu/program.cc), which
  // does not include the FP8/FP4 element types. The shader treats `x` as raw packed bytes/nibbles
  // regardless (looking up dequantized values via `kFpDequantLut`), so reinterpret the tensor as
  // the equivalent already-supported packed integer type (UInt4x2 for FP4, uint8_t for FP8)
  // without changing its shape or underlying data.
  std::optional<Tensor> data_representation_fp;
  if (is_fp_quantized) {
    MLDataType new_dtype = is_fp4 ? DataTypeImpl::GetType<UInt4x2>() : DataTypeImpl::GetType<uint8_t>();
    auto memory_info = OrtMemoryInfo{
        WEBGPU_BUFFER,
        OrtDeviceAllocator,
        OrtDevice{OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE, 0}};
    data_representation_fp.emplace(
        new_dtype,
        x->Shape(),
        const_cast<void*>(x->DataRaw()),
        memory_info);
    x = &data_representation_fp.value();
  }

  const auto& x_shape_intrinsic = x->Shape();
  // For bits == 2 with uint8 storage we don't construct a packed-type reinterpret (no UInt2x4 type
  // exists). Instead, build a logical "dequantized" shape (last dim x4) and feed that to the shader
  // as the input_shape uniform. The buffer remains the original uint8 storage with Flatten=4, and
  // the shader does explicit byte+bit-position extraction.
  // Native Float4E2M1x2 tensors (like Int4x2/UInt4x2) already report the logical (unpacked) shape,
  // so no special-casing is needed for FP4 here.
  TensorShape x_shape;
  if (bits == 2 && is_uint8) {
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

  // block_size == 0 (only valid for FP8/FP4 data) means the whole quantize_axis dimension is a
  // single block, i.e. one scale per row.
  int64_t effective_block_size = block_size_;
  if (effective_block_size == 0) {
    ORT_RETURN_IF_NOT(is_fp_quantized, "block_size=0 is only valid for FP8/FP4 data.");
    effective_block_size = x_shape[quantize_axis];
  }

  ORT_RETURN_IF_NOT(x_shape.NumDimensions() == scales_rank,
                    "data and scales must have the same rank.");
  for (size_t i = 0; i < x_shape.NumDimensions(); ++i) {
    ORT_RETURN_IF_NOT(i == static_cast<size_t>(quantize_axis)
                          ? (x_shape[i] * 1 + effective_block_size - 1) / effective_block_size == scales_shape[i]
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
  if (bits == 2 && is_uint8) {
    ORT_RETURN_IF_NOT(quantize_axis == x_rank - 1,
                      "For uint8 2-bit data, quantize_axis must be the last dimension.");
  }
  const uint32_t scale_qaxis_dim = static_cast<uint32_t>(scales_shape[quantize_axis]);
  const uint32_t zp_packed_qaxis_dim = (scale_qaxis_dim + 3) / 4;

  GatherBlockQuantizedProgram program{is_signed && !is_fp_quantized, is_int8, indices_rank, gather_axis, bits,
                                      zero_points != nullptr, x_shape, output_shape, is_fp_quantized,
                                      static_cast<int32_t>(x_dtype)};

  program
      .AddInputs({{x, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, (bits == 4) ? 8 : 4}})
      .AddIndices(x_shape)
      .AddInputs({{indices, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddInputs({{scales, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(output_size)}})
      .AddUniformVariables({{static_cast<uint32_t>(quantize_axis)}})
      .AddUniformVariables({{static_cast<uint32_t>(gather_axis)}})
      .AddUniformVariables({{static_cast<uint32_t>(effective_block_size)}})
      .AddUniformVariables({{scale_qaxis_dim}})
      .AddUniformVariables({{zp_packed_qaxis_dim}})
      .CacheHint(std::to_string(bits), std::to_string(gather_axis), std::to_string(quantize_axis),
                 std::to_string(effective_block_size), std::to_string(x_dtype));

  if (zero_points != nullptr) {
    if (bits == 2 && is_uint8) {
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
    program.AddInputs({{zero_points, ProgramTensorMetadataDependency::None, ProgramInput::Flatten, (bits == 4) ? 8 : 4}});
  }

  return context.RunProgram(program);
}

namespace {
const std::vector<MLDataType>& GatherBlockQuantizedT1Constraint() {
  static std::vector<MLDataType> types = [] {
    std::vector<MLDataType> t{
        DataTypeImpl::GetTensorType<Int4x2>(),
        DataTypeImpl::GetTensorType<UInt4x2>(),
        DataTypeImpl::GetTensorType<uint8_t>()};
#if !defined(DISABLE_FLOAT8_TYPES)
    t.push_back(DataTypeImpl::GetTensorType<Float8E4M3FN>());
    t.push_back(DataTypeImpl::GetTensorType<Float8E4M3FNUZ>());
    t.push_back(DataTypeImpl::GetTensorType<Float8E5M2>());
    t.push_back(DataTypeImpl::GetTensorType<Float8E5M2FNUZ>());
#endif  // !defined(DISABLE_FLOAT8_TYPES)
#if !defined(DISABLE_FLOAT4_TYPES)
    t.push_back(DataTypeImpl::GetTensorType<Float4E2M1x2>());
#endif  // !defined(DISABLE_FLOAT4_TYPES)
    return t;
  }();
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
