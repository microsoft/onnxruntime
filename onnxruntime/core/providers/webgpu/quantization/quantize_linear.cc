// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <string_view>
#include <vector>

#include "core/util/math.h"
#include "core/providers/webgpu/quantization/quantize_linear.h"
#include "core/framework/int4.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace webgpu {

Status DequantizeLinearProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& scale = shader.AddInput("scale", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride | ShaderUsage::UseValueTypeAlias);

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
      << "let output_indices = " << output.OffsetToIndices("global_idx") << ";\n";

  // Get x input
  if (packing_ == PackingMode::Packed4) {
    // 4-bit packing: 8 elements per u32
    shader.MainFunctionBody()
        << "let x = " << x.GetByOffset("global_idx / 8") << ";\n"
        << "let x_raw = (x >> ((global_idx % 8u) * 4u)) & 0xFu;\n";
    if (packed_signed_) {
      shader.MainFunctionBody()
          << "let x_value = select(input_element_t(x_raw), input_element_t(x_raw) - 16, x_raw >= 8u);\n";
    } else {
      shader.MainFunctionBody()
          << "let x_value = input_element_t(x_raw);\n";
    }
  } else if (packing_ == PackingMode::Packed8) {
    // 8-bit packing: 4 elements per u32
    std::string unpack = (packed_signed_) ? "unpack4xI8(x)" : "unpack4xU8(x)";
    if (output.NumComponents() == 1) {
      shader.MainFunctionBody()
          << "let x = " << x.GetByOffset("global_idx / 4") << ";\n"
          << "let x_vec = " << unpack << ";\n"
          << "let x_value = x_vec[global_idx % 4];\n";
    } else {
      shader.MainFunctionBody()
          << "let x = " << x.GetByOffset("global_idx") << ";\n"
          << "let x_vec = " << unpack << ";\n"
          << "let x_value = x_vec;\n";
    }
  } else {
    shader.MainFunctionBody()
        << "let x_value = " << x.GetByOffset("global_idx") << ";\n";
  }

  // Get scaler
  if (per_layer_) {
    // scale input is a scalar ()
    shader.MainFunctionBody()
        << "let scale_value = " << scale.GetByOffset("0") << ";\n";
  } else if (per_axis_) {
    shader.MainFunctionBody()
        << "let scale_index = " << output.IndicesGet("output_indices", "uniforms.axis") << ";\n"
        << "let scale_value = " << scale.GetByOffset("scale_index") << ";\n";
  } else {
    // Block quantization. Scale input rank is same as input/output rank.
    // On the block axis, divide by block_size; on other axes, use output index directly.
    shader.MainFunctionBody() << "var scale_indices: scale_indices_t;\n";
    for (int i = 0; i < rank_; i++) {
      std::string idx = output.IndicesGet("output_indices", i);
      std::string value_expr = "select(" + idx + ", " + idx + " / uniforms.block_size, " + std::to_string(i) + "u == uniforms.axis)";
      shader.MainFunctionBody() << scale.IndicesSet("scale_indices", i, value_expr) << "\n";
    }
    shader.MainFunctionBody()
        << "let scale_value = " << scale.GetByIndices("scale_indices") << ";\n";
  }

  // Get zero-point
  if (has_zeropoint_) {
    const auto& zero_point = shader.AddInput("zero_point", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);

    if (packing_ == PackingMode::Packed4) {
      // 4-bit zero-point: 8 elements per u32, with sign extension for signed types
      std::string sign_extend_prefix = packed_signed_ ? "let zp_raw = " : "let zero_point_value = input_element_t(";
      std::string sign_extend_suffix = packed_signed_ ? ";\nlet zero_point_value = select(input_element_t(zp_raw), input_element_t(zp_raw) - 16, zp_raw >= 8u);\n"
                                                      : ");\n";
      if (per_layer_) {
        shader.MainFunctionBody()
            << sign_extend_prefix << zero_point.GetByOffset("0") << " & 0xFu" << sign_extend_suffix;
      } else if (per_axis_) {
        shader.MainFunctionBody()
            << "let zero_point_index = " << output.IndicesGet("output_indices", "uniforms.axis") << ";\n"
            << "let zero_point_packed = " << zero_point.GetByOffset("zero_point_index / 8") << ";\n"
            << sign_extend_prefix << "(zero_point_packed >> ((zero_point_index % 8u) * 4u)) & 0xFu" << sign_extend_suffix;
      } else {
        shader.MainFunctionBody()
            << "let zero_point_offset = " << scale.IndicesToOffset("scale_indices") << ";\n"
            << "let zero_point_packed = " << zero_point.GetByOffset("zero_point_offset / 8") << ";\n"
            << sign_extend_prefix << "(zero_point_packed >> ((zero_point_offset % 8u) * 4u)) & 0xFu" << sign_extend_suffix;
      }
    } else {
      std::string unpack = (packed_signed_) ? "unpack4xI8(zero_point_input)" : "unpack4xU8(zero_point_input)";
      if (per_layer_) {
        // zero-point input is a scalar
        if (packing_ == PackingMode::Packed8) {
          shader.MainFunctionBody()
              << "let zero_point_input = " << zero_point.GetByOffset("0") << ";\n"
              << "let zero_point_vec = " << unpack << ";\n"
              << "let zero_point_value = zero_point_vec[0];\n";
        } else {
          shader.MainFunctionBody()
              << "let zero_point_value = " << zero_point.GetByOffset("0") << ";\n";
        }
      } else if (per_axis_) {
        // zero-point input is a 1D tensor
        if (packing_ == PackingMode::Packed8) {
          shader.MainFunctionBody()
              << "let zero_point_index = " << output.IndicesGet("output_indices", "uniforms.axis") << ";\n"
              << "let zero_point_input = " << zero_point.GetByOffset("zero_point_index / 4") << ";\n"
              << "let zero_point_vec = " << unpack << ";\n"
              << "let zero_point_value = zero_point_vec[zero_point_index % 4];\n";
        } else {
          shader.MainFunctionBody()
              << "let zero_point_index = " << output.IndicesGet("output_indices", "uniforms.axis") << ";\n"
              << "let zero_point_value = " << zero_point.GetByOffset("zero_point_index") << ";\n";
        }
      } else {
        // BlockedQuantization. The zero-point input shape is the same as the scale input shape.
        if (packing_ == PackingMode::Packed8) {
          shader.MainFunctionBody()
              << "let zero_point_offset = " << scale.IndicesToOffset("scale_indices") << ";\n"
              << "let zero_point_input = " << zero_point.GetByOffset("zero_point_offset / 4") << ";\n"
              << "let zero_point_vec = " << unpack << ";\n"
              << "let zero_point_value = zero_point_vec[zero_point_offset % 4];\n";
        } else {
          shader.MainFunctionBody()
              << "let zero_point_offset = " << scale.IndicesToOffset("scale_indices") << ";\n"
              << "let zero_point_value = " << zero_point.GetByOffset("zero_point_offset") << ";\n";
        }
      }
    }
  } else {
    shader.MainFunctionBody()
        << "let zero_point_value = input_element_t(0);\n";
  }

  // compute and write output
  shader.MainFunctionBody()
      << output.SetByOffset("global_idx", "(output_value_t(x_value) - scale_value_t(zero_point_value)) * scale_value");

  return Status::OK();
}

namespace {

// How a scale (and zero point) tensor maps onto the quantized/dequantized tensor.
struct QuantizationLayout {
  bool per_layer = false;  // single scale for all elements
  bool per_axis = false;   // 1-D scale along `axis`
  int64_t axis = 0;        // normalized; meaningful when !per_layer
  int64_t block_size = 0;  // blocked quantization along `axis` when !per_layer && !per_axis
};

// Classifies scale/zero point as per-tensor, per-axis, or blocked and validates their shapes.
// Shared by QuantizeLinear and DequantizeLinear.
Status GetQuantizationLayout(const TensorShape& x_shape, const Tensor& scale, const Tensor* zero_point,
                             int64_t axis_attr, int64_t block_size_attr,
                             std::string_view scale_name, std::string_view zero_point_name,
                             QuantizationLayout& layout) {
  const auto& scale_shape = scale.Shape();
  const int64_t scale_rank = static_cast<int64_t>(scale_shape.NumDimensions());
  const int64_t x_rank = static_cast<int64_t>(x_shape.NumDimensions());
  int64_t axis = (axis_attr >= 0) ? axis_attr : axis_attr + x_rank;

  // scaler - single scaler for all elements
  bool per_layer = block_size_attr == 0 &&
                   (scale_rank == 0 || (scale_rank == 1 && scale_shape[0] == 1));

  // 1D tensor - 1 scaler for per axis
  bool per_axis = block_size_attr == 0 && per_layer == false && scale_rank == 1;
  ORT_RETURN_IF(!per_layer && (axis < 0 || axis >= x_rank), "axis must be within the input rank");
  ORT_RETURN_IF(!per_layer && !per_axis && scale_rank != x_rank,
                scale_name, " and x must have the same rank for blocked quantization");

  // Compute effective block_size. When block_size is 0 (default) but scale is 1D with
  // fewer elements than the input dimension on the axis, infer block_size from the ratio.
  int64_t block_size = block_size_attr;
  if (per_axis && block_size == 0) {
    int64_t input_dim = x_shape[onnxruntime::narrow<size_t>(axis)];
    int64_t scale_dim = scale_shape[0];
    if (scale_dim < input_dim) {
      ORT_RETURN_IF(scale_dim == 0, scale_name, " must be nonempty on a nonempty quantize axis");
      block_size = input_dim / scale_dim;
      per_axis = false;  // treat as block quantization
    }
  }

  // When scale is N-D (block quantization) and block_size is 0, infer axis and block_size
  // from the shapes. Find the dimension where scale is smaller than input to determine axis,
  // then compute block_size from the ratio.
  if (!per_layer && !per_axis && block_size == 0) {
    for (size_t i = 0; i < x_shape.NumDimensions(); i++) {
      if (scale_shape[i] < x_shape[i]) {
        ORT_RETURN_IF(scale_shape[i] == 0, scale_name, " must be nonempty on a nonempty quantize axis");
        axis = static_cast<int64_t>(i);
        block_size = x_shape[i] / scale_shape[i];
        break;
      }
    }
    if (block_size == 0) {
      block_size = 1;  // all dims match, default to block_size=1
    }
  }

  // Validate shapes for blocked quantization.
  if (!per_layer && !per_axis && block_size > 0) {
    ORT_RETURN_IF(scale_shape.NumDimensions() != x_shape.NumDimensions(),
                  scale_name, " and x must have the same rank for blocked quantization");
    for (size_t i = 0; i < x_shape.NumDimensions(); i++) {
      if (static_cast<int64_t>(i) == axis) {
        ORT_RETURN_IF(scale_shape[i] != (x_shape[i] + block_size - 1) / block_size,
                      scale_name, " must be ceil(Di/block_size) on the quantize axis i for blocked quantization");
      } else {
        ORT_RETURN_IF(scale_shape[i] != x_shape[i],
                      scale_name, " and x must have the same shape on non-quantize axes for blocked quantization");
      }
    }
    if (zero_point != nullptr) {
      ORT_RETURN_IF(zero_point->Shape().NumDimensions() != scale_shape.NumDimensions(),
                    zero_point_name, " and ", scale_name, " must have the same rank for blocked quantization");
      for (size_t i = 0; i < x_shape.NumDimensions(); i++) {
        ORT_RETURN_IF(zero_point->Shape()[i] != scale_shape[i],
                      zero_point_name, " and ", scale_name, " must have the same shape for blocked quantization");
      }
    }
  }

  if (per_layer) {
    ORT_RETURN_IF(zero_point != nullptr &&
                      !(zero_point->Shape().NumDimensions() == 0 ||
                        (zero_point->Shape().NumDimensions() == 1 && zero_point->Shape()[0] == 1)),
                  zero_point_name, " must be a scalar or size-1 vector for per-tensor quantization");
  } else if (per_axis) {
    const int64_t input_dim = x_shape[onnxruntime::narrow<size_t>(axis)];
    ORT_RETURN_IF(scale_shape[0] != input_dim,
                  scale_name, " must match the quantize axis for per-axis quantization");
    ORT_RETURN_IF(zero_point != nullptr &&
                      (zero_point->Shape().NumDimensions() != 1 || zero_point->Shape()[0] != input_dim),
                  zero_point_name, " must match ", scale_name, " for per-axis quantization");
  }

  layout.per_layer = per_layer;
  layout.per_axis = per_axis;
  layout.axis = axis;
  layout.block_size = block_size;
  return Status::OK();
}

}  // namespace

Status DequantizeLinear::ComputeInternal(ComputeContext& context) const {
  const auto* x = context.Input(0);
  const auto* x_scale = context.Input(1);
  const auto* x_zeropoint = context.Input(2);
  const auto x_shape = x->Shape();
  int64_t x_size = x_shape.Size();
  auto* output_tensor = context.Output(0, x_shape);

  auto x_type = x->GetElementType();

  PackingMode packing = (x_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4 || x_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4)
                            ? PackingMode::Packed4
                        : (x_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || x_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
                            ? PackingMode::Packed8
                            : PackingMode::None;
  bool packed = packing != PackingMode::None;
  bool is_packed_signed = x_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || x_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4;

  QuantizationLayout layout;
  ORT_RETURN_IF_ERROR(GetQuantizationLayout(x_shape, *x_scale, x_zeropoint, axis_, block_size_,
                                            "x_scale", "x_zero_point", layout));
  const bool per_layer = layout.per_layer;
  const bool per_axis = layout.per_axis;
  const int64_t axis = layout.axis;
  const int64_t block_size = layout.block_size;

  if (x_size == 0) {
    return Status::OK();
  }

  int max_components = GetMaxComponents(x_size);
  bool use_components = per_layer && packing != PackingMode::Packed4 && (!packed || max_components == 4);
  int components = use_components ? max_components : 1;
  int input_component = use_components ? max_components : 1;
  // For 4-bit types, each u32 holds 8 elements; for 8-bit types, 4 elements.
  int pack_factor = (packing == PackingMode::Packed4) ? 8 : 4;

  DequantizeLinearProgram program{packing, is_packed_signed, per_layer, per_axis, x_zeropoint != nullptr,
                                  static_cast<int>(x_shape.NumDimensions())};

  program
      .AddInputs({{x, ProgramTensorMetadataDependency::TypeAndRank, ProgramInput::Flatten, packed ? pack_factor : input_component}})
      .AddInputs({{x_scale, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutput(use_components
                     ? ProgramOutput{output_tensor, ProgramTensorMetadataDependency::TypeAndRank, ProgramOutput::Flatten, components}
                     : ProgramOutput{output_tensor, ProgramTensorMetadataDependency::TypeAndRank, components})
      .SetDispatchGroupSize((x_size / components + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(axis)}})
      .AddUniformVariables({{static_cast<uint32_t>(block_size)}})
      .AddUniformVariables({{static_cast<uint32_t>(x_size / components)}})
      .CacheHint(std::to_string(axis), std::to_string(is_packed_signed), std::to_string(per_layer), std::to_string(per_axis), std::to_string(block_size), std::to_string(static_cast<int>(packing)));

  if (x_zeropoint != nullptr) {
    program.AddInputs({{x_zeropoint, ProgramTensorMetadataDependency::None, ProgramInput::Flatten, packed ? pack_factor : 1}});
  }

  return context.RunProgram(program);
}

Status QuantizeLinearProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& scale = shader.AddInput("scale", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const ShaderVariableHelper* zero_point =
      has_zeropoint_ ? &shader.AddInput("zero_point", ShaderUsage::UseUniform) : nullptr;
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);

  const char* unpack = is_signed_ ? "unpack4xI8" : "unpack4xU8";
  // Reads the zero point at `offset` from 4-per-u32 packed storage.
  const auto zero_point_at = [&](const std::string& offset) {
    return MakeString("f32(", unpack, "(", zero_point->GetByOffset("(" + offset + ") / 4u"), ")[(", offset, ") % 4u])");
  };

  // WGSL f32 division is only accurate to 2.5 ULP, which can move a quotient across a .5 boundary
  // (e.g. -127.4999 vs -127.5). To match the CPU kernel, which rounds a correctly rounded f32
  // quotient half-to-even, the quotient is recomputed exactly with integer long division of the
  // mantissas. Only 0.25 <= |q| < 1024 needs it: smaller values round to 0 and larger ones saturate.
  shader.AdditionalImplementation() << R"(
fn quantize_round(x: f32, s: f32) -> f32 {
  let q = x / s;
  let xb = bitcast<u32>(x);
  let sb = bitcast<u32>(s);
  let x_exp = (xb >> 23u) & 0xffu;
  let s_exp = (sb >> 23u) & 0xffu;
  if (!(abs(q) >= 0.25 && abs(q) < 1024.0) || x_exp == 0u || x_exp == 255u || s_exp == 0u || s_exp == 255u) {
    return round(q);
  }
  let mx = (xb & 0x7fffffu) | 0x800000u;
  let ms = (sb & 0x7fffffu) | 0x800000u;
  // quo = floor(mx / ms * 2^25); rem != 0 means inexact.
  var rem = mx;
  var quo = 0u;
  for (var i = 0u; i < 26u; i++) {
    quo = quo << 1u;
    if (rem >= ms) {
      rem -= ms;
      quo |= 1u;
    }
    rem = rem << 1u;
  }
  // Keep 24 significant bits, then round to nearest even: |x / s| = sig * 2^e.
  var e = i32(x_exp) - i32(s_exp);
  var sig: u32;
  var guard: u32;
  var sticky = u32(rem != 0u);
  if ((quo & 0x2000000u) != 0u) {
    sig = quo >> 2u;
    guard = (quo >> 1u) & 1u;
    sticky |= quo & 1u;
    e -= 23;
  } else {
    sig = quo >> 1u;
    guard = quo & 1u;
    e -= 24;
  }
  if (guard == 1u && (sticky == 1u || (sig & 1u) == 1u)) {
    sig += 1u;
  }
  if (e >= 0 || e < -31) {
    return round(q);
  }
  // Round sig * 2^e to an integer, half to even.
  let shift = u32(-e);
  var n = sig >> shift;
  let frac = sig & ((1u << shift) - 1u);
  let half = 1u << (shift - 1u);
  if (frac > half || (frac == half && (n & 1u) == 1u)) {
    n += 1u;
  }
  return select(f32(n), -f32(n), ((xb ^ sb) & 0x80000000u) != 0u);
}
)";

  auto& body = shader.MainFunctionBody();
  body << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
       << "var quantized = vec4<i32>(0);\n"
       << "for (var i = 0u; i < 4u; i++) {\n"
       << "  let element_offset = global_idx * 4u + i;\n"
       << "  if (element_offset >= uniforms.input_size) {\n"
       << "    break;\n"
       << "  }\n"
       << "  let x_value = f32(" << x.GetByOffset("element_offset") << ");\n";

  std::string zero_point_value = "0.0";
  if (per_layer_) {
    body << "  let scale_value = f32(" << scale.GetByOffset("0") << ");\n";
    if (zero_point) {
      zero_point_value = zero_point_at("0u");
    }
  } else {
    body << "  let input_indices = " << x.OffsetToIndices("element_offset") << ";\n";
    if (per_axis_) {
      body << "  let scale_offset = " << x.IndicesGet("input_indices", "uniforms.axis") << ";\n"
           << "  let scale_value = f32(" << scale.GetByOffset("scale_offset") << ");\n";
    } else {
      // Blocked: on the block axis divide the index by block_size; other axes map 1:1.
      body << "  var scale_indices: scale_indices_t;\n";
      for (int i = 0; i < rank_; i++) {
        const std::string idx = x.IndicesGet("input_indices", i);
        body << "  " << scale.IndicesSet("scale_indices", i, "select(" + idx + ", " + idx + " / uniforms.block_size, " + std::to_string(i) + "u == uniforms.axis)") << "\n";
      }
      body << "  let scale_offset = " << scale.IndicesToOffset("scale_indices") << ";\n"
           << "  let scale_value = f32(" << scale.GetByIndices("scale_indices") << ");\n";
    }
    if (zero_point) {
      zero_point_value = zero_point_at("scale_offset");
    }
  }

  // Saturating in f32 before the i32 conversion also handles values far outside the integer range.
  body << "  let zero_point_value = " << zero_point_value << ";\n"
       << "  quantized[i] = i32(clamp(quantize_round(x_value, scale_value) + zero_point_value, "
       << (is_signed_ ? "-128.0, 127.0" : "0.0, 255.0") << "));\n"
       << "}\n"
       << output.SetByOffset("global_idx", "quantized");

  return Status::OK();
}

Status QuantizeLinear::ComputeInternal(ComputeContext& context) const {
  const auto* x = context.Input(0);
  const auto* y_scale = context.Input(1);
  const auto* y_zero_point = context.Input(2);
  const auto& x_shape = x->Shape();
  auto* y = context.Output(0, x_shape);

  QuantizationLayout layout;
  ORT_RETURN_IF_ERROR(GetQuantizationLayout(x_shape, *y_scale, y_zero_point, axis_, block_size_,
                                            "y_scale", "y_zero_point", layout));

  const int64_t x_size = x_shape.Size();
  if (x_size == 0) {
    return Status::OK();
  }

  const auto y_type = y->GetElementType();
  ORT_RETURN_IF_NOT(y_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || y_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                    "WebGPU QuantizeLinear only supports int8 and uint8 output, got type ", y_type);
  const bool is_signed = y_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  const uint32_t output_size = onnxruntime::narrow<uint32_t>((x_size + 3) / 4);

  QuantizeLinearProgram program{is_signed, layout.per_layer, layout.per_axis, y_zero_point != nullptr,
                                static_cast<int>(x_shape.NumDimensions())};
  program
      .AddInputs({{x, ProgramTensorMetadataDependency::TypeAndRank},
                  {y_scale, ProgramTensorMetadataDependency::TypeAndRank}});
  if (y_zero_point != nullptr) {
    program.AddInput({y_zero_point, ProgramTensorMetadataDependency::None, ProgramInput::Flatten, 4});
  }
  program
      .AddOutput({y, ProgramTensorMetadataDependency::None, ProgramOutput::Flatten, 4})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(layout.axis)},
                            {static_cast<uint32_t>(layout.block_size)},
                            {output_size},
                            {onnxruntime::narrow<uint32_t>(x_size)}})
      .CacheHint(is_signed, layout.per_layer, layout.per_axis, y_zero_point != nullptr);

  return context.RunProgram(program);
}

namespace {
const std::vector<MLDataType>& DequantizeLinearConstraints() {
  static std::vector<MLDataType> types{
      DataTypeImpl::GetTensorType<int8_t>(),
      DataTypeImpl::GetTensorType<uint8_t>(),
      DataTypeImpl::GetTensorType<int32_t>(),
      DataTypeImpl::GetTensorType<UInt4x2>(),
      DataTypeImpl::GetTensorType<Int4x2>()};
  return types;
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    DequantizeLinear,
    kOnnxDomain,
    10, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DequantizeLinearConstraints()),
    DequantizeLinear);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    DequantizeLinear,
    kOnnxDomain,
    13, 18,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DequantizeLinearConstraints()),
    DequantizeLinear);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    DequantizeLinear,
    kOnnxDomain,
    19, 20,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DequantizeLinearConstraints())
        .TypeConstraint("T2", WebGpuSupportedFloatTypes()),
    DequantizeLinear);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    DequantizeLinear,
    kOnnxDomain,
    21, 22,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DequantizeLinearConstraints())
        .TypeConstraint("T2", WebGpuSupportedFloatTypes()),
    DequantizeLinear);

ONNX_OPERATOR_KERNEL_EX(
    DequantizeLinear,
    kOnnxDomain,
    23,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DequantizeLinearConstraints())
        .TypeConstraint("T2", WebGpuSupportedFloatTypes()),
    DequantizeLinear);

namespace {
const std::vector<MLDataType>& QuantizeLinearOutputConstraints() {
  static std::vector<MLDataType> types{
      DataTypeImpl::GetTensorType<int8_t>(),
      DataTypeImpl::GetTensorType<uint8_t>()};
  return types;
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    QuantizeLinear,
    kOnnxDomain,
    10, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", QuantizeLinearOutputConstraints()),
    QuantizeLinear);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    QuantizeLinear,
    kOnnxDomain,
    13, 18,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", QuantizeLinearOutputConstraints()),
    QuantizeLinear);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    QuantizeLinear,
    kOnnxDomain,
    19, 20,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", WebGpuSupportedFloatTypes())
        .TypeConstraint("T2", QuantizeLinearOutputConstraints()),
    QuantizeLinear);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    QuantizeLinear,
    kOnnxDomain,
    21, 22,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", WebGpuSupportedFloatTypes())
        .TypeConstraint("T2", QuantizeLinearOutputConstraints()),
    QuantizeLinear);

// Opset 23 adds the `precision` attribute, which defaults to the y_scale type. The kernel divides in
// f32, so it takes f32 y_scale only; GetCapability rejects any other explicit precision. Opsets 24, 25
// and 28 only add y_scale/y types that this kernel does not register.
#define WEBGPU_QUANTIZE_LINEAR_VERSIONED_KERNEL(start, end)           \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                  \
      QuantizeLinear,                                                 \
      kOnnxDomain,                                                    \
      start,                                                          \
      end,                                                            \
      kWebGpuExecutionProvider,                                       \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T1", WebGpuSupportedFloatTypes())          \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("T3", QuantizeLinearOutputConstraints()),   \
      QuantizeLinear);

#define WEBGPU_QUANTIZE_LINEAR_KERNEL(version)                        \
  ONNX_OPERATOR_KERNEL_EX(                                            \
      QuantizeLinear,                                                 \
      kOnnxDomain,                                                    \
      version,                                                        \
      kWebGpuExecutionProvider,                                       \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T1", WebGpuSupportedFloatTypes())          \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("T3", QuantizeLinearOutputConstraints()),   \
      QuantizeLinear);

WEBGPU_QUANTIZE_LINEAR_VERSIONED_KERNEL(23, 23)
WEBGPU_QUANTIZE_LINEAR_VERSIONED_KERNEL(24, 24)
WEBGPU_QUANTIZE_LINEAR_VERSIONED_KERNEL(25, 27)
WEBGPU_QUANTIZE_LINEAR_KERNEL(28)

}  // namespace webgpu
}  // namespace onnxruntime
