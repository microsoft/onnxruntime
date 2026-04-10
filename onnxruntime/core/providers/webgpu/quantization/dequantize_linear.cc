// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "core/util/math.h"
#include "core/providers/webgpu/quantization/dequantize_linear.h"
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
  if (packing_mode_ == util::U32PackingMode::Pack4bx8) {
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
  } else if (packing_mode_ == util::U32PackingMode::Pack8bx4) {
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
  if (quantization_type_ == util::QuantizationType::PerTensor) {
    // scale input is a scalar ()
    shader.MainFunctionBody()
        << "let scale_value = " << scale.GetByOffset("0") << ";\n";
  } else if (quantization_type_ == util::QuantizationType::PerAxis) {
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

    if (packing_mode_ == util::U32PackingMode::Pack4bx8) {
      // 4-bit zero-point: 8 elements per u32, with sign extension for signed types
      std::string sign_extend_prefix = packed_signed_ ? "let zp_raw = " : "let zero_point_value = input_element_t(";
      std::string sign_extend_suffix = packed_signed_ ? ";\nlet zero_point_value = select(input_element_t(zp_raw), input_element_t(zp_raw) - 16, zp_raw >= 8u);\n"
                                                      : ");\n";
      if (quantization_type_ == util::QuantizationType::PerTensor) {
        shader.MainFunctionBody()
            << sign_extend_prefix << zero_point.GetByOffset("0") << " & 0xFu" << sign_extend_suffix;
      } else if (quantization_type_ == util::QuantizationType::PerAxis) {
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
      if (quantization_type_ == util::QuantizationType::PerTensor) {
        // zero-point input is a scalar
        if (packing_mode_ == util::U32PackingMode::Pack8bx4) {
          shader.MainFunctionBody()
              << "let zero_point_input = " << zero_point.GetByOffset("0") << ";\n"
              << "let zero_point_vec = " << unpack << ";\n"
              << "let zero_point_value = zero_point_vec[0];\n";
        } else {
          shader.MainFunctionBody()
              << "let zero_point_value = " << zero_point.GetByOffset("0") << ";\n";
        }
      } else if (quantization_type_ == util::QuantizationType::PerAxis) {
        // zero-point input is a 1D tensor
        if (packing_mode_ == util::U32PackingMode::Pack8bx4) {
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
        if (packing_mode_ == util::U32PackingMode::Pack8bx4) {
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

Status DequantizeLinear::ComputeInternal(ComputeContext& context) const {
  const auto* x = context.Input(0);
  const auto* x_scale = context.Input(1);
  const auto* x_zeropoint = context.Input(2);
  const auto& x_shape = x->Shape();
  const auto x_size = x_shape.Size();
  auto* output_tensor = context.Output(0, x_shape);

  const auto x_type = x->GetElementType();
  const auto packing_mode = util::GetOnnxTensorElementDataTypeU32PackingMode(x_type);
  const bool packed = packing_mode != util::U32PackingMode::None;
  const bool is_packed_signed = packed && util::IsOnnxElementDataTypeSigned(x_type);

  util::QuantizationType quantization_type{};
  int64_t axis = axis_;
  int64_t block_size = block_size_;
  ORT_RETURN_IF_ERROR(util::ValidateAndDetectQuantizationType(x_shape,
                                                              x_scale->Shape(),
                                                              x_zeropoint ? &x_zeropoint->Shape() : nullptr,
                                                              axis,
                                                              block_size,
                                                              quantization_type));

  const int max_components = GetMaxComponents(x_size);

  const int pack_factor = util::GetU32PackingModeNumComponents(packing_mode).value_or(1);

  const bool use_components = quantization_type == util::QuantizationType::PerTensor &&
                              packing_mode != util::U32PackingMode::Pack4bx8 &&
                              (!packed || max_components == 4);
  const int components = use_components ? max_components : 1;
  const int input_component = use_components ? max_components : 1;

  DequantizeLinearProgram program{packing_mode, is_packed_signed, quantization_type, x_zeropoint != nullptr,
                                  static_cast<int>(x_shape.NumDimensions())};

  uint32_t axis_uniform = 0;
  uint32_t block_size_uniform = 1;
  if (quantization_type == util::QuantizationType::PerAxis || quantization_type == util::QuantizationType::Blocked) {
    axis_uniform = narrow<uint32_t>(axis);
  }
  if (quantization_type == util::QuantizationType::Blocked) {
    block_size_uniform = narrow<uint32_t>(block_size);
  }

  program
      .AddInputs({{x, ProgramTensorMetadataDependency::TypeAndRank, ProgramInput::Flatten, packed ? pack_factor : input_component}})
      .AddInputs({{x_scale, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutput(use_components
                     ? ProgramOutput{output_tensor, ProgramTensorMetadataDependency::Rank, ProgramOutput::Flatten, components}
                     : ProgramOutput{output_tensor, ProgramTensorMetadataDependency::Rank, components})
      .SetDispatchGroupSize((x_size / components + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{axis_uniform}})
      .AddUniformVariables({{block_size_uniform}})
      .AddUniformVariables({{static_cast<uint32_t>(x_size / components)}})
      .CacheHint(std::to_string(static_cast<int>(quantization_type)), std::to_string(is_packed_signed),
                 std::to_string(static_cast<int>(packing_mode)));

  if (x_zeropoint != nullptr) {
    program.AddInputs({{x_zeropoint, ProgramTensorMetadataDependency::None, ProgramInput::Flatten, packed ? pack_factor : 1}});
  }

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

}  // namespace webgpu
}  // namespace onnxruntime
