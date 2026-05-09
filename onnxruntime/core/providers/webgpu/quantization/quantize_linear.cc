// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

Status DequantizeLinear::ComputeInternal(ComputeContext& context) const {
  const auto* x = context.Input(0);
  const auto* x_scale = context.Input(1);
  const auto* x_zeropoint = context.Input(2);
  const auto x_shape = x->Shape();
  int64_t x_size = x_shape.Size();
  auto* output_tensor = context.Output(0, x_shape);
  int64_t x_scale_rank = x_scale->Shape().NumDimensions();

  auto x_type = x->GetElementType();

  PackingMode packing = (x_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4 || x_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4)
                            ? PackingMode::Packed4
                        : (x_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || x_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
                            ? PackingMode::Packed8
                            : PackingMode::None;
  bool packed = packing != PackingMode::None;
  bool is_packed_signed = x_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || x_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4;
  int64_t axis = (axis_ >= 0) ? axis_ : axis_ + x_shape.NumDimensions();

  int max_components = GetMaxComponents(x_size);

  // scaler - single scaler for all elements
  bool per_layer = x_scale_rank == 0 || (x_scale_rank == 1 && x_scale->Shape()[0] == 1);

  // 1D tensor - 1 scaler for per axis
  bool per_axis = per_layer == false && x_scale_rank == 1;

  // Compute effective block_size. When block_size_ is 0 (default) but scale is 1D with
  // fewer elements than the input dimension on the axis, infer block_size from the ratio.
  int64_t block_size = block_size_;
  if (per_axis && block_size == 0) {
    int64_t input_dim = x_shape[onnxruntime::narrow<size_t>(axis)];
    int64_t scale_dim = x_scale->Shape()[0];
    if (scale_dim < input_dim) {
      block_size = input_dim / scale_dim;
      per_axis = false;  // treat as block quantization
    }
  }

  // When scale is N-D (block quantization) and block_size is 0, infer axis and block_size
  // from the shapes. Find the dimension where scale is smaller than input to determine axis,
  // then compute block_size from the ratio.
  if (!per_layer && !per_axis && block_size == 0) {
    const auto& scale_shape = x_scale->Shape();
    for (size_t i = 0; i < x_shape.NumDimensions(); i++) {
      if (scale_shape[i] < x_shape[i]) {
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
    const auto& scale_shape = x_scale->Shape();
    ORT_RETURN_IF(scale_shape.NumDimensions() != x_shape.NumDimensions(),
                  "x_scale and x must have the same rank for blocked quantization");
    for (size_t i = 0; i < x_shape.NumDimensions(); i++) {
      if (static_cast<int64_t>(i) == axis) {
        ORT_RETURN_IF(scale_shape[i] != (x_shape[i] + block_size - 1) / block_size,
                      "x_scale must be ceil(Di/block_size) on the quantize axis i for blocked quantization");
      } else {
        ORT_RETURN_IF(scale_shape[i] != x_shape[i],
                      "x_scale and x must have the same shape on non-quantize axes for blocked quantization");
      }
    }
    if (x_zeropoint != nullptr) {
      for (size_t i = 0; i < x_shape.NumDimensions(); i++) {
        ORT_RETURN_IF(x_zeropoint->Shape()[i] != scale_shape[i],
                      "x_zero_point and x_scale must have the same shape for blocked quantization");
      }
    }
  }

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
                     ? ProgramOutput{output_tensor, ProgramTensorMetadataDependency::Rank, ProgramOutput::Flatten, components}
                     : ProgramOutput{output_tensor, ProgramTensorMetadataDependency::Rank, components})
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
