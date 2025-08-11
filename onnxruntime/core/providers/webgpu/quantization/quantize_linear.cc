// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "core/util/math.h"
#include "core/providers/webgpu/quantization/quantize_linear.h"
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
  if (packed_) {
    std::string unpack = (signed_) ? "unpack4xI8(x)" : "unpack4xU8(x)";
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
    shader.MainFunctionBody()
        << "var scale_indices: scale_indices_t = output_indices;\n"
        << "let index = " << scale.IndicesGet("scale_indices", "uniforms.axis") << "/ uniforms.block_size;\n"
        << scale.IndicesSet("scale_indices", "uniforms.axis", "index") << ";\n"
        << "let scale_value = " << scale.GetByIndices("scale_indices") << ";\n";
  }

  // Get zero-point
  if (has_zeropoint_) {
    const auto& zero_point = shader.AddInput("zero_point", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);

    std::string unpack = (signed_) ? "unpack4xI8(zero_point_input)" : "unpack4xU8(zero_point_input)";
    if (per_layer_) {
      // zero-point input is a scalar
      if (packed_) {
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
      if (packed_) {
        shader.MainFunctionBody()
            << "let zero_point_index = " << output.IndicesGet("output_indices", "uniforms.axis") << ";\n"
            << "let zero_point_input = " << zero_point.GetByOffset("u32(zero_point_index / 4)") << ";\n"
            << "let zero_point_vec = " << unpack << ";\n"
            << "let zero_point_value = zero_point_vec[zero_point_index % 4];\n";
      } else {
        shader.MainFunctionBody()
            << "let zero_point_index = " << output.IndicesGet("output_indices", "uniforms.axis") << ";\n"
            << "let zero_point_value = " << zero_point.GetByOffset("zero_point_index") << ";\n";
      }
    } else {
      // BlockedQuantization. The zero-point input shape is same as the input shape except along axis.
      if (packed_) {
        shader.MainFunctionBody()
            << "let zero_point_offset = " << scale.GetByIndices("scale_indices") << ";\n"
            << "let zero_point_input = " << zero_point.GetByOffset("u32(zero_point_offset / 4)") << ";\n"
            << "let zero_point_vec = " << unpack << ";\n"
            << "let zero_point_value = zero_point_vec[zero_point_offset % 4];\n";
      } else {
        shader.MainFunctionBody()
            << "let zero_point_value = " << zero_point.GetByIndices("scale_indices") << ";\n";
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

  // Currently only INT8, UINT8, and INT32 are registered.
  auto x_type = x->GetElementType();

  bool packed = x_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || x_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  bool is_signed = x_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  int64_t axis = (axis_ >= 0) ? axis_ : axis_ + x_shape.NumDimensions();

  int max_components = GetMaxComponents(x_size);

  // scaler - single scaler for all elements
  bool per_layer = x_scale_rank == 0 || (x_scale_rank == 1 && x_scale->Shape()[0] == 1);

  // 1D tensor - 1 scaler for per axis
  bool per_axis = per_layer == false && x_scale_rank == 1;

  bool use_components = per_layer && (!packed || max_components == 4);
  int components = use_components ? max_components : 1;
  int input_component = use_components ? max_components : 1;

  DequantizeLinearProgram program{packed, is_signed, per_layer, per_axis, x_zeropoint != nullptr};

  program
      .AddInputs({{x, ProgramTensorMetadataDependency::TypeAndRank, ProgramInput::Flatten, packed ? 4 : input_component}})
      .AddInputs({{x_scale, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::Rank, components})
      .SetDispatchGroupSize((x_size / components + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(axis)}})
      .AddUniformVariables({{static_cast<uint32_t>(block_size_)}})
      .AddUniformVariables({{static_cast<uint32_t>(x_size / components)}})
      .CacheHint(std::to_string(axis), std::to_string(is_signed), std::to_string(per_layer), std::to_string(per_axis), std::to_string(block_size_));

  if (x_zeropoint != nullptr) {
    program.AddInputs({{x_zeropoint, ProgramTensorMetadataDependency::None, ProgramInput::Flatten, packed ? 4 : 1}});
  }

  return context.RunProgram(program);
}

namespace {
const std::vector<MLDataType>& DequantizeLinearConstraints() {
  static std::vector<MLDataType> types{
      DataTypeImpl::GetTensorType<int8_t>(),
      DataTypeImpl::GetTensorType<uint8_t>(),
      DataTypeImpl::GetTensorType<int32_t>()};
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
