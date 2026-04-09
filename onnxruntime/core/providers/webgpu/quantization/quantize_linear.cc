// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/quantization/quantize_linear.h"

#include "core/framework/int4.h"
#include "core/providers/common.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime::webgpu {

// QuantizeLinearProgram

QuantizeLinearProgram::QuantizeLinearProgram(util::QuantizationType quantization_type, bool has_zero_point,
                                             int32_t y_element_data_type)
    : Program<QuantizeLinearProgram>{"QuantizeLinear"},
      quantization_type_{quantization_type},
      has_zero_point_{has_zero_point},
      y_is_signed_{util::IsOnnxElementDataTypeSigned(y_element_data_type)},
      y_packing_mode_{util::GetOnnxTensorElementDataTypePackingMode(y_element_data_type)} {
}

Status QuantizeLinearProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x_var = shader.AddInput("x", ShaderUsage::UseShapeAndStride | ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  shader.AddInput("y_scale", ShaderUsage::UseShapeAndStride | ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  if (has_zero_point_) {
    shader.AddInput("y_zero_point");
  }

  shader.AddOutput("y");

  return WGSL_TEMPLATE_APPLY(shader, "quantization/quantize_linear.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(HAS_ZERO_POINT, has_zero_point_),
                             WGSL_TEMPLATE_PARAMETER(QUANTIZATION_TYPE, quantization_type_),
                             WGSL_TEMPLATE_PARAMETER(Y_IS_SIGNED, y_is_signed_),
                             WGSL_TEMPLATE_PARAMETER(Y_PACKING_MODE, y_packing_mode_),
                             WGSL_TEMPLATE_VARIABLE(x, x_var));
}

// QuantizeLinear

Status QuantizeLinear::ComputeInternal(ComputeContext& context) const {
  const auto* x = context.Input(0);
  const auto* y_scale = context.Input(1);
  const auto* y_zero_point = context.Input(2);

  const auto& x_shape = x->Shape();
  const auto x_size = x_shape.Size();

  const auto& y_scale_shape = y_scale->Shape();

  auto* y = context.Output(0, x_shape);

  util::QuantizationType quantization_type{};
  int64_t axis = axis_;
  int64_t block_size = block_size_;
  ORT_RETURN_IF_ERROR(util::DetectQuantizationType(x_shape, y_scale_shape, axis, block_size, quantization_type));

  QuantizeLinearProgram program{quantization_type, y_zero_point != nullptr, y->GetElementType()};

  const auto x_components = 1;

  const auto y_packing_mode = util::GetOnnxTensorElementDataTypePackingMode(y->GetElementType());
  int y_components;
  ORT_RETURN_IF_ERROR(util::GetU32PackingNumComponents(y_packing_mode, y_components));

  uint32_t axis_uniform = 0;
  uint32_t block_size_uniform = 1;

  if (quantization_type == util::QuantizationType::PerAxis ||
      quantization_type == util::QuantizationType::Blocked) {
    axis_uniform = narrow<uint32_t>(axis);
  }
  if (quantization_type == util::QuantizationType::Blocked) {
    block_size_uniform = narrow<uint32_t>(block_size);
  }

  program.AddInput(ProgramInput{x, ProgramTensorMetadataDependency::TypeAndRank, x_components});
  program.AddInput(ProgramInput{y_scale, ProgramTensorMetadataDependency::TypeAndRank});
  if (y_zero_point != nullptr) {
    program.AddInput(ProgramInput{y_zero_point, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten,
                                  y_components});
  }

  program.AddOutput(ProgramOutput{y, ProgramTensorMetadataDependency::Type, ProgramOutput::Flatten, y_components});

  // Each thread packs y_components elements into one u32, so dispatch ceil(x_size/y_components) threads.
  const auto output_size = CeilDiv(x_size, static_cast<int64_t>(y_components));
  program.SetDispatchGroupSize(CeilDiv<decltype(WORKGROUP_SIZE)>(output_size, WORKGROUP_SIZE));

  program.AddUniformVariables({
      {narrow<uint32_t>(x_size)},  // data_size
      {axis_uniform},              // axis
      {block_size_uniform},        // block_size
  });

  return context.RunProgram(program);
}

namespace {
const std::vector<MLDataType>& InputTypeConstraints() {
  static const auto constraints = BuildKernelDefConstraints<float>();
  return constraints;
  // return WebGpuSupportedFloatTypes();
}

const std::vector<MLDataType>& OutputAndZeroPointTypeConstraints() {
  static const auto constraints = BuildKernelDefConstraints<uint8_t, int8_t, UInt4x2, Int4x2>();
  return constraints;
}
}  // namespace

// Kernel registration
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    QuantizeLinear,
    kOnnxDomain,
    10, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", InputTypeConstraints())
        .TypeConstraint("T2", OutputAndZeroPointTypeConstraints()),
    QuantizeLinear);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    QuantizeLinear,
    kOnnxDomain,
    13, 18,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", InputTypeConstraints())
        .TypeConstraint("T2", OutputAndZeroPointTypeConstraints()),
    QuantizeLinear);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    QuantizeLinear,
    kOnnxDomain,
    19, 20,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", InputTypeConstraints())
        .TypeConstraint("T2", OutputAndZeroPointTypeConstraints()),
    QuantizeLinear);

ONNX_OPERATOR_KERNEL_EX(
    QuantizeLinear,
    kOnnxDomain,
    21,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", InputTypeConstraints())
        .TypeConstraint("T2", OutputAndZeroPointTypeConstraints()),
    QuantizeLinear);

}  // namespace onnxruntime::webgpu
