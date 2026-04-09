// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/quantization/quantize_linear.h"

#include "core/providers/common.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime::webgpu {

// QuantizeLinearProgram

QuantizeLinearProgram::QuantizeLinearProgram(util::QuantizationType quantization_type, bool has_zero_point,
                                             uint32_t workgroup_size, int32_t y_element_data_type)
    : Program<QuantizeLinearProgram>{"QuantizeLinear"},
      quantization_type_{quantization_type},
      has_zero_point_{has_zero_point},
      workgroup_size_{workgroup_size},
      y_is_signed_{util::IsOnnxElementDataTypeSigned(y_element_data_type)},
      y_packing_mode_{util::GetOnnxTensorElementDataTypePackingMode(y_element_data_type)} {
}

Status QuantizeLinearProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("x");
  shader.AddInput("y_scale");

  if (has_zero_point_) {
    shader.AddInput("y_zero_point");
  }

  shader.AddOutput("y");

  return WGSL_TEMPLATE_APPLY(shader, "quantization/quantize_linear.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(HAS_ZERO_POINT, has_zero_point_),
                             WGSL_TEMPLATE_PARAMETER(QUANTIZATION_TYPE, quantization_type_),
                             WGSL_TEMPLATE_PARAMETER(WORKGROUP_SIZE, workgroup_size_),
                             WGSL_TEMPLATE_PARAMETER(Y_IS_SIGNED, y_is_signed_));
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

  if (quantization_type == util::QuantizationType::Blocked) {
    ORT_NOT_IMPLEMENTED("unsupported quantization type: Blocked");
  }

  QuantizeLinearProgram program{quantization_type, y_zero_point != nullptr, WORKGROUP_SIZE, y->GetElementType()};

  const auto x_components = 1;

  const auto y_components = 4;  // uint8/int8 use 4 components

  // Compute per-axis uniforms.
  uint32_t axis_stride = 1;
  uint32_t scale_dim_on_axis = 1;
  if (quantization_type == util::QuantizationType::PerAxis) {
    scale_dim_on_axis = narrow<uint32_t>(y_scale_shape[0]);
    axis_stride = narrow<uint32_t>(x_shape.SizeFromDimension(static_cast<size_t>(axis) + 1));
  }

  program.AddInput(ProgramInput{x, ProgramTensorMetadataDependency::TypeAndRank, x_components});
  program.AddInput(ProgramInput{y_scale, ProgramTensorMetadataDependency::TypeAndRank});
  if (y_zero_point != nullptr) {
    program.AddInput(ProgramInput{y_zero_point, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten,
                                  y_components});
  }

  program.AddOutput(ProgramOutput{y, ProgramTensorMetadataDependency::Type, ProgramOutput::Flatten, y_components});

  program.SetDispatchGroupSize(CeilDiv<decltype(WORKGROUP_SIZE)>(x_size, WORKGROUP_SIZE));

  program.AddUniformVariables({
      {narrow<uint32_t>(x_size)},   // data_size
      {axis_stride},                // axis_stride
      {scale_dim_on_axis},          // scale_dim_on_axis
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
  static const auto constraints = BuildKernelDefConstraints<uint8_t, int8_t>();
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
