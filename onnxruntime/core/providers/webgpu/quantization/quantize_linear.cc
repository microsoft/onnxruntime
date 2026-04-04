// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/quantization/quantize_linear.h"

#include "core/providers/common.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime::webgpu {

// QuantizeLinearProgram

Status QuantizeLinearProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("x");
  shader.AddInput("y_scale");

  if (has_zero_point_) {
    shader.AddInput("y_zero_point");
  }

  shader.AddOutput("y");

  return WGSL_TEMPLATE_APPLY(shader, "quantization/quantize_linear.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_zero_point, has_zero_point_));
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

  QuantizeLinearProgram program{quantization_type, y_zero_point != nullptr};

  const auto x_components = GetMaxComponents(x_shape.Size());

  // set up program...
  program.AddInput(ProgramInput{x, ProgramTensorMetadataDependency::TypeAndShape, x_components});
  program.AddInput(ProgramInput{y_scale, ProgramTensorMetadataDependency::TypeAndShape});
  if (y_zero_point != nullptr) {
    program.AddInput(ProgramInput{y_zero_point, ProgramTensorMetadataDependency::Type});
  }

  program.AddOutput(ProgramOutput{y, ProgramTensorMetadataDependency::Type});

  program.SetDispatchGroupSize(CeilDiv<decltype(WORKGROUP_SIZE)>(x_shape.Size(), WORKGROUP_SIZE));

  return context.RunProgram(program);
}

namespace {
const std::vector<MLDataType>& InputTypeConstraints() {
  static const auto constraints = BuildKernelDefConstraints<float>();
  return constraints;
  // return WebGpuSupportedFloatTypes();
}

const std::vector<MLDataType>& OutputAndZeroPointTypeConstraints() {
  static const auto constraints = BuildKernelDefConstraints<int8_t>();
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

}  // namespace onnxruntime::webgpu
