
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/wgsl_templates/wgsl_gen.h"
#include "core/providers/webgpu/nn/layer_norm.h"

namespace onnxruntime {
namespace webgpu {

static size_t NormalizeAxis(int64_t axis, size_t tensor_rank) {
  int64_t rank = static_cast<int64_t>(tensor_rank);
  if (axis < -rank && axis >= rank) {
    ORT_THROW("invalid axis: ", axis);
  }
  return onnxruntime::narrow<size_t>(axis < 0 ? axis + rank : axis);
}

// Get a dummy override shape to bypass the program's check of shape and components for inputs and outputs. It's okay,
// as 'LayerNormProgram' doesn't actually use the override shape.
static TensorShape GetOverrideShape(const TensorShape& shape, int components) {
  TensorShape override_shape{shape.Size() / components};
  return override_shape;
}

Status LayerNormProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddInput("scale", ShaderUsage::UseUniform);
  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  shader.AddOutput("y", ShaderUsage::UseUniform);
  if (has_mean_output_) {
    shader.AddOutput("mean_output", ShaderUsage::None);
  }
  if (has_inv_std_dev_output_) {
    shader.AddOutput("inv_std_dev_output", ShaderUsage::None);
  }

  int components = x.NumComponents();

  return WGSL_TEMPLATE_APPLY(shader, "nn/layer_norm.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(components, components),
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
                             WGSL_TEMPLATE_PARAMETER(has_inv_std_dev_output, has_inv_std_dev_output_),
                             WGSL_TEMPLATE_PARAMETER(has_mean_output, has_mean_output_),
                             WGSL_TEMPLATE_PARAMETER(simplified, simplified_),
                             WGSL_TEMPLATE_PARAMETER(split_norm_dim, split_norm_dim_));
}

template <bool simplified>
Status LayerNorm<simplified>::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* x = context.Input(0);
  const auto* scale = context.Input(1);
  const auto* bias = context.Input(2);

  const auto x_shape = x->Shape();

  const size_t axis = NormalizeAxis(axis_, x_shape.NumDimensions());
  const uint32_t norm_count = onnxruntime::narrow<uint32_t>(x_shape.SizeToDimension(axis));
  const int64_t norm_size = x_shape.SizeFromDimension(axis);

  const auto scale_size = scale->Shape().Size();
  const auto bias_size = (bias) ? bias->Shape().Size() : 0;
  if (scale_size != norm_size || (bias && bias_size != norm_size)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Size of X.shape()[axis:] == ", norm_size,
                           ". Size of scale and bias (if provided) must match this. Got scale size of ",
                           scale_size, " and bias size of ", bias_size);
  }

  TensorShapeVector mean_dim;
  for (size_t i = 0; i < x_shape.NumDimensions(); ++i) {
    if (i < axis) {
      mean_dim.push_back(x_shape[i]);
    } else {
      mean_dim.push_back(1);
    }
  }
  TensorShape mean_shape(mean_dim);

  auto* y = context.Output(0, x_shape);
  auto* mean = context.Output(1, mean_shape);
  auto* inv_std_dev = context.Output(2, mean_shape);

  return RunLayerNormProgram(context, x, scale, bias, epsilon_, norm_count, norm_size,
                             simplified, y, mean, inv_std_dev);
}

Status RunLayerNormProgram(ComputeContext& context,
                           const Tensor* x,
                           const Tensor* scale,
                           const Tensor* bias,
                           float epsilon,
                           uint32_t norm_count,
                           int64_t norm_size,
                           bool simplified,
                           Tensor* y,
                           Tensor* mean,
                           Tensor* inv_std_dev) {
  if (x->Shape().Size() == 0) {
    return Status::OK();
  }
  const int components = GetMaxComponents(norm_size);
  const uint32_t norm_size_vectorized = onnxruntime::narrow<uint32_t>((norm_size + components - 1) / components);

  // Check if we should use split norm dimension optimization
  const bool split_norm_dim = norm_size % 512 == 0 && norm_count == 1;

  LayerNormProgram program{bias != nullptr, simplified, mean != nullptr, inv_std_dev != nullptr, split_norm_dim};

  program.CacheHint(components, simplified, split_norm_dim)
      .AddInputs({{x, ProgramTensorMetadataDependency::Type, GetOverrideShape(x->Shape(), components), components}})
      .AddInputs(
          {{scale, ProgramTensorMetadataDependency::Type, GetOverrideShape(scale->Shape(), components), components}})
      .AddOutputs({{y, ProgramTensorMetadataDependency::None, GetOverrideShape(y->Shape(), components), components}})
      .AddUniformVariables({
          {static_cast<uint32_t>(components)},
      })
      .AddUniformVariables({
          {static_cast<uint32_t>(norm_count)},
      })
      .AddUniformVariables({
          {static_cast<uint32_t>(norm_size)},
      })
      .AddUniformVariables({
          {static_cast<uint32_t>(norm_size_vectorized)},
      })
      .AddUniformVariables({
          {static_cast<float>(epsilon)},
      });

  if (split_norm_dim) {
    const uint32_t workgroup_size_x = 128;
    const uint32_t dispatch_size_x = onnxruntime::narrow<uint32_t>(norm_size / (workgroup_size_x * components));
    program.SetDispatchGroupSize(dispatch_size_x, 1, 1)
        .SetWorkgroupSize(workgroup_size_x);
  } else {
    program.SetDispatchGroupSize(norm_count);
  }

  if (bias != nullptr) {
    program.AddInput(
        {bias, ProgramTensorMetadataDependency::Type, GetOverrideShape(bias->Shape(), components), components});
  }

  if (mean != nullptr) {
    program.AddOutputs({{mean, ProgramTensorMetadataDependency::None}});
  }
  if (inv_std_dev != nullptr) {
    program.AddOutputs({{inv_std_dev, ProgramTensorMetadataDependency::None}});
  }

  return context.RunProgram(program);
}

ONNX_OPERATOR_KERNEL_EX(LayerNormalization, kOnnxDomain, 17, kWebGpuExecutionProvider,
                        (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()),
                        LayerNorm<false>);

ONNX_OPERATOR_KERNEL_EX(SimplifiedLayerNormalization, kOnnxDomain, 1, kWebGpuExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", WebGpuSupportedFloatTypes())
                            .TypeConstraint("U", WebGpuSupportedFloatTypes()),
                        LayerNorm<true>);

}  // namespace webgpu
}  // namespace onnxruntime
