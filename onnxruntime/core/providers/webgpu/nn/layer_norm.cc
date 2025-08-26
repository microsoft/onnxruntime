
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"
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

  std::string simpl1 = (simplified_) ? "" : "- mean * mean ";
  std::string simpl2 = (simplified_) ? "" : "- x_element_t(mean) ";

  if (split_norm_dim_) {
    shader.AdditionalImplementation()
        << "var<workgroup> sum_shared : array<f32, workgroup_size_x>;\n"
        << "var<workgroup> sum_squared_shared : array<f32, workgroup_size_x>;\n";

    shader.MainFunctionBody()
        << "  var sum_vec4 = vec4<f32>(0);\n"
        << "  var sum_squared_vec4 = vec4<f32>(0);\n"
        << "  var cur_input = x_value_t(0);\n"
        << "  for (var i: u32 = 0; i < uniforms.norm_size / (workgroup_size_x * 4); i++) {\n"
        << "    let input_offset = i * workgroup_size_x + local_idx;\n"
        << "    let input_value = x[input_offset];\n"
        << "    if (i == workgroup_idx) {\n"
        << "      cur_input = input_value;\n"
        << "    }\n"
        << "    let f32_value = vec4<f32>(input_value);\n"
        << "    sum_vec4 += f32_value;\n"
        << "    sum_squared_vec4 += f32_value * f32_value;\n"
        << "  }\n"
        << "  var sum = " << SumVector("sum_vec4", 4) << ";\n"
        << "  var sum_squared = " << SumVector("sum_squared_vec4", 4) << ";\n"
        << "  sum_shared[local_idx] = sum;\n"
        << "  sum_squared_shared[local_idx] = sum_squared;\n"
        << "  workgroupBarrier();\n"
        << "  var reduce_size : u32 = workgroup_size_x;\n"
        << "  for (var curr_size = reduce_size >> 1;  curr_size > 0; curr_size = reduce_size >> 1) {\n"
        << "    reduce_size = curr_size + (reduce_size & 1);\n"
        << "    if (local_idx < curr_size) {\n"
        << "      sum_shared[local_idx] += sum_shared[local_idx + reduce_size];\n"
        << "      sum_squared_shared[local_idx] += sum_squared_shared[local_idx + reduce_size];\n"
        << "    }\n"
        << "    workgroupBarrier();\n"
        << "  }\n"
        << "  let mean = sum_shared[0] / f32(uniforms.norm_size);\n"
        << "  let inv_std_dev = inverseSqrt(sum_squared_shared[0] / f32(uniforms.norm_size) " << simpl1 << "+ uniforms.epsilon);\n"
        << "  let offset = workgroup_idx * workgroup_size_x + local_idx;\n"
        << "  y[offset] = ((cur_input " << simpl2 << ") * x_element_t(inv_std_dev) * scale[offset]" << (has_bias_ ? " + bias[offset] " : "") << ");\n";

    if (has_mean_output_) {
      shader.MainFunctionBody() << "  if (local_idx == 0 && workgroup_idx == 0) {\n"
                                << "    mean_output[global_idx / uniforms.norm_size] = mean;\n"
                                << "  }\n";
    }
    if (has_inv_std_dev_output_) {
      shader.MainFunctionBody() << "  if (local_idx == 0 && workgroup_idx == 0) {\n"
                                << "    inv_std_dev_output[global_idx / uniforms.norm_size] = inv_std_dev;\n"
                                << "  }\n";
    }
  } else {
    int components = x.NumComponents();
    std::string bias = (has_bias_) ? " + bias[offset1d + i] " : "";

    shader.AdditionalImplementation()
        << "alias f32_val_t = " << (components == 4 ? "vec4<f32>" : (components == 2 ? "vec2<f32>" : "f32")) << ";\n"
        << "var<workgroup> sum_shared : array<f32_val_t, workgroup_size_x>;\n"
        << "var<workgroup> sum_squared_shared : array<f32_val_t, workgroup_size_x>;\n";

    shader.MainFunctionBody()
        << "let ix = local_idx;\n"
        << "let iy = global_idx / workgroup_size_x;\n"
        << "let norm_size_vectorized: u32 = uniforms.norm_size / uniforms.components;\n"
        << "var stride = norm_size_vectorized / workgroup_size_x;\n"
        << "let offset = ix * stride + iy * norm_size_vectorized;\n"
        << "let offset1d = stride * ix;\n"
        << "sum_shared[ix] = f32_val_t(0);\n"
        << "sum_squared_shared[ix] = f32_val_t(0);\n"
        << "if (ix == workgroup_size_x - 1) {\n"
        << " stride = norm_size_vectorized - stride * ix;\n"
        << "}\n"
        << "for (var i: u32 = 0; i < stride; i++) {\n"
        << " let input_value = x[offset + i];\n"
        << " y[offset + i] = input_value;\n"
        << " let f32_value = f32_val_t(input_value);\n"
        << " sum_shared[ix] += f32_value;\n"
        << " sum_squared_shared[ix] += f32_value * f32_value;\n"
        << "}\n"
        << "workgroupBarrier();\n"
        << "var reduce_size : u32 = workgroup_size_x;\n"
        << "for (var curr_size = reduce_size >> 1;  curr_size > 0; curr_size = reduce_size >> 1) {\n"
        << " reduce_size = curr_size + (reduce_size & 1);\n"
        << " if (ix < curr_size) {\n"
        << "  sum_shared[ix] += sum_shared[ix + reduce_size];\n"
        << "  sum_squared_shared[ix] += sum_squared_shared[ix + reduce_size];\n"
        << " }\n"
        << " workgroupBarrier();\n"
        << "}\n"
        << "let sum = sum_shared[0];\n"
        << "let square_sum = sum_squared_shared[0];\n"
        << "let mean = " << SumVector("sum", components) << " / f32(uniforms.norm_size);\n"
        << "let inv_std_dev = inverseSqrt(" << SumVector("square_sum", components) << " / f32(uniforms.norm_size) " << simpl1 << "+ uniforms.epsilon);\n"
        << "for (var i: u32 = 0; i < stride; i++) {\n"
        << " y[offset + i] = (y[offset + i] " << simpl2 << ") * x_element_t(inv_std_dev) * scale[offset1d + i]" << bias << ";\n"
        << "};\n";

    if (has_mean_output_) {
      shader.MainFunctionBody() << "if (ix == 0) {\n"
                                << "  mean_output[iy] = mean;\n"
                                << "}\n";
    }
    if (has_inv_std_dev_output_) {
      shader.MainFunctionBody() << "if (ix == 0) {\n"
                                << "  inv_std_dev_output[iy] = inv_std_dev;\n"
                                << "}\n";
    }
  }

  return Status::OK();
}

template <bool simplified>
Status LayerNorm<simplified>::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* x = context.Input(0);
  const auto* scale = context.Input(1);
  const auto* bias = context.Input(2);

  const auto x_shape = x->Shape();

  const bool is_fp16 = x->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;

  const size_t axis = NormalizeAxis(axis_, x_shape.NumDimensions());
  const uint32_t norm_count = onnxruntime::narrow<uint32_t>(x_shape.SizeToDimension(axis));
  const int64_t norm_size = x_shape.SizeFromDimension(axis);
  const int components = GetMaxComponents(norm_size);
  const uint32_t norm_size_vectorized = onnxruntime::narrow<uint32_t>((norm_size + components - 1) / components);

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

  if (x_shape.Size() == 0) {
    return Status::OK();
  }

  // Check if we should use split norm dimension optimization
  const bool split_norm_dim = norm_size % 512 == 0 && norm_count == 1;

  LayerNormProgram program{bias != nullptr, is_fp16, simplified, mean != nullptr, inv_std_dev != nullptr, split_norm_dim};

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
          {static_cast<float>(epsilon_)},
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
