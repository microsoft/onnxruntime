// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/string_macros.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "contrib_ops/webgpu/bert/skip_layer_norm.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

Status SkipLayerNormProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddInput("skip", ShaderUsage::UseUniform);
  shader.AddInput("gamma", ShaderUsage::UseUniform);
  if (hasBeta_) {
    shader.AddInput("beta", ShaderUsage::UseUniform);
  }
  if (hasBias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseUniform);
  if (has_input_skip_bias_sum_) {
    shader.AddOutput("input_skip_bias_sum", ShaderUsage::UseUniform);
  }

  std::string simpl1 = (simplified_) ? "" : "- mean * mean ";
  std::string simpl2 = (simplified_) ? "" : "- x_element_t(mean) ";
  if (split_hidden_dim_) {
    shader.AdditionalImplementation()
        << "var<workgroup> sum_shared : array<f32, workgroup_size_x>;\n"
        << "var<workgroup> sum_squared_shared : array<f32, workgroup_size_x>;\n";

    SS(input_skip_bias_sum_ss, 512);
    if (has_input_skip_bias_sum_) {
      input_skip_bias_sum_ss
          << "  let workgroup_half_idx = uniforms.hidden_size / (workgroup_size_x * 4);\n"
          << "  if (workgroup_idx >= workgroup_half_idx) {\n"
          << "    offset = (workgroup_idx - workgroup_half_idx) * workgroup_size_x + local_idx;\n"
          << "    let skip_value = skip[offset];\n"
          << "    let input_value = x[offset];\n"
          << "    let value = input_value + skip_value" << (hasBias_ ? " + bias[offset]" : "") << ";\n"
          << "    input_skip_bias_sum[offset] = value;\n"
          << "    return;\n"
          << "  }\n";
    }

    shader.MainFunctionBody()
        << "  var offset: u32 = 0;\n"
        << (has_input_skip_bias_sum_ ? SS_GET(input_skip_bias_sum_ss) : "")
        << "  var sum_vec4 = vec4<f32>(0);\n"
        << "  var sum_squared_vec4 = vec4<f32>(0);\n"
        << "  var cur_input_skip_bias_sum = x_value_t(0);\n"
        << "  for (var i: u32 = 0; i < uniforms.hidden_size / (workgroup_size_x * 4); i++) {\n"
        << "    let input_offset = i * workgroup_size_x + local_idx;\n"
        << "    let skip_value = skip[input_offset];\n"
        << "    let input_value = x[input_offset];\n"
        << "    let value = input_value + skip_value" << (hasBias_ ? " + bias[input_offset]" : "") << ";\n"
        << "    if (i == workgroup_idx) {\n"
        << "      cur_input_skip_bias_sum = value;\n"
        << "    }\n"
        << "    let f32_value = vec4<f32>(value);\n"
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
        << "  let mean = sum_shared[0] / f32(uniforms.hidden_size);\n"
        << "  let inv_std_dev = inverseSqrt(sum_squared_shared[0] / f32(uniforms.hidden_size) " << simpl1 << "+ uniforms.epsilon);\n"
        << "  offset = workgroup_idx * workgroup_size_x + local_idx;\n"
        << "  output[offset] = ((cur_input_skip_bias_sum " << simpl2 << ") * x_element_t(inv_std_dev) * gamma[offset]" << (hasBeta_ ? " + beta[offset] " : "") << ");\n";
  } else {
    int components = x.NumComponents();
    std::string bias = (hasBias_) ? " + bias[offset1d + i] " : "";
    std::string beta = (hasBeta_) ? " + beta[offset1d + i] " : "";
    std::string input_skip_bias_sum = (has_input_skip_bias_sum_) ? "input_skip_bias_sum[offset + i] = value;\n" : "";

    shader.AdditionalImplementation()
        << "alias f32_val_t = " << (components == 4 ? "vec4<f32>" : (components == 2 ? "vec2<f32>" : "f32")) << ";\n"
        << "var<workgroup> sum_shared : array<f32_val_t, workgroup_size_x>;\n"
        << "var<workgroup> sum_squared_shared : array<f32_val_t, workgroup_size_x>;\n";

    shader.MainFunctionBody()
        << "let ix = local_idx;\n"
        << "let iy = global_idx / workgroup_size_x;\n"
        << "let hidden_size_vectorized: u32 = uniforms.hidden_size / uniforms.components;\n"
        << "var stride = hidden_size_vectorized / workgroup_size_x;\n"
        << "let offset = ix * stride + iy * hidden_size_vectorized;\n"
        << "let offset1d = stride * ix;\n"
        << "if (ix == workgroup_size_x - 1) {\n"
        << " stride = hidden_size_vectorized - stride * ix;\n"
        << "}\n"
        << "for (var i: u32 = 0; i < stride; i++) {\n"
        << " let skip_value = skip[offset + i];\n"
        << " let input_value = x[offset + i];\n"
        << " let value = input_value + skip_value" << bias << ";\n"
        << " output[offset + i] = value;\n"
        << input_skip_bias_sum
        << " let f32_value = f32_val_t(value);\n"
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
        << "let mean = " << SumVector("sum", components) << " / f32(uniforms.hidden_size);\n"
        << "let inv_std_dev = inverseSqrt(" << SumVector("square_sum", components) << " / f32(uniforms.hidden_size) " << simpl1 << "+ uniforms.epsilon);\n"
        << "for (var i: u32 = 0; i < stride; i++) {\n"
        << " output[offset + i] = (output[offset + i] " << simpl2 << ") * x_element_t(inv_std_dev) * gamma[offset1d + i]" << beta << ";\n"
        << "};\n";
  }

  return Status::OK();
}

template <bool simplified>
Status SkipLayerNorm<simplified>::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const Tensor* x = context.Input(0);
  const Tensor* skip = context.Input(1);
  const Tensor* gamma = context.Input(2);
  // optional
  const Tensor* beta = context.Input(3);
  const Tensor* bias = context.Input(4);

  const auto x_shape = x->Shape();

  auto* output = context.Output(0, x_shape);
  auto* input_skip_bias_sum = context.Output(3, x_shape);

  int64_t data_size = x_shape.Size();
  if (data_size == 0) {
    return Status::OK();
  }

  const uint32_t hidden_size = onnxruntime::narrow<uint32_t>(x_shape[x_shape.NumDimensions() - 1]);
  const int components = GetMaxComponents(hidden_size);
  const bool has_input_skip_bias_sum = input_skip_bias_sum != nullptr;
  const uint32_t norm_count = onnxruntime::narrow<uint32_t>(x_shape.SizeToDimension(x_shape.NumDimensions() - 1));
  const bool split_hidden_dim = hidden_size % 512 == 0 && norm_count == 1;

  SkipLayerNormProgram program{beta != nullptr, bias != nullptr, epsilon_, hidden_size, has_input_skip_bias_sum, simplified, split_hidden_dim};
  program
      .CacheHint(simplified, has_input_skip_bias_sum, split_hidden_dim)
      .AddInputs({{x, ProgramTensorMetadataDependency::Type, components}})
      .AddInputs({{skip, ProgramTensorMetadataDependency::Type, components}})
      .AddInputs({{gamma, ProgramTensorMetadataDependency::Type, components}})
      .AddOutputs({{output, ProgramTensorMetadataDependency::None, components}})
      .SetDispatchGroupSize(onnxruntime::narrow<uint32_t>(ceil(1.0 * data_size / hidden_size)))
      .AddUniformVariables({
          {static_cast<uint32_t>(components)},
      })
      .AddUniformVariables({
          {static_cast<uint32_t>(hidden_size)},
      })
      .AddUniformVariables({
          {static_cast<float>(epsilon_)},
      });

  if (split_hidden_dim) {
    const uint32_t workgroup_size_x = 128;
    const uint32_t dispatch_size_x = (has_input_skip_bias_sum ? 2 : 1) * hidden_size / (workgroup_size_x * components);
    program.SetDispatchGroupSize(dispatch_size_x, 1, 1)
        .SetWorkgroupSize(workgroup_size_x);
  }

  if (beta != nullptr) {
    program.AddInput({beta, ProgramTensorMetadataDependency::Type, components});
  }
  if (bias != nullptr) {
    program.AddInput({bias, ProgramTensorMetadataDependency::Type, components});
  }
  if (has_input_skip_bias_sum) {
    program.AddOutputs({{input_skip_bias_sum, ProgramTensorMetadataDependency::None, components}});
  }
  return context.RunProgram(program);
}

ONNX_OPERATOR_KERNEL_EX(
    SkipLayerNormalization,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()),
    SkipLayerNorm<false>);

ONNX_OPERATOR_KERNEL_EX(
    SkipSimplifiedLayerNormalization,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()),
    SkipLayerNorm<true>);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
