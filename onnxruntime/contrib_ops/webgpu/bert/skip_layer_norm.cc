// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "contrib_ops/webgpu/bert/skip_layer_norm.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

static uint32_t GetMaxComponents(int size) {
  if (size % 4 == 0) {
    return 4;
  } else if (size % 2 == 0) {
    return 2;
  }
  return 1;
}

static std::string SumVector(std::string x, int components) {
  switch (components) {
    case 1:
      return x;
    case 2:
      return "(" + x + ".x + " + x + ".y" + ")";
    case 4:
      return "(" + x + ".x + " + x + ".y + " + x + ".w + " + x + ".z" + ")";
    default:
      ORT_THROW("Unsupported number of components: ", components);
  }
}

Status SkipLayerNormProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
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

  int components = x.NumComponents();

  std::string bias = (hasBias_) ? " + bias[offset1d + i] " : "";
  std::string simpl1 = (simplified_) ? "" : "- mean * mean ";
  std::string simpl2 = (simplified_) ? "" : "- element_t(mean) ";
  std::string beta = (hasBeta_) ? " + beta[offset1d + i] " : "";
  std::string input_skip_bias_sum = (has_input_skip_bias_sum_) ? "input_skip_bias_sum[offset + i] = value;\n" : "";

  shader.AdditionalImplementation()
      << "alias element_t = " << (is_fp16_ ? "f16;\n" : "f32;\n")
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
      << " output[offset + i] = (output[offset + i] " << simpl2 << ") * element_t(inv_std_dev) * gamma[offset1d + i]" << beta << ";\n"
      << "};\n";

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

  const bool is_fp16 = x->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  const uint32_t hidden_size = gsl::narrow<uint32_t>(x_shape[x_shape.NumDimensions() - 1]);
  const int components = GetMaxComponents(hidden_size);
  const bool has_input_skip_bias_sum = input_skip_bias_sum != nullptr;

  SkipLayerNormProgram program{beta != nullptr, bias != nullptr, epsilon_, hidden_size, has_input_skip_bias_sum, is_fp16, simplified};
  program
      .CacheHint(simplified, has_input_skip_bias_sum)
      .AddInputs({{x, ProgramTensorMetadataDependency::Type, components}})
      .AddInputs({{skip, ProgramTensorMetadataDependency::Type, components}})
      .AddInputs({{gamma, ProgramTensorMetadataDependency::Type, components}})
      .AddOutputs({{output, ProgramTensorMetadataDependency::None, components}})
      .SetDispatchGroupSize(gsl::narrow<uint32_t>(ceil(1.0 * data_size / hidden_size)))
      .AddUniformVariables({
          {static_cast<uint32_t>(components)},
      })
      .AddUniformVariables({
          {static_cast<uint32_t>(hidden_size)},
      })
      .AddUniformVariables({
          {static_cast<float>(epsilon_)},
      });

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
