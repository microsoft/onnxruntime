// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/skip_layer_norm.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

static uint32_t getMaxComponents(int size) {
  if (size % 4 == 0) {
    return 4;
  } else if (size % 2 == 0) {
    return 2;
  }
  return 1;
}

static std::string fillVar(std::string dataType, int components, std::string value) {
  if (components == 1) {
    return dataType + "(" + value + ")";
  }
  return "vec" + std::to_string(components) + "<" + dataType + ">(" + value + ")";
}

static std::string sumVector(std::string x, int components) {
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

static std::string castToF32(int components, std::string value) {
  if (components == 1) {
    return "f32(" + value + ")";
  }
  return "vec" + std::to_string(components) + "<f32>(" + value + ")";
};

static std::string vecDataType(std::string datatype, int components) {
  if (components == 1) {
    return datatype;
  }
  return "vec" + std::to_string(components) + "<" + datatype + ">";
};

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

  int components = x.NumComponents();

  std::string bias = (hasBias_) ? " + bias[offset1d + i] " : "";
  std::string simpl1 = (simplified_) ? "" : "- mean * mean ";
  std::string simpl2 = (simplified_) ? "" : "- element_t(mean) ";
  std::string fillvec = fillVar("f32", components, "0");
  std::string beta = (hasBeta_) ? " + beta[offset1d + i] " : "";
  std::string element_type = (isFP16_) ? "f16;\n" : "f32;\n";

  shader.AppendImplementation(
      "alias element_t = " + element_type +
      "var<workgroup> sum_shared : array<" + vecDataType("f32", components) +
      ",workgroup_size_x>;\n"
      "var<workgroup> sum_squared_shared : array<" +
      vecDataType("f32", components) + ",workgroup_size_x>;\n");

  std::stringstream ss;
  ss << "let ix = local_idx;\n"
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
     << " let f32_value = " << castToF32(components, "value") << ";\n"
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
     << "let mean = " << sumVector("sum", components) << " / f32(uniforms.hidden_size);\n"
     << "let inv_std_dev = inverseSqrt(" << sumVector("square_sum", components) << " / f32(uniforms.hidden_size) " << simpl1 << "+ uniforms.epsilon);\n"
     << "for (var i: u32 = 0; i < stride; i++) {\n"
     << " output[offset + i] = (output[offset + i] " << simpl2 << ") * element_t(inv_std_dev) * gamma[offset1d + i]" << beta << ";\n"
     << "};\n";

  shader.SetMainFunctionBody(ss.str());
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

  size_t data_size = x_shape.Size();
  if (data_size == 0) {
    return Status::OK();
  }

  const bool is_fp16 = x->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  const int hidden_size = x_shape[x_shape.NumDimensions() - 1];
  const int components = getMaxComponents(hidden_size);

  SkipLayerNormProgram program{beta != nullptr, bias != nullptr, epsilon_, hidden_size, is_fp16, simplified};
  program
      .CacheHint(simplified)
      .AddInputs({{x, ProgramTensorMetadataDependency::Type, components}})
      .AddInputs({{skip, ProgramTensorMetadataDependency::Type, components}})
      .AddInputs({{gamma, ProgramTensorMetadataDependency::Type, components}})
      .AddOutputs({{output, ProgramTensorMetadataDependency::None, components}})
      .SetDispatchGroupSize(ceil(data_size / hidden_size))
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
