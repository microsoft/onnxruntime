
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/nn/layer_norm.h"

namespace onnxruntime {
namespace webgpu {

static uint32_t getMaxComponents(int size) {
  if (size % 4 == 0) {
    return 4;
  } else if (size % 2 == 0) {
    return 2;
  }
  return 1;
}

static int normalizeAxis(int axis, int tensorRank) {
  if (axis < -tensorRank && axis >= tensorRank) {
    ORT_THROW("invalid axis: ", axis);
  }
  return axis < 0 ? axis + tensorRank : axis;
}

static std::string fillVar(std::string dataType, int components, std::string value) {
  if (components == 1) {
    return dataType + "(" + value + ")";
  }
  return "vec" + std::to_string(components) + "<" + dataType + ">(" + value + ")";
}

static std::string castToF32(int components, std::string value) {
  if (components == 1) {
    return "f32(" + value + ")";
  }
  return "vec" + std::to_string(components) + "<f32>(" + value + ")";
};

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

Status LayerNormProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  shader.AddInput("scale", ShaderUsage::UseUniform);
  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseUniform);

  int components = x.NumComponents();
  std::string bias = (has_bias_) ? " + bias[j] " : "";
  std::string simpl1 = (simplified_) ? "" : "- mean * mean ";
  std::string simpl2 = (simplified_) ? "" : "- mean ";
  std::string fillvec = fillVar("f32", components, "0");
  std::string element_type = (isFP16_) ? "f16;\n" : "f32;\n";

  shader.AppendImplementation("alias element_t = " + element_type);

  std::stringstream ss;
  ss << "let offset = global_idx * uniforms.norm_size_vectorized;\n"
     << "var mean_vector = " << fillvec << ";\n"
     << "var mean_square_vector = " << fillvec << ";\n"
     << "for (var h: u32 = 0u; h < uniforms.norm_size_vectorized; h++) {\n"
     << "   let value = " << castToF32(components, "x[h + offset]") << ";\n"
     << "   mean_vector += value;\n"
     << "   mean_square_vector += value * value;\n"
     << "}\n"
     << "let mean = " << sumVector("mean_vector", components) << " / f32(uniforms.norm_size);\n"
     << "let inv_std_dev = inverseSqrt(" << sumVector("mean_square_vector", components) << " / f32(uniforms.norm_size) " << simpl1 << "+ uniforms.epsilon);\n"
     << "for (var j: u32 = 0; j < uniforms.norm_size_vectorized; j++) {\n"
     << "   let f32input = " << castToF32(components, "x[j + offset]") << ";\n"
     << "   let f32scale = " << castToF32(components, "scale[j]") << ";\n"
     << "   output[j + offset] =  x_value_t((f32input " << simpl2 << ") * inv_std_dev * f32scale)" << bias << ";\n"
     << "}\n";

  shader.SetMainFunctionBody(shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.norm_count"), ss.str());
  return Status::OK();
}

template <bool simplified>
Status LayerNorm<simplified>::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* x = context.Input(0);
  const auto* scale = context.Input(1);
  const auto* bias = context.Input(2);

  const auto x_shape = x->Shape();

  auto* output = context.Output(0, x_shape);

  size_t data_size = x_shape.Size();
  if (data_size == 0) {
    return Status::OK();
  }

  const bool is_fp16 = x->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;

  const int axis = normalizeAxis(axis_, x_shape.NumDimensions());
  const int norm_count = x_shape.SizeToDimension(axis);
  const int norm_size = x_shape.SizeFromDimension(axis);
  const int components = getMaxComponents(norm_size);
  const int norm_size_vectorized = (norm_size + components - 1) / components;

  const auto scale_size = scale->Shape().Size();
  const auto bias_size = (bias) ? bias->Shape().Size() : 0;
  if (scale_size != norm_size || (bias && bias_size != norm_size)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Size of X.shape()[axis:] == ", norm_size,
                           ". Size of scale and bias (if provided) must match this. Got scale size of ",
                           scale_size, " and bias size of ", bias_size);
  }

  LayerNormProgram program{axis_, epsilon_, stash_type_, bias != nullptr, data_size, is_fp16, simplified};

  program
      .CacheHint(simplified)
      .AddInputs({{x, ProgramTensorMetadataDependency::Type, components}})
      .AddInputs({{scale, ProgramTensorMetadataDependency::Type, components}})
      .AddOutputs({{output, ProgramTensorMetadataDependency::None, components}})
      .SetDispatchGroupSize((norm_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
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

  if (bias != nullptr) {
    program.AddInput({bias, ProgramTensorMetadataDependency::Type, components});
  }
  return context.RunProgram(program);
}


ONNX_OPERATOR_KERNEL_EX(
    LayerNormalization,
    kOnnxDomain,
    17,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
      .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    LayerNorm<false>);

ONNX_OPERATOR_KERNEL_EX(
    SimplifiedLayerNormalization,
    kOnnxDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()),
    LayerNorm<true>);

}  // namespace webgpu
}  // namespace onnxruntime
