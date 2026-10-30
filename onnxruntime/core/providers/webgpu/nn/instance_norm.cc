// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/nn/instance_norm.h"
#include "core/providers/webgpu/nn/normalization_utils.h"

#include "core/providers/cpu/nn/instance_norm_helper.h"
#include "core/providers/webgpu/tensor/transpose.h"
#include "core/common/inlined_containers.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"
namespace onnxruntime {
namespace webgpu {

Status ComputeChannelStatsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias);
  shader.AdditionalImplementation()
      << "alias f32_val_t = " << (components_ == 4 ? "vec4<f32>" : components_ == 2 ? "vec2<f32>"
                                                                                    : "f32")
      << ";\n"
      << "var<workgroup> partial : array<f32_val_t, workgroup_size_x>;\n";
  auto& body = shader.MainFunctionBody();
  body << "if (workgroup_idx >= uniforms.x_shape[0] * uniforms.x_shape[1]) { return; }\n"
       << "let batch = workgroup_idx / uniforms.x_shape[1];\n"
       << "let channel = workgroup_idx % uniforms.x_shape[1];\n"
       << "let height = uniforms.x_shape[2];\n"
       << "let count = f32(height * " << components_ << "u);\n";
  auto reduce = [&]() {
    body << "workgroupBarrier();\n"
         << "for (var width = workgroup_size_x / 2u; width > 0u; width /= 2u) {\n"
         << "  if (local_idx < width) { partial[local_idx] += partial[local_idx + width]; }\n"
         << "  workgroupBarrier();\n}\n";
  };
  AppendStableNormalizationMean(shader, components_, "height", "count", [&](const std::string& i) {
    return input.GetByIndices("x_indices_t(batch, channel, " + i + ")");
  });
  body << "let mean = norm_mean; let shift = norm_shift;\n"
       << "var squared = f32_val_t(0);\n"
       << "for (var h = local_idx; h < height; h += workgroup_size_x) {\n"
       << "  let centered = norm_center(f32_val_t(" << input.GetByIndices("x_indices_t(batch, channel, h)") << "), f32_val_t(shift), f32_val_t(mean));\n"
       << "  squared += centered * centered;\n}\n"
       << "partial[local_idx] = squared;\n";
  reduce();
  body << "if (local_idx == 0u) {\n"
       << "  let variance = " << SumVector("partial[0]", components_) << " / count;\n"
       << "  let inv_std_dev = inverseSqrt(variance + uniforms.epsilon);\n"
       << "  " << output.SetByOffset("workgroup_idx", "output_value_t(shift, mean, inv_std_dev, 0)") << "\n}\n";
  return Status::OK();
}

// This function expects channels first. The spacial dimensions are expected to be in the last dimensions.
Status ComputeChannelStats(ComputeContext& context, const Tensor* input, float epsilon, Tensor* output) {
  const auto& input_shape = input->Shape();
  const auto batch_size = input_shape[0];
  const auto channels = input_shape[1];
  const auto spatial_size = input->Shape().SizeFromDimension(2);
  const auto components = GetMaxComponents(spatial_size);
  auto units_of_work = batch_size * channels;
  auto workgroup_size = units_of_work == 1 ? static_cast<int>(WORKGROUP_SIZE) : static_cast<int>(256);
  TensorShapeVector reduce_input_shape_vector = {batch_size, channels, spatial_size / components};
  TensorShapeVector output_shape_vector = {batch_size, channels, 4};
  TensorShapeVector reduced_output_shape_vector = {batch_size, channels, 1};
  TensorShape reduced_input_shape(reduce_input_shape_vector);
  TensorShape output_shape(output_shape_vector);
  TensorShape reduced_output_shape(reduced_output_shape_vector);
  *output = context.CreateGPUTensor(DataTypeImpl::GetType<float>(), output_shape);
  ComputeChannelStatsProgram program = ComputeChannelStatsProgram(components);
  program.CacheHint(components)
      .AddInput({input, ProgramTensorMetadataDependency::TypeAndRank, reduced_input_shape, components})
      .AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank, reduced_output_shape, 4}})
      .SetDispatchGroupSize(static_cast<uint32_t>(units_of_work))
      .SetWorkgroupSize(workgroup_size)
      .AddUniformVariable(epsilon);
  return context.RunProgram(program);
}

Status InstanceNormProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& stats = shader.AddInput("stats", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& scale = shader.AddInput("scale", ShaderUsage::UseUniform);
  const auto& bias = shader.AddInput("bias", ShaderUsage::UseUniform);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  AppendNormalizationCenter(shader, input.NumComponents());
  shader.AdditionalImplementation() << "alias acc_value_t = "
                                    << (input.NumComponents() == 4 ? "vec4<f32>" : input.NumComponents() == 2 ? "vec2<f32>"
                                                                                                              : "f32")
                                    << ";\n";
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "let index = " << input.OffsetToIndices("global_idx") << ";\n"
                            << "let state = " << stats.GetByIndices("stats_indices_t(index[0], index[1], 0)") << ";\n"
                            << "let centered = norm_center(acc_value_t(" << input.GetByOffset("global_idx") << "), acc_value_t(state.x), acc_value_t(state.y));\n"
                            << output.SetByOffset("global_idx", "output_value_t(centered * state.z * f32(" + scale.GetByOffset("index[1]") + ") + f32(" + bias.GetByOffset("index[1]") + "))") << "\n";
  return Status::OK();
}

Status InstanceNormProgramNHWC::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform);
  const auto& stats = shader.AddInput("stats", ShaderUsage::UseUniform);
  const auto& scale = shader.AddInput("scale", ShaderUsage::UseUniform);
  const auto& bias = shader.AddInput("bias", ShaderUsage::UseUniform);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  AppendNormalizationCenter(shader, input.NumComponents());
  shader.AdditionalImplementation() << "alias acc_value_t = "
                                    << (components_ == 4 ? "vec4<f32>" : components_ == 2 ? "vec2<f32>"
                                                                                          : "f32")
                                    << ";\n";
  auto& body = shader.MainFunctionBody();
  body << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
       << "let batch = global_idx / (uniforms.C * uniforms.H);\n"
       << "let channel = global_idx % uniforms.C;\n"
       << "let base = (batch * uniforms.C + channel) * uniforms.components;\n"
       << "var shift: acc_value_t; var mean: acc_value_t; var inv_std: acc_value_t;\n";
  if (components_ > 1) {
    body << "for (var i = 0u; i < uniforms.components; i++) {\n"
         << "  let state = " << stats.GetByOffset("base + i") << ";\n"
         << "  shift[i] = state.x; mean[i] = state.y; inv_std[i] = state.z;\n}\n";
  } else {
    body << "let state = " << stats.GetByOffset("base") << ";\n"
         << "shift = state.x; mean = state.y; inv_std = state.z;\n";
  }
  body << "let centered = norm_center(acc_value_t(" << input.GetByOffset("global_idx") << "), shift, mean);\n"
       << output.SetByOffset("global_idx", "output_value_t(centered * inv_std * acc_value_t(" + scale.GetByOffset("channel") + ") + acc_value_t(" + bias.GetByOffset("channel") + "))") << "\n";
  return Status::OK();
}

template <>
Status InstanceNorm<true>::ComputeInternal(ComputeContext& context) const {
  const auto* input = context.Input<Tensor>(0);
  const auto* scale = context.Input<Tensor>(1);
  const auto* bias = context.Input<Tensor>(2);
  ORT_RETURN_IF_ERROR(InstanceNormHelper::ValidateInputs(input, scale, bias, true));
  if (input->Shape().Size() == 0) {
    context.Output(0, input->Shape());
    return Status::OK();
  }
  TensorShape input_shape = input->Shape();
  TensorShapeVector input_shape_vector = input->Shape().AsShapeVector();
  const auto rank = input_shape_vector.size();
  const auto batch_size = input_shape_vector[0];
  const auto channels = input_shape_vector[rank - 1];
  const auto spatial_size = input->Shape().SizeFromDimension(1) / channels;
  Tensor input_transpose;
  // Transpose input to NCHW format
  TensorShapeVector input_transpose_shape_vector(rank);
  input_transpose_shape_vector[0] = input_shape_vector[0];
  input_transpose_shape_vector[1] = input_shape_vector[rank - 1];
  InlinedVector<size_t> permute(rank);
  permute[0] = static_cast<size_t>(0);
  permute[1] = rank - 1;
  for (size_t i = 0; i < rank - 2; ++i) {
    input_transpose_shape_vector[i + 2] = input_shape_vector[i + 1];
    permute[i + 2] = i + 1;
  }
  auto input_transpose_size = static_cast<uint32_t>(input_shape.Size());
  TensorShape input_transpose_shape(input_transpose_shape_vector);
  input_transpose = context.CreateGPUTensor(input->DataType(), input_transpose_shape);
  TransposeProgram transpose_program{permute, false};
  transpose_program
      .CacheHint(absl::StrJoin(permute, "-"))
      .AddInput({input, ProgramTensorMetadataDependency::TypeAndRank, input_shape, 1})
      .AddOutput({&input_transpose, ProgramTensorMetadataDependency::TypeAndRank})
      .AddUniformVariable({input_transpose_size})
      .SetDispatchGroupSize((input_transpose_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  ORT_RETURN_IF_ERROR(context.RunProgram(transpose_program));

  Tensor channel_stats;
  ORT_RETURN_IF_ERROR(ComputeChannelStats(context, &input_transpose, epsilon_, &channel_stats));
  TensorShape output_shape(input_shape_vector);
  Tensor* output = context.Output(0, output_shape);
  const auto components = GetMaxComponents(channels);
  auto output_size = (output_shape.Size() + components - 1) / components;
  InstanceNormProgramNHWC program(components);
  TensorShapeVector channel_stats_shape_vector = {batch_size, channels, 1};
  TensorShape reduced_channel_stats_shape(channel_stats_shape_vector);
  program.CacheHint(components)
      .AddInputs({{input, ProgramTensorMetadataDependency::TypeAndRank, components},
                  {&channel_stats, ProgramTensorMetadataDependency::TypeAndRank, reduced_channel_stats_shape, 4},
                  {scale, ProgramTensorMetadataDependency::TypeAndRank, components},
                  {bias, ProgramTensorMetadataDependency::TypeAndRank, components}})
      .AddOutput({output, ProgramTensorMetadataDependency::TypeAndRank, components})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({static_cast<uint32_t>(output_size), static_cast<uint32_t>(components), static_cast<uint32_t>(channels / components), static_cast<uint32_t>(spatial_size)});
  return context.RunProgram(program);
}

template <>
Status InstanceNorm<false>::ComputeInternal(ComputeContext& context) const {
  const auto* input = context.Input<Tensor>(0);
  const auto* scale = context.Input<Tensor>(1);
  const auto* bias = context.Input<Tensor>(2);
  ORT_RETURN_IF_ERROR(InstanceNormHelper::ValidateInputs(input, scale, bias, false));
  if (input->Shape().Size() == 0) {
    context.Output(0, input->Shape());
    return Status::OK();
  }
  TensorShape input_shape = input->Shape();
  TensorShapeVector input_shape_vector = input->Shape().AsShapeVector();
  const auto batch_size = input_shape_vector[0];
  const auto channels = input_shape_vector[1];
  const auto spatial_size = input->Shape().SizeFromDimension(2);
  Tensor channel_stats;
  ORT_RETURN_IF_ERROR(ComputeChannelStats(context, input, epsilon_, &channel_stats));
  TensorShape output_shape(input_shape_vector);
  Tensor* output = context.Output(0, output_shape);
  const auto components = GetMaxComponents(spatial_size);
  TensorShapeVector modified_input_shape_vector = {batch_size, channels, spatial_size / components};
  TensorShape modified_input_shape(modified_input_shape_vector);
  TensorShape modified_output_shape(modified_input_shape_vector);
  auto output_size = modified_output_shape.Size();
  TensorShapeVector channel_stats_shape_vector = {batch_size, channels, 1};
  TensorShape reduced_channel_stats_shape(channel_stats_shape_vector);
  InstanceNormProgram program;
  program
      .AddInputs({{input, ProgramTensorMetadataDependency::TypeAndRank, modified_input_shape, components},
                  {&channel_stats, ProgramTensorMetadataDependency::TypeAndRank, reduced_channel_stats_shape, 4},
                  {scale, ProgramTensorMetadataDependency::TypeAndRank},
                  {bias, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutput({output, ProgramTensorMetadataDependency::TypeAndRank, modified_output_shape, components})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({static_cast<uint32_t>(output_size)});
  return context.RunProgram(program);
}

#define WEBGPU_INSTANCE_NORM_VERSIONED_KERNEL(start, end, domain, is_nhwc) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                       \
      InstanceNormalization,                                               \
      domain,                                                              \
      start,                                                               \
      end,                                                                 \
      kWebGpuExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                        \
          .TypeConstraint("T", WebGpuSupportedFloatTypes()),               \
      InstanceNorm<is_nhwc>);

#define WEBGPU_INSTANCE_NORM_KERNEL(version, domain, is_nhwc) \
  ONNX_OPERATOR_KERNEL_EX(                                    \
      InstanceNormalization,                                  \
      domain,                                                 \
      version,                                                \
      kWebGpuExecutionProvider,                               \
      (*KernelDefBuilder::Create())                           \
          .TypeConstraint("T", WebGpuSupportedFloatTypes()),  \
      InstanceNorm<is_nhwc>);

WEBGPU_INSTANCE_NORM_VERSIONED_KERNEL(1, 5, kOnnxDomain, false)
WEBGPU_INSTANCE_NORM_VERSIONED_KERNEL(6, 21, kOnnxDomain, false)
WEBGPU_INSTANCE_NORM_KERNEL(22, kOnnxDomain, false)

WEBGPU_INSTANCE_NORM_VERSIONED_KERNEL(1, 5, kMSInternalNHWCDomain, true)
WEBGPU_INSTANCE_NORM_VERSIONED_KERNEL(6, 21, kMSInternalNHWCDomain, true)
WEBGPU_INSTANCE_NORM_KERNEL(22, kMSInternalNHWCDomain, true)

}  // namespace webgpu
}  // namespace onnxruntime
