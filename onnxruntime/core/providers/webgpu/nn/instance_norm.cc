// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/nn/instance_norm.h"
#include "core/providers/cpu/nn/instance_norm_helper.h"
#include "core/providers/webgpu/tensor/transpose.h"
#include "core/common/inlined_containers.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"
namespace onnxruntime {
namespace webgpu {

Status ComputeChannelScaleShiftProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& scale = shader.AddInput("scale", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& bias = shader.AddInput("bias", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  shader.AdditionalImplementation() << "alias f32_val_t = " << (components_ == 4 ? "vec4<f32>" : (components_ == 2 ? "vec2<f32>" : "f32")) << ";\n"
                                    << "var<workgroup> workgroup_shared_sum : array<f32_val_t, " << workgroup_size_ << ">;\n"
                                    << "var<workgroup> workgroup_shared_squared_sum : array<f32_val_t, " << workgroup_size_ << ">;\n"
                                    << "const workgroup_size = " << workgroup_size_ << ";\n";

  shader.MainFunctionBody() << "  let batch = workgroup_idx / uniforms.x_shape[1];\n"
                            << "  let channel = workgroup_idx % uniforms.x_shape[1];\n"
                            << "  let height = uniforms.x_shape[2];\n"
                            << "   // initialize workgroup memory<< \n"
                            << "  var sum = f32_val_t(0);\n"
                            << "  var squared_sum = f32_val_t(0);\n"
                            << "  for (var h = local_idx; h < height; h += workgroup_size) {\n"
                            << "    let indices = x_indices_t(batch, channel, h);\n"
                            << "    let value = f32_val_t(" << input.GetByIndices("indices") << ");\n"
                            << "    sum += value;\n"
                            << "    squared_sum += value * value;\n"
                            << "  }\n"
                            << "  workgroup_shared_sum[local_idx] = sum;\n"
                            << "  workgroup_shared_squared_sum[local_idx] = squared_sum;\n"
                            << "  workgroupBarrier();\n"
                            << "  for (var currSize = workgroup_size >> 1; currSize > 0; currSize = currSize >> 1) {\n"
                            << "    if (local_idx < u32(currSize)) {\n"
                            << "      workgroup_shared_sum[local_idx] = workgroup_shared_sum[local_idx] + workgroup_shared_sum[local_idx + u32(currSize)];\n"
                            << "      workgroup_shared_squared_sum[local_idx] = workgroup_shared_squared_sum[local_idx] + workgroup_shared_squared_sum[local_idx + u32(currSize)];\n"
                            << "    }\n"
                            << "    workgroupBarrier();\n"
                            << "  }\n"
                            << "  if (local_idx == 0) {\n"
                            << "    let sum_final = " << SumVector("workgroup_shared_sum[0]", components_) << " / f32(height * " << components_ << ");\n"
                            << "    let squared_sum_final = " << SumVector("workgroup_shared_squared_sum[0]", components_) << " / f32(height * " << components_ << ");\n"
                            << "    let inv_std_dev = inverseSqrt(squared_sum_final - sum_final * sum_final + f32(" << std::to_string(epsilon_) << "));\n"
                            << "    let channel_scale = inv_std_dev * f32(" << scale.GetByOffset("channel") << ");\n"
                            << "    let channel_shift = f32(" << bias.GetByOffset("channel") << ") - sum_final * channel_scale;\n"
                            << "    " << output.SetByOffset("workgroup_idx", "output_value_t(output_element_t(channel_scale), output_element_t(channel_shift))") << ";\n"
                            << "  }\n";
  return Status::OK();
}

// This function expects channels first. The spacial dimensions are expected to be in the last dimensions.
Status ComputeChannelScaleAndShift(ComputeContext& context, const Tensor* input, const Tensor* scale, const Tensor* bias, float epsilon, Tensor* output) {
  const auto& input_shape = input->Shape();
  const auto batch_size = input_shape[0];
  const auto channels = input_shape[1];
  const auto spatial_size = input->Shape().SizeFromDimension(2);
  const auto components = GetMaxComponents(spatial_size);
  auto units_of_work = batch_size * channels;
  auto workgroup_size = units_of_work == 1 ? static_cast<int>(WORKGROUP_SIZE) : static_cast<int>(256);
  TensorShapeVector reduce_input_shape_vector = {batch_size, channels, spatial_size / components};
  TensorShapeVector output_shape_vector = {batch_size, channels, 2};
  TensorShapeVector reduced_output_shape_vector = {batch_size, channels, 1};
  TensorShape reduced_input_shape(reduce_input_shape_vector);
  TensorShape output_shape(output_shape_vector);
  TensorShape reduced_output_shape(reduced_output_shape_vector);
  *output = context.CreateGPUTensor(input->DataType(), output_shape);
  ComputeChannelScaleShiftProgram program = ComputeChannelScaleShiftProgram(components, epsilon, workgroup_size);
  program.CacheHint(components, units_of_work)
      .AddInputs({{input, ProgramTensorMetadataDependency::TypeAndRank, reduced_input_shape, components},
                  {scale, ProgramTensorMetadataDependency::TypeAndRank},
                  {bias, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank, reduced_output_shape, 2}})
      .SetDispatchGroupSize(static_cast<uint32_t>(units_of_work))
      .SetWorkgroupSize(workgroup_size);
  return context.RunProgram(program);
}

Status InstanceNormProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& channel_scale_shift = shader.AddInput("channel_scale_shift", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "let outputIndices = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "let batch = outputIndices[0];\n"
                            << "let channel = outputIndices[1];\n"
                            << "let channel_scale_shift_indices = channel_scale_shift_indices_t(batch, channel, 0);\n"
                            << "let channel_scale_shift = " << channel_scale_shift.GetByIndices("channel_scale_shift_indices") << ";\n"
                            << "let input_value = " << input.GetByOffset("global_idx") << ";\n"
                            << "let output_value = input_value * output_value_t(channel_scale_shift.x) + output_value_t(channel_scale_shift.y);\n"
                            << output.SetByOffset("global_idx", "output_value") << ";\n";
  return Status::OK();
}

Status InstanceNormProgramNHWC::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& channel_scale_shift = shader.AddInput("channel_scale_shift", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "let current_image_number = global_idx / (uniforms.C * uniforms.H);\n"
                            << "let current_channel_number = global_idx % uniforms.C;\n"
                            << "let scale_offset = (current_image_number * uniforms.C + current_channel_number);\n"
                            << "var scale : input_value_t;\n"
                            << "var shift : input_value_t;\n"
                            << "let input_value = " << input.GetByOffset("global_idx") << ";\n";
  if (components_ > 1) {
    shader.MainFunctionBody() << "for (var i : u32 = 0; i < uniforms.components; i = i + 1) {\n"
                              << "  let scale_sift = " << channel_scale_shift.GetByOffset("uniforms.components * scale_offset + i") << ";\n"
                              << "  scale[i] = input_element_t(scale_sift.x);\n"
                              << "  shift[i] = input_element_t(scale_sift.y);\n"
                              << "}\n";
  } else {
    shader.MainFunctionBody() << "let scale_shift = " << channel_scale_shift.GetByOffset("scale_offset") << ";\n"
                              << "scale = scale_shift.x;\n"
                              << "shift = scale_shift.y;\n";
  }
  shader.MainFunctionBody() << "let output_value = fma(input_value, scale, shift);\n";
  shader.MainFunctionBody() << output.SetByOffset("global_idx", "output_value") << ";\n";

  return Status::OK();
}

template <>
Status InstanceNorm<true>::ComputeInternal(ComputeContext& context) const {
  const auto* input = context.Input<Tensor>(0);
  const auto* scale = context.Input<Tensor>(1);
  const auto* bias = context.Input<Tensor>(2);
  ORT_RETURN_IF_ERROR(InstanceNormHelper::ValidateInputs(input, scale, bias, true));
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

  Tensor channel_scale_shift;
  ORT_RETURN_IF_ERROR(ComputeChannelScaleAndShift(context, &input_transpose, scale, bias, epsilon_, &channel_scale_shift));
  TensorShape output_shape(input_shape_vector);
  Tensor* output = context.Output(0, output_shape);
  const auto components = GetMaxComponents(channels);
  auto output_size = (output_shape.Size() + components - 1) / components;
  InstanceNormProgramNHWC program(components);
  TensorShapeVector channel_scale_shift_shape_vector = {batch_size, channels, 1};
  TensorShape reduced_channel_scale_shift_shape(channel_scale_shift_shape_vector);
  program.CacheHint(components)
      .AddInputs({{input, ProgramTensorMetadataDependency::TypeAndRank, components},
                  {&channel_scale_shift, ProgramTensorMetadataDependency::TypeAndRank, reduced_channel_scale_shift_shape, 2}})
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
  TensorShape input_shape = input->Shape();
  TensorShapeVector input_shape_vector = input->Shape().AsShapeVector();
  const auto batch_size = input_shape_vector[0];
  const auto channels = input_shape_vector[1];
  const auto spatial_size = input->Shape().SizeFromDimension(2);
  Tensor channel_scale_shift;
  ORT_RETURN_IF_ERROR(ComputeChannelScaleAndShift(context, input, scale, bias, epsilon_, &channel_scale_shift));
  TensorShape output_shape(input_shape_vector);
  Tensor* output = context.Output(0, output_shape);
  const auto components = GetMaxComponents(spatial_size);
  TensorShapeVector modified_input_shape_vector = {batch_size, channels, spatial_size / components};
  TensorShape modified_input_shape(modified_input_shape_vector);
  TensorShape modified_output_shape(modified_input_shape_vector);
  auto output_size = modified_output_shape.Size();
  TensorShapeVector channel_scale_shift_shape_vector = {batch_size, channels, 1};
  TensorShape reduced_channel_scale_shift_shape(channel_scale_shift_shape_vector);
  InstanceNormProgram program;
  program
      .AddInputs({{input, ProgramTensorMetadataDependency::TypeAndRank, modified_input_shape, components},
                  {&channel_scale_shift, ProgramTensorMetadataDependency::TypeAndRank, reduced_channel_scale_shift_shape, 2}})
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
