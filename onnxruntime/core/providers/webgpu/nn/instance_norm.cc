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
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& scale = shader.AddInput("scale", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& bias = shader.AddInput("bias", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  shader.AdditionalImplementation() << "var<workgroup> workgroup_shared_sum = array<x_value, WORKGROUP_SIZE>;\n";
  shader.AdditionalImplementation() << "var<workgroup> workgroup_shared_squared_sum = array<x_value, WORKGROUP_SIZE>;\n";
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "let batch = workgroup_index / uniforms.x_shape[1];\n"
                            << "let channel = workgroup_index % uniforms.x_shape[1];\n"
                            << "let hight = uniforms.x_shape[2];\n"
                            << " // initialize workgroup memory<< \n"
                            << "var sum = x_value_(0);\n"
                            << "var squared_sum = x_value_t(0);\n"
                            << "for (var h = local_idx; h < hight; h += workgroup_size) {<< \n"
                            << "  let indices = hight_indices_t(batch, channel, h);\n"
                            << "  let value =" << input.GetByIndices("indices") << ";\n"
                            << "  sum += value;\n"
                            << "  squared_sum += value * value;\n"
                            << "}\n"
                            << "workgroup_shared_sum[local_idx] = sum;\n"
                            << "workgroup_shared_squared_sum[local_idx] = squared_sum;\n"
                            << "workgroupBarrier();\n"
                            << "for (var currSize = workgroup_size >> 1; currSize > 0; currSize = currSize >> 1) {<< \n"
                            << "  if (local_idx < currSize) {<< \n"
                            << "    workgroup_shared_sum[local_idx] = workgroup_shared_sum[local_idx] + workgroup_shared_sum[local_idx + currSize];\n"
                            << "    workgroup_shared_squared_sum[local_idx] = workgroup_shared_squared_sum[local_idx] + workgroup_shared_squared_sum[local_idx + currSize];\n"
                            << "  }\n"
                            << "  workgroupBarrier();\n"
                            << "}\n"
                            << "if (local_idx == 0) {\n"
                            << "  let sum_final = " << SumVector("sum", components_) << " / f32(hight * " << components_ << ");\n"
                            << "  let squared_sum_final = " << SumVector("squared_sum", components_) << " / x_element_t(hight * " << components_ << ");\n"
                            << "  let inv_std_dev = inverseSqrt(squared_sum_final - sum_final * sum_final + x_element_t(uniforms.epsilon));\n"
                            << "  let channel_scale = inv_std_dev * " << scale.GetByOffset("channel") << ");\n"
                            << "  let channel_shift = " << bias.GetByOffset("channel") << " - sum_final * channel_scale;\n"
                            << "  " << output.SetByOffset("workgroup_index", "output_t(channel_scale, channel_shift)") << ";\n"
                            << "}\n"
                            << "}\n";
  return Status::OK();
}
template <bool is_nhwc>
Status InstanceNorm<is_nhwc>::ComputeChannelScaleAndShift(ComputeContext& context,
                                                          const Tensor* input, const Tensor* scale, const Tensor* bias, float epsilon, Tensor* output) const {
  const auto& input_shape = input->Shape();
  const auto batch_size = input_shape[0];
  const auto rank = input_shape.NumDimensions();
  const auto channels = input_shape[is_nhwc ? rank - 1 : 1];
  const auto spatial_size = input->Shape().SizeFromDimension(2);
  const auto components = GetMaxComponents(spatial_size);
  TensorShapeVector reduce_input_shape_vector = {batch_size, channels, spatial_size / components};
  TensorShapeVector output_shape_vector = {batch_size, channels, 2};
  TensorShapeVector reduced_output_shape_vector = {batch_size, channels, 2};
  TensorShape reduced_input_shape(reduce_input_shape_vector);
  TensorShape output_shape(output_shape_vector);
  TensorShape reduced_output_shape(reduced_output_shape_vector);
  *output = context.CreateGPUTensor(input->DataType(), output_shape);
  auto output_size = output_shape.Size() / 2;
  ComputeChannelScaleShiftProgram program = ComputeChannelScaleShiftProgram(components);
  program.CacheHint(components)
      .AddInputs({{input, ProgramTensorMetadataDependency::TypeAndRank, reduced_input_shape, components},
                  {scale, ProgramTensorMetadataDependency::TypeAndRank},
                  {bias, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank, reduced_output_shape, 2}})
      .AddUniformVariables({{static_cast<uint32_t>(output_size)}, {epsilon}})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  return context.RunProgram(program);
}
Status InstanceNormProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& channel_scale_shift = shader.AddInput("channel_scale_shift", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "let outputIndices = " << output.OffsetToIndices("global_idx")
                            << "let batch = outputIndices[0];\n"
                            << "let channel = outputIndices[1];\n"
                            << "let channel_scale_shift_indices = channel_scale_shift_indices_t(batch, channel);\n"
                            << "let channel_scale_shift = " << channel_scale_shift.GetByIndices("channel_scale_shift_indices") << ";\n"
                            << "let channel_scale = channel_scale_shift.x;\n"
                            << "let channel_shift = channel_scale_shift.y;\n"
                            << "let hight = uniforms.x_shape[2];\n"
                            << "let input_indices = input_indices_t(batch, channel, height);\n"
                            << "let value = " << input.GetByIndices("input_indices") << ";\n"
                            << "let output_value = value * output_t(channel_scale) + output_t(channel_shift);\n"
                            << output.SetByIndices("outputIndices", "output_value") << "\n";
  return Status::OK();
}

template <bool is_nhwc>
Status InstanceNorm<is_nhwc>::ComputeInternal(ComputeContext& context) const {
  const auto* input = context.Input<Tensor>(0);
  const auto* scale = context.Input<Tensor>(1);
  const auto* bias = context.Input<Tensor>(2);
  ORT_RETURN_IF_ERROR(InstanceNormHelper::ValidateInputs(input, scale, bias, is_nhwc));
  TensorShape input_shape = input->Shape();
  TensorShapeVector input_shape_vector = input->Shape().AsShapeVector();
  const auto rank = input_shape_vector.size();
  const auto channels = input_shape_vector[is_nhwc ? rank - 1 : 1];
  const auto spatial_size = is_nhwc ? input->Shape().SizeFromDimension(2) : input->Shape().SizeFromDimension(1) / channels;
  Tensor input_transpose;
  if (is_nhwc) {
    // Transpose input to NCHW format
    TensorShapeVector input_transpose_shape_vector(rank);
    input_transpose_shape_vector[0] = input_shape_vector[0];
    input_transpose_shape_vector[1] = input_shape_vector[rank - 1];
    InlinedVector<size_t> permute(rank);
    permute[0] = static_cast<size_t>(0);
    permute[1] = rank - 1;
    bool need_transpose = false;
    for (size_t i = 0; i < rank - 2; ++i) {
      need_transpose = need_transpose || input_shape_vector[i + 1] != 1;
      input_transpose_shape_vector[i] = input_shape_vector[i + 1];
      permute[i + 2] = i + 1;
    }
    auto input_transpose_size = static_cast<uint32_t>(input_shape.Size());
    TensorShape input_transpose_shape(input_transpose_shape_vector);
    if (need_transpose) {
      input_transpose = context.CreateGPUTensor(input->DataType(), input_transpose_shape);
      TransposeProgram transpose_program{permute, false};
      transpose_program
          .CacheHint(absl::StrJoin(permute, "-"))
          .AddInput({input, ProgramTensorMetadataDependency::TypeAndRank, input_shape, 1})
          .AddOutput({&input_transpose, ProgramTensorMetadataDependency::TypeAndRank})
          .AddUniformVariable({input_transpose_size})
          .SetDispatchGroupSize((input_transpose_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
      ORT_RETURN_IF_ERROR(context.RunProgram(transpose_program));
      input = &input_transpose;
    } else {
      input_shape = input_transpose_shape;
    }
  }
  Tensor channel_scale_shift;
  ORT_RETURN_IF_ERROR(ComputeChannelScaleAndShift(context, input, scale, bias, epsilon_, &channel_scale_shift));
  const auto output_shape(input_shape);
  Tensor* output = context.Output(0, output_shape);
  const auto components = GetMaxComponents(spatial_size);
  TensorShapeVector output_shape_vector(input_shape_vector);
  output_shape_vector[rank - 1] /= components;
  input_shape_vector[rank - 1] /= components;
  TensorShape reduced_input_shape(input_shape_vector);
  TensorShape reduced_output_shape(output_shape_vector);
  InstanceNormProgram program;
  TensorShape channel_scale_shift_shape = channel_scale_shift.Shape();
  TensorShapeVector channel_scale_shift_shape_vector = channel_scale_shift_shape.AsShapeVector();
  TensorShapeVector reduced_channel_scale_shift_shape_vector = channel_scale_shift_shape_vector;
  reduced_channel_scale_shift_shape_vector[rank - 1] /= 2;
  TensorShape reduced_channel_scale_shift_shape(reduced_channel_scale_shift_shape_vector);
  auto output_size = output_shape.Size() / components;
  program
      .CacheHint(epsilon_, is_nhwc, components)
      .AddInputs({{input, ProgramTensorMetadataDependency::TypeAndRank, reduced_input_shape, components},
                  {&channel_scale_shift, ProgramTensorMetadataDependency::TypeAndRank, reduced_channel_scale_shift_shape, 2}})
      .AddOutput({output, ProgramTensorMetadataDependency::TypeAndRank, reduced_output_shape, components})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(output_size)}});
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
