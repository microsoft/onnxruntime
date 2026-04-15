// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/compress.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_COMPRESS_VERSIONED_KERNEL(start, end)                             \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                             \
      Compress,                                                                  \
      kOnnxDomain,                                                               \
      start,                                                                     \
      end,                                                                       \
      kWebGpuExecutionProvider,                                                  \
      (*KernelDefBuilder::Create())                                              \
          .TypeConstraint("T", WebGpuSupportedNumberAndBoolTypes())              \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>())             \
          .InputMemoryType(OrtMemTypeCPU, 1),                                    \
      Compress);

#define WEBGPU_COMPRESS_KERNEL(version)                                          \
  ONNX_OPERATOR_KERNEL_EX(                                                       \
      Compress,                                                                  \
      kOnnxDomain,                                                               \
      version,                                                                   \
      kWebGpuExecutionProvider,                                                  \
      (*KernelDefBuilder::Create())                                              \
          .TypeConstraint("T", WebGpuSupportedNumberAndBoolTypes())              \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>())             \
          .InputMemoryType(OrtMemTypeCPU, 1),                                    \
      Compress);

WEBGPU_COMPRESS_VERSIONED_KERNEL(9, 10)
WEBGPU_COMPRESS_KERNEL(11)

Status CompressProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform);
  const auto& indices = shader.AddInput("indices", ShaderUsage::UseUniform);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "  let axis_right_stride = uniforms.axis_right_stride;\n";

  if (has_axis_) {
    // With axis: output is indexed as (left, compressed_idx, right)
    // For each output element, find which axis slice it maps to via the indices buffer
    shader.MainFunctionBody()
        << "  let compressed_dim = uniforms.compressed_dim;\n"
        << "  let input_axis_dim = uniforms.input_axis_dim;\n"
        << "  let right_idx = global_idx % axis_right_stride;\n"
        << "  let compressed_idx = (global_idx / axis_right_stride) % compressed_dim;\n"
        << "  let left_idx = global_idx / (axis_right_stride * compressed_dim);\n"
        << "  let src_axis_idx = " << indices.GetByOffset("compressed_idx") << ";\n"
        << "  let src_offset = left_idx * input_axis_dim * axis_right_stride + src_axis_idx * axis_right_stride + right_idx;\n"
        << "  " << output.SetByOffset("global_idx", input.GetByOffset("src_offset"));
  } else {
    // Without axis (flattened): output[i] = input[indices[i]]
    shader.MainFunctionBody()
        << "  let src_offset = " << indices.GetByOffset("global_idx") << ";\n"
        << "  " << output.SetByOffset("global_idx", input.GetByOffset("src_offset"));
  }

  return Status::OK();
}

Status Compress::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  const auto* condition_tensor = context.Input(1);  // On CPU due to InputMemoryType

  const auto& input_shape = input_tensor->Shape();
  int64_t input_rank = input_shape.NumDimensions();

  const bool* condition_data = condition_tensor->Data<bool>();
  int64_t condition_length = condition_tensor->Shape().Size();

  int64_t axis = axis_;
  if (has_axis_) {
    if (axis < 0) {
      axis += input_rank;
    }
    ORT_ENFORCE(axis >= 0 && axis < input_rank, "Invalid axis value.");
  }

  // Determine the length along the compression dimension
  int64_t compress_length = has_axis_ ? input_shape[axis] : input_shape.Size();
  int64_t valid_condition_length = std::min(condition_length, compress_length);

  // Count true values and build index map (selected indices along axis)
  std::vector<uint32_t> selected_indices;
  selected_indices.reserve(valid_condition_length);
  for (int64_t i = 0; i < valid_condition_length; ++i) {
    if (condition_data[i]) {
      selected_indices.push_back(static_cast<uint32_t>(i));
    }
  }
  int64_t output_count = static_cast<int64_t>(selected_indices.size());

  // Compute output shape
  TensorShape output_shape;
  if (has_axis_) {
    auto dims = input_shape.GetDims();
    std::vector<int64_t> output_dims(dims.begin(), dims.end());
    output_dims[axis] = output_count;
    output_shape = TensorShape(output_dims);
  } else {
    output_shape = TensorShape({output_count});
  }

  auto* output_tensor = context.Output(0, output_shape);
  int64_t output_size = output_tensor->Shape().Size();

  if (output_size == 0) {
    return Status::OK();
  }

  // Compute axis_right_stride: product of dims after the axis
  uint32_t axis_right_stride = 1;
  if (has_axis_) {
    for (int64_t i = axis + 1; i < input_rank; ++i) {
      axis_right_stride *= static_cast<uint32_t>(input_shape[i]);
    }
  }

  // Create CPU tensor with selected indices, then copy to GPU
  TensorShape indices_shape({static_cast<int64_t>(selected_indices.size())});
  Tensor cpu_indices_tensor = context.CreateCPUTensor(DataTypeImpl::GetType<uint32_t>(), indices_shape);
  memcpy(cpu_indices_tensor.MutableDataRaw(), selected_indices.data(),
         selected_indices.size() * sizeof(uint32_t));

  Tensor gpu_indices_tensor = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), indices_shape);
  ORT_RETURN_IF_ERROR(context.CopyTensor(cpu_indices_tensor, gpu_indices_tensor));

  uint32_t data_size = onnxruntime::narrow<uint32_t>(output_size);
  uint32_t compressed_dim = onnxruntime::narrow<uint32_t>(output_count);
  uint32_t input_axis_dim = has_axis_ ? onnxruntime::narrow<uint32_t>(input_shape[axis]) : 0;

  CompressProgram program{has_axis_};
  program
      .AddInput({input_tensor, ProgramTensorMetadataDependency::TypeAndRank, {static_cast<uint32_t>(input_tensor->Shape().Size())}, 1})
      .AddInput({&gpu_indices_tensor, ProgramTensorMetadataDependency::Rank, indices_shape, 1})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::TypeAndRank, {data_size}, 1})
      .SetDispatchGroupSize((data_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .CacheHint(has_axis_ ? "axis" : "flat")
      .AddUniformVariables({{data_size}, {axis_right_stride}, {compressed_dim}, {input_axis_dim}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
