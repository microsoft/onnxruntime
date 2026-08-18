// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/split.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

namespace {

// Helper function to calculate the output index based on the input index and the sizes of the splits.
void CalculateOutputIndex(OStringStream& os, size_t output_count) {
  os << "fn calculate_output_index(index: u32) -> u32 {\n"
     << "  for (var i: u32 = 0u; i < " << output_count << "u; i += 1u ) {\n"
     << "    if (index < " << GetElementAt("uniforms.sizes_in_split_axis", "i", output_count) << ") {\n"
     << "      return i;\n"
     << "    }\n"
     << "  }\n"
     << "  return " << output_count << "u;\n"
     << "}\n";
}

// Helper function to write the buffer data for each output.
void WriteBufferData(OStringStream& os, const ShaderVariableHelper& input,
                     gsl::span<const ShaderVariableHelper*> outputs) {
  os << "fn write_buffer_data(output_number: u32, global_idx: u32,  indices: output_0_indices_t) {\n";
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto buffer_write = outputs[i]->SetByIndices("indices", input.GetByOffset("global_idx"));
    if (outputs.size() == 1) {
      os << buffer_write << "\n";
    } else if (i == 0) {
      os << "  if (output_number == 0u) { " << buffer_write << " }\n";
    } else if (i == outputs.size() - 1) {
      os << "  else { " << buffer_write << " }\n";
    } else {
      os << "  else if (output_number == " << i << "u) { " << buffer_write << " }\n";
    }
  }
  os << "}\n";
}

}  // namespace

Status SplitProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);

  size_t output_count = Outputs().size();
  std::vector<const ShaderVariableHelper*> outputs;
  outputs.reserve(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    outputs.push_back(
        &shader.AddOutput("output_" + std::to_string(i), ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias));
  }

  // Add implementation of fn calculate_output_index.
  CalculateOutputIndex(shader.AdditionalImplementation(), output_count);
  // Add implementation of fn write_buffer_data.
  WriteBufferData(shader.AdditionalImplementation(), input, outputs);

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.input_size")
                            << "  var indices = " << input.OffsetToIndices("global_idx") << ";\n"
                            << "  var index = " << input.IndicesGet("indices", axis_) << ";\n"
                            << "  let output_number = calculate_output_index(index);\n"
                            << "  if (output_number != 0u) {\n"
                            << "    index -= " << GetElementAt("uniforms.sizes_in_split_axis", "output_number - 1u", output_count) << ";\n"
                            << "    " << input.IndicesSet("indices", axis_, "index") << "\n"
                            << "  }\n"
                            << "  write_buffer_data(output_number, global_idx, indices);\n";

  return Status::OK();
}

Status SplitContiguousVec4Program::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  std::vector<const ShaderVariableHelper*> outputs;
  outputs.reserve(segment_vector_sizes_.size());
  for (size_t i = 0; i < segment_vector_sizes_.size(); ++i) {
    outputs.push_back(
        &shader.AddOutput("output_" + std::to_string(i), ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias));
  }

  uint32_t total_vectors = 0;
  for (uint32_t segment_size : segment_vector_sizes_) {
    total_vectors += segment_size;
  }

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.input_size")
                            << "  let outer_index = global_idx / " << total_vectors << "u;\n"
                            << "  let within_outer = global_idx % " << total_vectors << "u;\n";

  uint32_t segment_start = 0;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const uint32_t segment_end = segment_start + segment_vector_sizes_[i];
    const std::string output_offset =
        "outer_index * " + std::to_string(segment_vector_sizes_[i]) +
        "u + within_outer - " + std::to_string(segment_start) + "u";
    shader.MainFunctionBody() << "  if (within_outer < " << segment_end << "u) {\n"
                              << "    " << outputs[i]->SetByOffset(output_offset, input.GetByOffset("global_idx")) << "\n"
                              << "    return;\n"
                              << "  }\n";
    segment_start = segment_end;
  }

  return Status::OK();
}

Status Split::ComputeInternal(ComputeContext& context) const {
  const Tensor* input = context.Input<Tensor>(0);
  auto& input_shape = input->Shape();
  auto num_outputs = context.OutputCount();

  int64_t axis = axis_;
  std::vector<int64_t> split_sizes;

  split_sizes.assign(split_sizes_.begin(), split_sizes_.end());
  // Compute split_sizes from the 'split' input tensor.
  if (split_sizes_.empty() && context.InputCount() > 1) {
    const Tensor* split_tensor = context.Input<Tensor>(1);
    // Check if split_tensor is valid.
    if (split_tensor != nullptr) {
      ORT_ENFORCE(split_tensor->Shape().NumDimensions() == 1, "The split tensor must be a vector tensor.");
      // Get split_sizes from the input tensor.
      auto nDims = static_cast<size_t>(split_tensor->Shape()[0]);
      const auto* data = split_tensor->Data<int64_t>();
      split_sizes.assign(data, data + nDims);
    }
  }

  // The variables below are not actually used in the current implementation.
  int before_dims = 0;
  int after_dims_including_split_axis = 0;
  int after_dims_excluding_split = 0;
  // This handles the case where the axis is negative. It also splits outputs evenly according to num_ouputs if
  // split_sizes is empty.
  ORT_RETURN_IF_ERROR(PrepareForCompute(input_shape, num_outputs, axis, before_dims, after_dims_including_split_axis,
                                        after_dims_excluding_split, split_sizes));

  // Create all output tensors first (required for ONNX node contract)
  auto output_dimensions = input_shape.AsShapeVector();
  std::vector<Tensor*> all_outputs;
  std::vector<int> non_empty_output_indices;

  for (int i = 0; i < num_outputs; ++i) {
    // Update the size of dimension for axis we're splitting on.
    auto split_size = narrow<int>(split_sizes[i]);
    output_dimensions[narrow<size_t>(axis)] = split_size;

    Tensor* output = context.Output(i, TensorShape{output_dimensions});
    all_outputs.push_back(output);

    // Only include non-empty outputs in the GPU program
    if (split_size > 0) {
      non_empty_output_indices.push_back(i);
    }
  }

  uint32_t input_size = onnxruntime::narrow<uint32_t>(input_shape.Size());
  // Early return if the input tensor is empty or all outputs are empty.
  if (input_size == 0 || non_empty_output_indices.empty()) {
    return Status::OK();
  }

  uint32_t previous_sum = 0;
  std::vector<uint32_t> sizes_in_split_axis;
  // sizes_in_split_axis are the cumulative sizes of the NON-EMPTY splits in the split axis.
  for (int output_idx : non_empty_output_indices) {
    previous_sum += onnxruntime::narrow<uint32_t>(split_sizes[output_idx]);
    sizes_in_split_axis.push_back(previous_sum);
  }

  const int64_t inner_size = input_shape.SizeFromDimension(narrow<size_t>(axis) + 1);
  InlinedVector<uint32_t> segment_vector_sizes;
  segment_vector_sizes.reserve(non_empty_output_indices.size());
  bool use_contiguous_vec4 = input->DataType() == DataTypeImpl::GetType<float>();
  for (int output_idx : non_empty_output_indices) {
    const int64_t segment_size = split_sizes[output_idx] * inner_size;
    if (segment_size % 4 != 0) {
      use_contiguous_vec4 = false;
      break;
    }
    segment_vector_sizes.push_back(narrow<uint32_t>(segment_size / 4));
  }

  if (use_contiguous_vec4) {
    const uint32_t input_vector_count = input_size / 4;
    SplitContiguousVec4Program program{segment_vector_sizes};
    program.AddInput(
        {input, ProgramTensorMetadataDependency::Type, TensorShape({input_vector_count}), 4});
    for (int output_idx : non_empty_output_indices) {
      Tensor* output = all_outputs[output_idx];
      program.AddOutput(
          {output, ProgramTensorMetadataDependency::Type, TensorShape({output->Shape().Size() / 4}), 4});
    }
    program.SetDispatchGroupSize((input_vector_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .CacheHint(absl::StrJoin(segment_vector_sizes, ","))
        .AddUniformVariable({input_vector_count});
    return context.RunProgram(program);
  }

  SplitProgram program{static_cast<uint32_t>(axis)};
  program.AddInput({input, ProgramTensorMetadataDependency::TypeAndRank});
  for (int output_idx : non_empty_output_indices) {
    program.AddOutput({all_outputs[output_idx], ProgramTensorMetadataDependency::Rank});
  }
  program
      .SetDispatchGroupSize((input_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .CacheHint(std::to_string(axis))
      .AddUniformVariables(
          {input_size, gsl::span<const uint32_t>(sizes_in_split_axis.data(), sizes_in_split_axis.size())});
  return context.RunProgram(program);
}

#define WEBGPU_SPLIT_KERNEL(OP_TYPE, VERSION, KERNEL_CLASS, TYPE)                                         \
  ONNX_OPERATOR_KERNEL_EX(OP_TYPE, kOnnxDomain, VERSION, kWebGpuExecutionProvider,                        \
                          KernelDefBuilder().TypeConstraint("T", TYPE).InputMemoryType(OrtMemTypeCPU, 1), \
                          KERNEL_CLASS);

#define WEBGPU_SPLIT_VERSIONED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, KERNEL_CLASS, TYPE)                        \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(OP_TYPE, kOnnxDomain, VERSION_FROM, VERSION_TO, kWebGpuExecutionProvider,       \
                                    KernelDefBuilder().TypeConstraint("T", TYPE).InputMemoryType(OrtMemTypeCPU, 1), \
                                    KERNEL_CLASS);

WEBGPU_SPLIT_VERSIONED_KERNEL(Split, 1, 1, Split_1, WebGpuSupportedNumberTypes())
WEBGPU_SPLIT_VERSIONED_KERNEL(Split, 2, 10, Split_2_10, WebGpuSupportedNumberTypes())
WEBGPU_SPLIT_VERSIONED_KERNEL(Split, 11, 12, Split_11_12, WebGpuSupportedNumberTypes())
WEBGPU_SPLIT_VERSIONED_KERNEL(Split, 13, 17, Split_13_17, WebGpuSupportedNumberTypes())
WEBGPU_SPLIT_KERNEL(Split, 18, Split_18, WebGpuSupportedNumberTypes());

}  // namespace webgpu
}  // namespace onnxruntime
