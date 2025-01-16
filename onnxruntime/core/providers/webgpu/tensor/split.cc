// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/split.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

namespace {

// Helper function to calculate the output index based on the input index and the sizes of the splits.
void CalculateOutputIndex(std::ostream& os, size_t output_count) {
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
void WriteBufferData(std::ostream& os, const ShaderVariableHelper& input,
                     gsl::span<const ShaderVariableHelper*> outputs) {
  os << "fn write_buffer_data(output_number: u32, global_idx: u32,  indices: output_0_indices_t) {\n";
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto buffer_write = outputs[i]->SetByIndices("indices", input.GetByOffset("global_idx"));
    if (outputs.size() == 1) {
      os << buffer_write;
    } else if (i == 0) {
      os << "  if (output_number == 0u) {\n"
         << "    " << buffer_write << "\n";
    } else if (i == outputs.size() - 1) {
      os << "  } else {\n"
         << "    " << buffer_write << "\n";
    } else {
      os << "  } else if (output_number == " << i << "u) {\n"
         << "    " << buffer_write << "\n";
    }
  }
  os << "  }\n"
     << "}\n";
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
                            << "    index -= uniforms.sizes_in_split_axis[output_number - 1u];\n"
                            << "    " << input.IndicesSet("indices", axis_, "index") << "\n"
                            << "  }\n"
                            << "  write_buffer_data(output_number, global_idx, indices);\n";

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
  if (split_sizes_.size() == 0 && context.InputCount() > 1) {
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

  SplitProgram program{gsl::narrow_cast<uint32_t>(axis)};
  program.AddInput({input, ProgramTensorMetadataDependency::TypeAndRank});

  auto output_dimensions = input_shape.AsShapeVector();
  for (int i = 0; i < num_outputs; ++i) {
    // Update the size of dimension for axis we're splitting on.
    auto split_size = narrow<int>(split_sizes[i]);
    output_dimensions[narrow<size_t>(axis)] = split_size;

    Tensor* output = context.Output(i, TensorShape{output_dimensions});
    program.AddOutput({output, ProgramTensorMetadataDependency::Rank});
  }

  uint32_t input_size = gsl::narrow<uint32_t>(input_shape.Size());
  // Early return if the input tensor is empty.
  if (input_size == 0) {
    return Status::OK();
  }

  uint32_t previous_sum = 0;
  std::vector<uint32_t> sizes_in_split_axis;
  // sizes_in_split_axis are the cumulative sizes of the splits in the split axis.
  for (auto split_size : split_sizes) {
    previous_sum += gsl::narrow<uint32_t>(split_size);
    sizes_in_split_axis.push_back(previous_sum);
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
