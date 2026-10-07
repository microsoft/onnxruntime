// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/split.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

Status SplitProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  InlinedVector<const ShaderVariableHelper*> outputs;
  outputs.reserve(output_count_);
  for (size_t i = 0; i < output_count_; ++i) {
    outputs.push_back(
        &shader.AddOutput("output_" + std::to_string(i), ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias));
  }

  // Splitting along `axis` cuts the input into `outer` identical blocks of `total_segment_elements`,
  // and each output takes one contiguous run out of every block. So the flat offset alone decides
  // both which output an element belongs to and where it lands, with no indices arithmetic.
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.input_size")
                            << "  let outer_index = global_idx / uniforms.total_segment_elements;\n"
                            << "  let within_outer = global_idx % uniforms.total_segment_elements;\n"
                            << "  var segment_start = 0u;\n";

  for (size_t i = 0; i < output_count_; ++i) {
    const std::string segment_size = GetElementAt("uniforms.segment_sizes", i, output_count_);
    shader.MainFunctionBody()
        << "  {\n"
        << "    let segment_size = " << segment_size << ";\n"
        << "    if (within_outer < segment_start + segment_size) {\n"
        << "      "
        << outputs[i]->SetByOffset("outer_index * segment_size + within_outer - segment_start",
                                   input.GetByOffset("global_idx"))
        << "\n"
        << "      return;\n"
        << "    }\n"
        << "    segment_start += segment_size;\n"
        << "  }\n";
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

  // Each output takes `split_size * inner_size` contiguous elements out of every outer block, so
  // vec4 is usable whenever every one of those runs is a whole number of vec4s. Otherwise the same
  // program runs with a single component.
  const int64_t inner_size = input_shape.SizeFromDimension(narrow<size_t>(axis) + 1);
  bool use_vec4 = true;
  for (int output_idx : non_empty_output_indices) {
    if ((split_sizes[output_idx] * inner_size) % 4 != 0) {
      use_vec4 = false;
      break;
    }
  }
  const int components = use_vec4 ? 4 : 1;

  InlinedVector<uint32_t> segment_sizes;
  segment_sizes.reserve(non_empty_output_indices.size());
  uint32_t total_segment_elements = 0;
  for (int output_idx : non_empty_output_indices) {
    const uint32_t segment_size = narrow<uint32_t>((split_sizes[output_idx] * inner_size) / components);
    segment_sizes.push_back(segment_size);
    total_segment_elements += segment_size;
  }

  const uint32_t element_count = input_size / components;

  SplitProgram program{non_empty_output_indices.size()};
  program.AddInput({input, ProgramTensorMetadataDependency::Type, TensorShape({element_count}), components});
  for (int output_idx : non_empty_output_indices) {
    Tensor* output = all_outputs[output_idx];
    program.AddOutput({output, ProgramTensorMetadataDependency::Type,
                       TensorShape({output->Shape().Size() / components}), components});
  }
  program.SetDispatchGroupSize((element_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .CacheHint(non_empty_output_indices.size(), components)
      .AddUniformVariables({element_count, total_segment_elements,
                            gsl::span<const uint32_t>(segment_sizes.data(), segment_sizes.size())});
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
