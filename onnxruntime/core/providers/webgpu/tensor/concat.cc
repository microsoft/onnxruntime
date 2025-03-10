// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/webgpu/tensor/concat.h"

#include "core/common/inlined_containers.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/shader_variable.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_CONCAT_VERSIONED_KERNEL(start, end)            \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                          \
      Concat,                                                 \
      kOnnxDomain,                                            \
      start,                                                  \
      end,                                                    \
      kWebGpuExecutionProvider,                               \
      (*KernelDefBuilder::Create())                           \
          .TypeConstraint("T", WebGpuSupportedNumberTypes()), \
      Concat);

#define WEBGPU_CONCAT_KERNEL(version)                         \
  ONNX_OPERATOR_KERNEL_EX(                                    \
      Concat,                                                 \
      kOnnxDomain,                                            \
      version,                                                \
      kWebGpuExecutionProvider,                               \
      (*KernelDefBuilder::Create())                           \
          .TypeConstraint("T", WebGpuSupportedNumberTypes()), \
      Concat);

WEBGPU_CONCAT_VERSIONED_KERNEL(1, 3)
WEBGPU_CONCAT_VERSIONED_KERNEL(4, 10)
WEBGPU_CONCAT_VERSIONED_KERNEL(11, 12)
WEBGPU_CONCAT_KERNEL(13)

void AppendCalCulateInputIndexFunction(std::ostream& os, size_t input_count) {
  os << "fn calculate_input_index(index: u32) -> u32 {\n"
     << "  for (var i = 0u; i < " << input_count << "; i = i + 1u) {\n"
     << "    if (index < " << GetElementAt("uniforms.size_in_concat_axis", "i", input_count) << ") {\n"
     << "      return i;\n"
     << "    }\n"
     << "  }\n"
     << "  return " << input_count << ";\n"
     << "}\n";
}

void AppendAssignOutputDataFunction(std::ostream& os, gsl::span<const ShaderVariableHelper*> inputs, const ShaderVariableHelper& output) {
  os << "fn assign_output_data(global_idx: u32, input_index: u32, indices: output_indices_t) {\n";
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (i == 0) {
      os << "  if (input_index == 0u) {\n";
    } else if (i == inputs.size() - 1) {
      os << "  } else {\n";
    } else {
      os << "  } else if (input_index == " << i << "u) {\n";
    }
    os << "     " << output.SetByOffset("global_idx", inputs[i]->GetByIndices("indices")) << ";\n";
  }
  os << "  }\n"
        "}\n";
}

Status ConcatProgram::GenerateShaderCode(ShaderHelper& shader) const {
  size_t input_count = Inputs().size();
  std::vector<const ShaderVariableHelper*> inputs;
  inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    inputs.push_back(&shader.AddInput("input_" + std::to_string(i), ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias));
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);

  // add implementation of fn calculate_input_index
  AppendCalCulateInputIndexFunction(shader.AdditionalImplementation(), input_count);
  // add implementation of fn assign_output_data
  AppendAssignOutputDataFunction(shader.AdditionalImplementation(), inputs, output);
  const std::string size_in_concat_axis = GetElementAt("uniforms.size_in_concat_axis", "input_index - 1", input_count);
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "  var indices = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "  let indices_axis = " << output.IndicesGet("indices", axis_) << ";\n"
                            << "  let input_index = calculate_input_index(indices_axis);\n"
                            << "  if (input_index != 0u) {\n"
                            << "    " << output.IndicesSet("indices", axis_, "indices_axis - " + size_in_concat_axis) << ";\n"
                            << "  }\n"
                               "  assign_output_data(global_idx, input_index, indices);\n";
  return Status::OK();
}

Status Concat::ComputeInternal(ComputeContext& context) const {
  int input_count = context.InputCount();
  InlinedTensorsVector input_tensors;
  input_tensors.reserve(input_count);
  for (int i = 0; i < input_count; ++i) {
    input_tensors.push_back(context.Input<Tensor>(i));
  }

  Prepare prepare;
  ORT_RETURN_IF_ERROR(PrepareForCompute(&context.KernelContext(), input_tensors, prepare));
  if (prepare.output_num_elements == 0) {
    return Status::OK();
  }

  uint32_t output_size = gsl::narrow_cast<int32_t>(prepare.output_tensor->Shape().Size());

  size_t axis = static_cast<size_t>(prepare.axis);
  ConcatProgram program{axis};

  std::vector<uint32_t> sizes_in_concat_axis;
  sizes_in_concat_axis.reserve(input_count);
  uint32_t sum = 0;
  for (int i = 0; i < input_count; ++i) {
    const auto& input = prepare.inputs[i];
    if (input.tensor->Shape().Size() == 0) {
      continue;
    }
    program.AddInput({input.tensor, ProgramTensorMetadataDependency::TypeAndRank});

    auto axis_size = input.tensor->Shape()[axis];
    sum += static_cast<uint32_t>(axis_size);
    sizes_in_concat_axis.push_back(sum);
  }

  size_t non_empty_input_count = sizes_in_concat_axis.size();

  if (non_empty_input_count + 1 > context.DeviceLimits().maxStorageBuffersPerShaderStage) {
    // TODO: support when input_count + 1 > maxStorageBuffersPerShaderStage, by raising the limit or run the program in multiple passes.
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "The number of storage buffer (input=",
                           input_count, ", output=1) exceeds the limit (",
                           context.DeviceLimits().maxStorageBuffersPerShaderStage, ") of the device.");
  }

  program.CacheHint(absl::StrJoin(std::make_tuple(non_empty_input_count, prepare.axis), ","))
      .AddOutputs({prepare.output_tensor})
      .SetDispatchGroupSize((prepare.output_num_elements + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({gsl::span<const uint32_t>(sizes_in_concat_axis.data(), sizes_in_concat_axis.size()),
                            output_size});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
