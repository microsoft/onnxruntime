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

Status ConcatProgram::GenerateShaderCode(ShaderHelper& shader) const {
  size_t input_count = Inputs().size();
  std::vector<const ShaderVariableHelper*> inputs;
  inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    inputs.push_back(&shader.AddInput("input_" + std::to_string(i), ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias));
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size");
  for (size_t i = 0; i < input_count; ++i) {
    const std::string output_indices_i = absl::StrCat("output_indices_", i);
    const std::string output_indices_i_axis = output_indices_i + (inputs[i]->Rank() > 1 ? "[" + std::to_string(axis_) + "]" : "");
    const std::string concat_axis_offset = GetElementAt("uniforms.sizes_in_concat_axis", std::to_string(i), input_count);

    shader.MainFunctionBody() << "    var " << output_indices_i << " = " << inputs[i]->OffsetToIndices("global_idx") << ";\n"
                              << "    " << output_indices_i_axis << " += " << concat_axis_offset << ";\n"
                              << "    " << output.SetByIndices(output_indices_i, inputs[i]->GetByOffset("global_idx")) << "\n";
  }

  return Status::OK();
}

Status Concat::ComputeInternal(ComputeContext& context) const {
  uint32_t input_count = context.InputCount();
  InlinedTensorsVector input_tensors;
  input_tensors.reserve(input_count);
  for (uint32_t i = 0; i < input_count; ++i) {
    input_tensors.push_back(context.Input<Tensor>(i));
  }

  Prepare prepare;
  ORT_RETURN_IF_ERROR(PrepareForCompute(&context.KernelContext(), input_tensors, prepare));
  if (prepare.output_num_elements == 0) {
    return Status::OK();
  }

  uint32_t axis = static_cast<uint32_t>(prepare.axis);
  uint32_t max_inputs_per_concat = context.DeviceLimits().maxStorageBuffersPerShaderStage - 1;

  uint32_t input_index = 0;
  uint32_t cumulative_output_size = 0;
  uint32_t cumulative_size_in_concat_axis = 0;

  while (input_index < input_count) {
    ConcatProgram program{axis};
    uint32_t num_inputs_this_concat = std::min(max_inputs_per_concat, input_count - input_index);

    std::vector<uint32_t> sizes;
    std::vector<uint32_t> sizes_in_concat_axis;
    sizes.reserve(num_inputs_this_concat + 1);
    sizes_in_concat_axis.reserve(num_inputs_this_concat + 1);

    // Start with the cumulative size from previous dispatches
    sizes.push_back(cumulative_output_size);
    sizes_in_concat_axis.push_back(cumulative_size_in_concat_axis);

    uint32_t dispatch_size = 0;
    for (uint32_t i = 0; i < num_inputs_this_concat; i++) {
      auto& input = prepare.inputs[input_index + i];
      program.AddInput({input.tensor, ProgramTensorMetadataDependency::TypeAndRank});

      uint32_t size = onnxruntime::narrow<int32_t>(input.tensor->Shape().Size());
      uint32_t axis_size = static_cast<uint32_t>(input.tensor->Shape()[axis]);

      cumulative_output_size += size;
      dispatch_size += size;
      sizes.push_back(cumulative_output_size);

      cumulative_size_in_concat_axis += axis_size;
      sizes_in_concat_axis.push_back(cumulative_size_in_concat_axis);
    }

    program.CacheHint(absl::StrJoin(std::make_tuple(num_inputs_this_concat, prepare.axis), ","))
        .AddOutputs({prepare.output_tensor})
        .SetDispatchGroupSize((dispatch_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .AddUniformVariables({gsl::span<const uint32_t>(sizes_in_concat_axis.data(), sizes_in_concat_axis.size()), dispatch_size});
    ORT_RETURN_IF_ERROR(context.RunProgram(program));

    input_index += num_inputs_this_concat;
  }

  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime