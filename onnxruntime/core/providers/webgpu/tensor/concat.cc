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

const std::string AppendCalCulateInputIndexFunction(size_t input_count) {
  std::ostringstream ss;
  ss.imbue(std::locale::classic());
  ss << "fn calculate_input_index(index: u32) -> u32 {" << std::endl
     << "  for (var i = 0u; i < " << input_count << "; i = i + 1u) {" << std::endl
     << "    if (index < uniforms.size_in_concat_axis[i]) {" << std::endl
     << "      return i;" << std::endl
     << "    }" << std::endl
     << "  }" << std::endl
     << "  return " << input_count << ";" << std::endl
     << "}" << std::endl;
  return ss.str();
}

const void AppendAssignOutput(std::ostringstream& ss, const ShaderVariableHelper& input, const ShaderVariableHelper& output) {
  ss << output.SetByOffset("global_idx", input.GetByIndices("indices")) << ";" << std::endl;
}

const std::string AppendAssignOutputDataFunction(gsl::span<const ShaderVariableHelper*> inputs, const ShaderVariableHelper& output) {
  std::ostringstream ss;
  size_t input_count = inputs.size();
  ss.imbue(std::locale::classic());
  ss << "fn assign_output_data(global_idx: u32, input_index: u32, indices: output_indices_t) {" << std::endl;
  if (input_count == 0) {
    AppendAssignOutput(ss, *inputs[0], output);
  } else {
    for (size_t i = 0; i < input_count; ++i) {
      if (i == 0) {
        ss << "  if (input_index == 0u) {" << std::endl;
      } else if (i == input_count - 1) {
        ss << "  } else {" << std::endl;
      } else {
        ss << "  } else if (input_index == " << i << "u) {" << std::endl;
      }
      ss << "     ";
      AppendAssignOutput(ss, *inputs[i], output);
    }
    ss << "  }" << std::endl;
  }
  ss << "}" << std::endl;
  return ss.str();
}
Status ConcatProgram::GenerateShaderCode(ShaderHelper& shader) const {
  std::vector<const ShaderVariableHelper*> inputs;
  inputs.reserve(input_count_);
  for (size_t i = 0; i < input_count_; ++i) {
    inputs.push_back(&shader.AddInput("input_" + std::to_string(i), ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias));
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  shader.AppendImplementation(AppendCalCulateInputIndexFunction(input_count_));
  shader.AppendImplementation(AppendAssignOutputDataFunction(gsl::make_span(inputs), output));
  shader.SetMainFunctionBody(shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size"),
                             "  var indices = ", output.OffsetToIndices("global_idx"), ";\n",
                             "  let indices_axis = ", output.IndicesGet("indices", axis_), ";\n",
                             "  let input_index = calculate_input_index(indices_axis);\n",
                             "  if (input_index != 0u) {\n",
                             "     ", output.IndicesSet("indices", axis_, "indices_axis - uniforms.size_in_concat_axis[input_index - 1]"), ";\n",
                             "  }\n",
                             "  assign_output_data(global_idx, input_index, indices);\n");
  return Status::OK();
}

Status Concat::ComputeInternal(ComputeContext& context) const {
  auto input_count = context.InputCount();
  InlinedTensorsVector input_tensors;
  input_tensors.reserve(input_count);
  if (SafeInt<uint32_t>(input_count + 1) > context.DeviceLimits().maxStorageBuffersPerShaderStage) {
    // TODO: support when input_count + 1 > maxStorageBuffersPerShaderStage, by raising the limit or run the program in multiple passes.
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "The number of storage buffer (input=",
                           input_count, ", output=1) exceeds the limit (",
                           context.DeviceLimits().maxStorageBuffersPerShaderStage, ") of the device.");
  }

  for (int i = 0; i < input_count; ++i) {
    input_tensors.push_back(context.Input<Tensor>(i));
  }
  Prepare prepare;
  ORT_RETURN_IF_ERROR(PrepareForCompute(&context.GetKernelContext(), input_tensors, prepare));
  if (prepare.output_num_elements == 0) {
    return Status::OK();
  }
  auto* output_tensor = context.Output(0, prepare.output_tensor->Shape());
  uint32_t output_size = gsl::narrow_cast<int32_t>(output_tensor->Shape().Size());
  std::vector<uint32_t> sizes_in_concat_axis;
  sizes_in_concat_axis.reserve(input_count);
  uint32_t sum = 0;
  sizes_in_concat_axis.reserve(input_count);
  for (int i = 0; i < input_count; ++i) {
    const auto& input = prepare.inputs[i];
    auto axis_size = input.tensor->Shape()[prepare.axis];
    sum += static_cast<uint32_t>(axis_size);
    sizes_in_concat_axis.push_back(sum);
  }

  ConcatProgram program(input_count, prepare.axis);
  for (int i = 0; i < input_count; ++i) {
    program.AddInput({input_tensors[i], ProgramTensorMetadataDependency::TypeAndRank});
  }
  program.CacheHint(absl::StrCat(input_count, prepare.axis))
      .AddOutputs({output_tensor})
      .SetDispatchGroupSize((prepare.output_num_elements + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({gsl::span<const uint32_t>(sizes_in_concat_axis.data(), sizes_in_concat_axis.size()),
                            {static_cast<uint32_t>(output_size)}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
