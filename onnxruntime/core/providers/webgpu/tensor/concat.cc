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
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const std::string output_indices_str = "output_indices" + (input.Rank() > 1 ? "[" + std::to_string(axis_) + "]" : "");

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "  var output_indices = " << input.OffsetToIndices("global_idx") << ";\n"
                            << "  " << output_indices_str << " += uniforms.concat_axis_offset;\n"
                            << "  " << output.SetByIndices("output_indices", input.GetByOffset("global_idx")) << "\n";

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

  uint32_t output_size = onnxruntime::narrow<int32_t>(prepare.output_tensor->Shape().Size());
  size_t axis = static_cast<size_t>(prepare.axis);

  uint32_t concat_axis_offset = 0;
  for (uint32_t input_index = 0; input_index < input_count; input_index++) {
    const auto& input = prepare.inputs[input_index];
    auto axis_size = input.tensor->Shape()[axis];

    ConcatProgram pass_program{axis};
    pass_program.CacheHint(absl::StrJoin(std::make_tuple(prepare.axis), ","))
        .AddInput({input.tensor, ProgramTensorMetadataDependency::TypeAndRank})
        .AddOutputs({prepare.output_tensor})
        .SetDispatchGroupSize((input.tensor->Shape().Size() + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .AddUniformVariables({output_size, concat_axis_offset});
    ORT_RETURN_IF_ERROR(context.RunProgram(pass_program));

    concat_axis_offset += static_cast<uint32_t>(axis_size);
  }

  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime