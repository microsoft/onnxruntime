// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/gather.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

Status GatherProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& data = shader.AddInput("data", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& indices = shader.AddInput("input_indices", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.data_size")
                            << "  let output_indices = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "  var indices_indices = input_indices_indices_t(0);\n";
  for (int i = 0; i < indices.Rank(); i++) {
    shader.MainFunctionBody() << "  " << indices.IndicesSet("indices_indices", i, output.IndicesGet("output_indices", axis_ + i)) << ";\n";
  }
  shader.MainFunctionBody() << "  var idx = " << indices.GetByIndices("indices_indices") << ";\n"
                            << "  if (idx < 0) {\n"
                            << "    idx = idx + input_indices_value_t(" << data.IndicesGet("uniforms.data_shape", axis_) << ");\n"
                            << "  }\n"
                            << "  var data_indices : data_indices_t;\n";
  for (int i = 0, j = 0; i < data.Rank(); i++) {
    if (i == SafeInt<int>(axis_)) {
      shader.MainFunctionBody() << "  " << data.IndicesSet("data_indices", i, "u32(idx)") << ";\n";
      j += indices.Rank();
    } else {
      shader.MainFunctionBody() << "  " << data.IndicesSet("data_indices", i, output.IndicesGet("output_indices", j)) << ";\n";
      j++;
    }
  }

  shader.MainFunctionBody() << "  " << output.SetByOffset("global_idx", data.GetByIndices("data_indices"));

  return Status::OK();
}

Status Gather::ComputeInternal(ComputeContext& context) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(&context.KernelContext(), p));
  uint32_t data_size = SafeInt<uint32_t>(p.output_tensor->Shape().Size());
  if (data_size == 0) {
    return Status::OK();
  }

  uint32_t axis = static_cast<uint32_t>(p.axis);
  GatherProgram program{axis};
  program
      .AddInputs({{p.input_tensor, ProgramTensorMetadataDependency::TypeAndRank},
                  {p.indices_tensor, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutput({p.output_tensor, ProgramTensorMetadataDependency::Rank})
      .SetDispatchGroupSize((data_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .CacheHint(std::to_string(axis))
      .AddUniformVariables({{data_size}});
  return context.RunProgram(program);
}

#define WEBGPU_GATHER_KERNEL(OP_TYPE, VERSION, KERNEL_CLASS, TYPE)                                                                              \
  ONNX_OPERATOR_KERNEL_EX(                                                                                                                      \
      OP_TYPE, kOnnxDomain, VERSION, kWebGpuExecutionProvider,                                                                                  \
      KernelDefBuilder().TypeConstraint("T", TYPE).TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<TypeList<int32_t, int64_t>>()), \
      KERNEL_CLASS);

#define WEBGPU_GATHER_VERSIONED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, KERNEL_CLASS, TYPE)                                                   \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                                                                            \
      OP_TYPE, kOnnxDomain, VERSION_FROM, VERSION_TO, kWebGpuExecutionProvider,                                                                 \
      KernelDefBuilder().TypeConstraint("T", TYPE).TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<TypeList<int32_t, int64_t>>()), \
      KERNEL_CLASS);

WEBGPU_GATHER_VERSIONED_KERNEL(Gather, 1, 10, Gather, WebGpuSupportedNumberTypes())
WEBGPU_GATHER_VERSIONED_KERNEL(Gather, 11, 12, Gather, WebGpuSupportedNumberTypes())
WEBGPU_GATHER_KERNEL(Gather, 13, Gather, WebGpuSupportedNumberTypes())

}  // namespace webgpu
}  // namespace onnxruntime
