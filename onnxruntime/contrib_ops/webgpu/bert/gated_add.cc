// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/gated_add.h"

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    GatedAdd,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    GatedAdd);

Status GatedAddProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x", ShaderUsage::UseUniform);
  const auto& y = shader.AddInput("y", ShaderUsage::UseUniform);
  const auto& gate = shader.AddInput("gate", ShaderUsage::UseUniform);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "  let gate_idx = global_idx / uniforms.hidden_size;\n"
                            << "  let gate_value = " << gate.GetByOffset("gate_idx") << ";\n"
                            << "  let value = " << x.GetByOffset("global_idx")
                            << " + (" << y.GetByOffset("global_idx") << " * gate_value);\n"
                            << "  " << output.SetByOffset("global_idx", "value");

  return Status::OK();
}

Status GatedAdd::ComputeInternal(ComputeContext& context) const {
  const auto* x = context.Input(0);
  const auto* y = context.Input(1);
  const auto* gate = context.Input(2);
  const TensorShape& shape = x->Shape();

  ORT_RETURN_IF_NOT(shape.NumDimensions() >= 1, "X must have rank >= 1");
  ORT_RETURN_IF_NOT(y->Shape() == shape, "Y must have the same shape as X");
  ORT_RETURN_IF_NOT(gate->Shape().NumDimensions() == shape.NumDimensions(),
                    "gate must have the same rank as X");

  const size_t last_axis = shape.NumDimensions() - 1;
  const int64_t hidden_size = shape[last_axis];
  ORT_RETURN_IF_NOT(hidden_size > 0, "X last dimension must be positive");
  ORT_RETURN_IF_NOT(gate->Shape()[last_axis] == 1, "gate last dimension must be 1");
  for (size_t axis = 0; axis < last_axis; ++axis) {
    ORT_RETURN_IF_NOT(gate->Shape()[axis] == shape[axis],
                      "gate dimension ", axis, " must match X");
  }

  auto* output = context.Output(0, shape);
  const int64_t output_size = shape.Size();
  if (output_size == 0) {
    return Status::OK();
  }

  GatedAddProgram program{};
  program.AddInputs({{x, ProgramTensorMetadataDependency::Type},
                     {y, ProgramTensorMetadataDependency::Type},
                     {gate, ProgramTensorMetadataDependency::Type}})
      .AddOutput({output, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(output_size) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(output_size)},
                            {onnxruntime::narrow<uint32_t>(hidden_size)}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
