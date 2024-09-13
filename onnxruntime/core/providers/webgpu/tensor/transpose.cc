// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/tensor/transpose.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_variable.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Transpose,
    kOnnxDomain,
    1, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    Transpose);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Transpose,
    kOnnxDomain,
    13, 20,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    Transpose);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Transpose,
    kOnnxDomain,
    21, 22,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    Transpose);

ONNX_OPERATOR_KERNEL_EX(
    Transpose,
    kOnnxDomain,
    23,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    Transpose);

const std::string AppendPermFunction(gsl::span<const size_t> perm) {
  std::ostringstream ss;
  ss.imbue(std::locale::classic());
  ss << "fn perm(i: y_indices_t)->x_indices_t {\n"
        "  var a: x_indices_t;\n";
  for (size_t i = 0; i < perm.size(); ++i) {
    ss << "  a[" << perm[i] << "] = i[" << i << "];\n";
  }
  ss << "  return a;\n"
        "}\n";
  return ss.str();
}

Status TransposeProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("x", ShaderVariable::UseUniform | ShaderVariable::UseIndicesTypeAlias);
  const auto& output = shader.AddOutput("y", ShaderVariable::UseUniform | ShaderVariable::UseIndicesTypeAlias);
  shader.AppendImplementation(AppendPermFunction(this->perm_));
  shader.SetMainFunctionBody(shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size"),
                             "  let indices = ", output.OffsetToIndices("global_idx"),
                             ";\n"
                             "  let x_indices = perm(indices); \n"
                             "  ",
                             output.SetByOffset("global_idx", input.GetByIndices("x_indices")));
  return Status::OK();
}

Status Transpose::ComputeInternal(ComputeContext& context) const {
  // TODO: there is an optimized version of transpose to port.
  const auto* input_tensor = context.Input(0);
  const TensorShape& input_shape = input_tensor->Shape();
  int32_t rank = gsl::narrow_cast<int32_t>(input_shape.NumDimensions());

  TensorShapeVector output_dims(rank);
  InlinedVector<size_t> default_perm(rank);
  const InlinedVector<size_t>* p_perm = nullptr;
  ORT_RETURN_IF_ERROR(ComputeOutputShape(*input_tensor, output_dims, default_perm, p_perm));
  TensorShape output_shape(output_dims);
  auto* output_tensor = context.Output(0, output_shape);

  uint32_t output_size = gsl::narrow_cast<int32_t>(input_tensor->Shape().Size());
  TransposeProgram program{*p_perm};
  program
      .CacheHint(absl::StrJoin(*p_perm, "-"))
      .AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutputs({output_tensor})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({
          {static_cast<uint32_t>(output_size)},
      });
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
