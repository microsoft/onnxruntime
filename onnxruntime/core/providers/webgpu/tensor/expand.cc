// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/expand.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

namespace {
Status ComputeOutputShape(const std::string& node_name, const TensorShape& lhs_shape, const TensorShape& rhs_shape, TensorShape& out_shape) {
  size_t lhs_rank = lhs_shape.NumDimensions();
  size_t rhs_rank = rhs_shape.NumDimensions();
  size_t out_rank = std::max(lhs_rank, rhs_rank);

  std::vector<int64_t> output_dims(out_rank, 0);
  for (size_t i = 0; i < out_rank; ++i) {
    int64_t lhs_dim = 1;
    if (i < lhs_rank)
      lhs_dim = lhs_shape[lhs_rank - 1 - i];
    int64_t rhs_dim = 1;
    if (i < rhs_rank)
      rhs_dim = rhs_shape[rhs_rank - 1 - i];
    int64_t max = std::max(lhs_dim, rhs_dim);
    int64_t min = std::min(lhs_dim, rhs_dim);
    int64_t out_dim = (min == 0 ? min : max);  // special case a dim value of 0.
    if (lhs_dim != out_dim && lhs_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": left operand cannot broadcast on dim ", lhs_rank - 1 - i,
                             " LeftShape: ", lhs_shape.ToString(), ", RightShape: ", rhs_shape.ToString());
    if (rhs_dim != out_dim && rhs_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": right operand cannot broadcast on dim ", rhs_rank - 1 - i,
                             " LeftShape: ", lhs_shape.ToString(), ", RightShape: ", rhs_shape.ToString());
    output_dims[out_rank - 1 - i] = out_dim;
  }
  out_shape = TensorShape(output_dims);
  return Status::OK();
}
}  // namespace

Status ExpandProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input",
                                      ToProgramVariableDataType(Inputs()[0].tensor->GetElementType()),
                                      ShaderVariable::UseUniform);
  const auto& output = shader.AddOutput("output",
                                        ToProgramVariableDataType(Outputs()[0].tensor->GetElementType()),
                                        ShaderVariable::UseUniform);

  shader.MainFunctionBody(shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size"),
                          "let output_indices = ", output.OffsetToIndices("global_idx"), ";\n",
                          "let input_offset = ", input.BroadcastedIndicesToOffset("output_indices", output), ";\n",
                          output.SetByOffset("global_idx", input.GetByOffset("input_offset")));

  return Status::OK();
}

Status Expand::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  const auto* input_shape_tensor = context.Input(1);

  const auto* p_shape = input_shape_tensor->Data<int64_t>();
  TensorShapeVector output_dims{p_shape, p_shape + input_shape_tensor->Shape().Size()};
  TensorShape output_shape(output_dims);
  ORT_RETURN_IF_ERROR(ComputeOutputShape(Node().Name(), input_tensor->Shape(), output_dims, output_shape));

  auto* output_tensor = context.Output(0, output_shape);
  SafeInt<uint32_t> vec_size = output_shape.Size();
  ExpandProgram program{"Expand"};
  program
      .Inputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank}})
      .Outputs({{output_tensor, ProgramTensorMetadataDependency::Rank}})
      .DispatchGroupSize((vec_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .UniformVariables({
          {static_cast<uint32_t>(vec_size)},
      });
  return context.RunProgram(program);
}

#define WEBGPU_EXPAND_KERNEL(OP_TYPE, VERSION, KERNEL_CLASS, TYPE)                    \
  ONNX_OPERATOR_KERNEL_EX(                                                            \
      OP_TYPE, kOnnxDomain, VERSION, kWebGpuExecutionProvider,                        \
      KernelDefBuilder().TypeConstraint("T", TYPE).InputMemoryType(OrtMemTypeCPU, 1), \
      KERNEL_CLASS);

#define WEBGPU_EXPAND_VERSIONED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, KERNEL_CLASS, TYPE) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                          \
      OP_TYPE, kOnnxDomain, VERSION_FROM, VERSION_TO, kWebGpuExecutionProvider,               \
      KernelDefBuilder().TypeConstraint("T", TYPE).InputMemoryType(OrtMemTypeCPU, 1),         \
      KERNEL_CLASS);

WEBGPU_EXPAND_VERSIONED_KERNEL(Expand, 8, 12, Expand, WebGpuSupportedFloatTypes())
WEBGPU_EXPAND_KERNEL(Expand, 13, Expand, WebGpuSupportedFloatTypes())

}  // namespace webgpu
};  // namespace onnxruntime
