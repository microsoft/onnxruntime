// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/subgroup_matrix_pad_b.h"

#include <limits>

#include "core/common/narrow.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace webgpu {

namespace {

class SubgroupMatrixPadBProgram final : public Program<SubgroupMatrixPadBProgram> {
 public:
  SubgroupMatrixPadBProgram() : Program{"SubgroupMatrixPadB"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& input_b = shader.AddInput("input_b", ShaderUsage::UseValueTypeAlias);
    const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias);
    return WGSL_TEMPLATE_APPLY(shader, "math/subgroup_matrix_matmul_pad_b.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(input_b, input_b),
                               WGSL_TEMPLATE_VARIABLE(output, output));
  }
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"N_b", ProgramUniformVariableDataType::Uint32});
};

}  // namespace

Status PadSubgroupMatrixWeightColumns(ComputeContext& context, const Tensor& b, const TensorShape& b_shape,
                                      uint32_t N, uint32_t n_b, std::unique_ptr<Tensor>& padded) {
  TensorShapeVector padded_dims{b_shape.GetDims().begin(), b_shape.GetDims().end()};
  padded_dims.back() = static_cast<int64_t>(n_b);
  const TensorShape padded_shape{padded_dims};
  const int64_t output_size_i64 = padded_shape.Size();
  ORT_RETURN_IF_NOT(output_size_i64 <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                    "Cannot pad the subgroup-matrix weight because the padded tensor has ", output_size_i64,
                    " elements, exceeding uint32_t shader indexing range.");
  const uint32_t output_size = narrow<uint32_t>(output_size_i64);

  padded = std::make_unique<Tensor>(context.CreateGPUTensor(b.DataType(), padded_shape));
  // A zero-element padded tensor needs no pad pass -- dispatching 0 workgroups is
  // pointless and some drivers reject it.
  if (output_size == 0) {
    return Status::OK();
  }
  SubgroupMatrixPadBProgram program;
  program.SetWorkgroupSize(WORKGROUP_SIZE)
      .SetDispatchGroupSize(CeilDiv<uint32_t>(output_size, WORKGROUP_SIZE))
      .AddInput({&b, ProgramTensorMetadataDependency::TypeAndRank, b_shape, 1})
      .AddOutput({padded.get(), ProgramTensorMetadataDependency::TypeAndRank, padded->Shape(), 1})
      .AddUniformVariables({{output_size}, {N}, {n_b}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
