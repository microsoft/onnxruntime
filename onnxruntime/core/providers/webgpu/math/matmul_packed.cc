// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/matmul_packed.h"

#include "core/providers/webgpu/math/gemm_utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include <string>
namespace onnxruntime {
namespace webgpu {

Status MatMulProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& b = shader.AddInput("b", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);

  const bool need_split_k = NeedSplitK();
  ShaderUsage output_usage = ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias;
  if (need_split_k) {
    // When Split-K is enabled, we will declare output as `atomic<i32>` to call atomic built-in
    // functions on it, so we need below information to correctly compute the index on the output.
    output_usage |= ShaderUsage::UseIndicesToOffset | ShaderUsage::UseShapeAndStride;
  }
  const auto& output = shader.AddOutput("output", output_usage);

  const auto& batch_dims = shader.AddIndices("batch_dims", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);

  const ShaderVariableHelper* bias = nullptr;
  if (has_bias_) {
    bias = &shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  std::string apply_activation = GetActivationSnippet(activation_, "output_value_t", "output_element_t");
  ProgramVariableDataType output_var_type = this->Outputs()[0].var_type;
  // declare the read and write functions
  MatMulReadFnSource(shader, a, b, &batch_dims, /*transA = */ false, /*transB = */ false);
  MatMulWriteFnSource(shader, output, bias, /* is_gemm = */ false, 1, false, apply_activation, is_channels_last_, need_split_k, output_var_type);
  std::string data_type = "a_element_t";
  // generate the main function
  if (is_vec4_) {
    ORT_RETURN_IF_ERROR(MakeMatMulPackedVec4Source(
        shader, elements_per_thread_, WorkgroupSizeX(), WorkgroupSizeY(), data_type, &batch_dims,
        /*transA = */ false, /*transB = */ false, /*alpha = */ 1.f, /*need_handle_matmul = */ true,
        /*output_components = */ 4, /*tile_inner = */ 32, need_split_k, split_dim_inner_));
  } else {
    ORT_RETURN_IF_ERROR(MakeMatMulPackedSource(shader, elements_per_thread_, WorkgroupSizeX(), WorkgroupSizeY(), data_type, &batch_dims));
  }
  return Status::OK();
}

bool MatMulProgram::NeedSplitK() const {
  return split_dim_inner_ > 1;
}

Status MatMulFillBiasOrZeroBeforeSplitKProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  const ShaderVariableHelper* bias = nullptr;
  if (has_bias_) {
    bias = &shader.AddInput("bias", ShaderUsage::UseUniform);
  }

  // Handle bias with `MatMulWriteFnSource()`.
  // Here `use_split_k` is false because we just initialize `output` with bias.
  // `use_split_k` is true only when we do the actual MatMul with Split-K.
  const uint32_t bias_components = output_components_;
  MatMulWriteFnSource(
      shader, output, bias, is_gemm_, bias_components, bias_is_scalar_,
      /*activation_snippet*/ "", /*is_channels_last*/ true, /*use_split_k*/ false);

  shader.MainFunctionBody() << "  let output_components = " << output_components_ << ";\n";
  shader.MainFunctionBody() << R"(
  let output_id = i32(global_idx);

  let dim_a_outer = i32(uniforms.dim_a_outer);
  let dim_b_outer = i32(uniforms.dim_b_outer) / output_components;
  if (output_id >= dim_a_outer * dim_b_outer) {
    return;
  }

  let output_row = output_id / dim_b_outer;
  let output_col = output_id % dim_b_outer;
  let output_batch = 0;
  let output_value = output_value_t();
  mm_write(output_batch, output_row, output_col, output_value);
)";

  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
