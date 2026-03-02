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
  if (need_split_k) {
    MatMulWriteFnSourceWithSplitK(shader, output, /*is_gemm = */ false, output_var_type);
  } else {
    MatMulWriteFnSourceForMatMul(shader, output, bias, apply_activation, is_channels_last_);
  }
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

  // Handle bias with `MatMulWriteFnSourceForGemm() or MatMulWriteFnSourceForMatMul()`.
  // const uint32_t bias_components = output_components_;
  if (is_gemm_) {
    MatMulWriteFnSourceForGemm(shader, output, bias, bias_is_scalar_);
  } else {
    // Currently we only support `is_channels_last` to be true and no activation.
    MatMulWriteFnSourceForMatMul(shader, output, bias, /*activation_snippet*/ "", /*is_channels_last*/ true);
  }

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
