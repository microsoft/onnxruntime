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

  ShaderUsage output_usage = ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias;
  ProgramVariableDataType output_var_type = this->Outputs()[0].var_type;
  if (NeedSplitK()) {
    // When Split-K is enabled, we should declare output as `atomic<u32>` to call atomic built-in functions on it.
    output_usage |= ShaderUsage::UseAtomicU32ForSplitK | ShaderUsage::UseIndicesToOffset | ShaderUsage::UseShapeAndStride;
  }
  const auto& output = shader.AddOutput("output", output_usage);

  const auto& batch_dims = shader.AddIndices("batch_dims", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);

  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  std::string apply_activation = GetActivationSnippet(activation_, "output_value_t", "output_element_t");
  // declare the read and write functions
  MatMulReadFnSource(shader, a, b, &batch_dims, /*transA = */ false, /*transB = */ false, is_vec4_);
  MatMulWriteFnSource(shader, output, has_bias_, /* is_gemm = */ false, 1, is_vec4_ ? 4 : 1, false, apply_activation, is_channels_last_, NeedSplitK(), output_var_type);
  std::string data_type = "a_element_t";
  // generate the main function
  if (is_vec4_) {
    ORT_RETURN_IF_ERROR(MakeMatMulPackedVec4Source(
        shader, elements_per_thread_, WorkgroupSizeX(), WorkgroupSizeY(), data_type, &batch_dims,
        /*transA = */ false, /*transB = */ false, /*alpha = */ 1.f, /*need_handle_matmul = */ true,
        /*output_components = */ 4, /*tile_inner = */ 32, NeedSplitK(), split_dim_inner_));
  } else {
    ORT_RETURN_IF_ERROR(MakeMatMulPackedSource(shader, elements_per_thread_, WorkgroupSizeX(), WorkgroupSizeY(), data_type, &batch_dims));
  }
  return Status::OK();
}

bool MatMulProgram::NeedSplitK() const {
  return split_dim_inner_ > 1;
}

Status MatMulFillBiasBeforeSplitKProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }

  // Handle bias with `MatMulWriteFnSource()`.
  // Here `use_split_k` is false because we just initialize `output` with bias.
  // The computation with Split-K will all be implemented in `MakeMatMulPackedVec4Source()`.
  MatMulWriteFnSource(
      shader, output, has_bias_, /*is_gemm*/ false, /*c_components*/ 4, /*output_components*/ 4, /*c_is_scalar*/ false,
      /*activation_snippet*/ "", is_channels_last_, /*use_split_k*/ false);

  shader.MainFunctionBody() << R"(
  let output_components = 4;)";
  shader.MainFunctionBody() << R"(
  let elements_per_thread = )"
                            << ELEMENTS_PER_THREAD
                            << ";\n";
  shader.MainFunctionBody() << R"(
  let global_row = global_id.y;
  if (global_row >= uniforms.dim_a_outer) {
    return;
  }
  let dim_b_outer = i32(uniforms.dim_b_outer) / output_components;
  let batch = 0;
  let row = i32(global_row);
  let value = output_value_t();
  let start_col = i32(global_id.x) * elements_per_thread;
  for (var col = start_col; col < start_col + elements_per_thread; col++) {
     mm_write(batch, row, col, value);
  })";

  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
