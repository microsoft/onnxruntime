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
  const ShaderUsage output_usage = ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias;
  const auto& output = shader.AddOutput("output", output_usage);

  const auto& batch_dims = shader.AddIndices("batch_dims", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);

  const ShaderVariableHelper* bias = nullptr;
  if (has_bias_) {
    bias = &shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  std::string apply_activation = GetActivationSnippet(activation_, "output_value_t", "output_element_t");
  // Emit activation helpers before the write function uses them.
  shader.AdditionalImplementation() << GetActivationDeclaration(activation_, "output_value_t", "output_element_t");
  // declare the read and write functions
  MatMulReadFnSource(shader, a, b, &batch_dims, /*transA = */ false, /*transB = */ false);
  if (need_split_k) {
    // Bias and activation are applied by `MatMulSplitKReduceProgram` after the partials are summed,
    // so `bias` is deliberately not bound here.
    MatMulWriteFnSourceWithSplitK(shader, output, /*is_gemm = */ false);
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

Status MatMulSplitKReduceProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& partials = shader.AddInput("partials", ShaderUsage::UseUniform);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  const ShaderVariableHelper* bias = nullptr;
  if (has_bias_) {
    bias = &shader.AddInput("bias", ShaderUsage::UseUniform);
  }

  const std::string apply_activation = GetActivationSnippet(activation_, "output_value_t", "output_element_t");
  // Emit activation helpers before the write function uses them.
  shader.AdditionalImplementation() << GetActivationDeclaration(activation_, "output_value_t", "output_element_t");
  if (is_gemm_) {
    // GEMM has no fused activation.
    MatMulWriteFnSourceForGemm(shader, output, bias, bias_is_scalar_);
  } else {
    // `UseSplitK` only admits `is_channels_last`.
    MatMulWriteFnSourceForMatMul(shader, output, bias, apply_activation, /*is_channels_last*/ true);
  }

  shader.MainFunctionBody() << "  let output_components = " << output_components_ << ";\n";
  shader.MainFunctionBody() << R"(
  let output_id = i32(global_idx);

  let batch_size = i32(uniforms.batch_size);
  let dim_a_outer = i32(uniforms.dim_a_outer);
  let dim_b_outer = i32(uniforms.dim_b_outer) / output_components;
  let elements_per_batch = dim_a_outer * dim_b_outer;
  let out_elems = batch_size * elements_per_batch;
  if (output_id >= out_elems) {
    return;
  }

  let output_batch = output_id / elements_per_batch;
  let remaining = output_id % elements_per_batch;
  let output_row = remaining / dim_b_outer;
  let output_col = remaining % dim_b_outer;
)";

  // Accumulating in f32 for f16 output avoids rounding the running sum at every step.
  const std::string acc_type = MakeScalarOrVectorType(static_cast<int>(output_components_), "f32");
  shader.MainFunctionBody()
      << "  var acc = " << acc_type << "(0.0);\n"
      << "  for (var s = 0; s < i32(uniforms.splits); s++) {\n"
      << "    acc += " << acc_type << "(" << partials.GetByOffset("u32(s * out_elems + output_id)") << ");\n"
      << "  }\n"
      << "  mm_write(output_batch, output_row, output_col, output_value_t(acc));\n";

  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
