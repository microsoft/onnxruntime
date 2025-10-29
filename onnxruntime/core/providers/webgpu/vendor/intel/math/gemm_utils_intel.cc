// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/math/matmul_utils.h"
#include "core/providers/webgpu/vendor/intel/math/gemm_utils_intel.h"

namespace onnxruntime {
namespace webgpu {

// Some helper functions to handle the bias for GEMM and MatMul,
// which are used in the MatMulWriteFnSource function.
namespace {

void HanldeMaybeHaveBiasForGEMM(ShaderHelper& shader,
                                const ShaderVariableHelper& output,
                                bool has_bias,
                                int c_components,
                                int output_components,
                                bool c_is_scalar) {
  shader.AdditionalImplementation() << "    let coords = vec2(u32(row), u32(colIn));\n";

  if (has_bias) {
    const ShaderVariableHelper& C = shader.AddInput("C", ShaderUsage::UseUniform);
    shader.AdditionalImplementation() << "    value += output_element_t(uniforms.beta) * ";
    // We can be allowed to use broadcasting only when both components are equal.
    // There is only one case for c_components_ is not equal output_components.
    // I.g. the former is `1` and the latter is `4`.
    // That means the shape of C is either {M,1} or {1,1}
    if (c_components == output_components) {
      shader.AdditionalImplementation() << "output_value_t("
                                        << C.GetByOffset(C.BroadcastedIndicesToOffset("vec2(u32(row), u32(colIn))", output)) << ");\n";
    } else if (c_is_scalar) {
      shader.AdditionalImplementation() << "output_value_t(C[0]);\n";
    } else {
      shader.AdditionalImplementation() << "output_value_t(C[row]);\n";
    }
  }
  shader.AdditionalImplementation() << "    " << output.SetByIndices("coords", "value") << "\n";
}

void HandleMaybeBiasForMatMul(ShaderHelper& shader,
                              const ShaderVariableHelper& output,
                              bool has_bias,
                              std::string activation_snippet,
                              bool is_channels_last) {
  shader.AdditionalImplementation() << "    let coords = vec3(u32(batch), u32(row), u32(colIn));\n";
  if (has_bias) {
    shader.AdditionalImplementation() << "    value = value + output_value_t(" << (is_channels_last ? "bias[colIn]" : "bias[row]") << ");\n";
  }
  shader.AdditionalImplementation() << "    " << activation_snippet << "\n"
                                    << output.SetByIndices("coords", "value") << "\n";
}

}  // namespace

void MatMulReadFnSourceIntel(ShaderHelper& shader,
                             const ShaderVariableHelper& a,
                             const ShaderVariableHelper& b,
                             const ShaderIndicesHelper* batch_dims,
                             bool transA,
                             bool transB,
                             bool is_vec4,
                             bool use_subgroup) {
  // Always read A with 1-component when using subgroup.
  int components = use_subgroup ? 1 : (is_vec4 ? 4 : 1);
  const std::string data_type = "output_element_t";
  std::string type_string = MakeScalarOrVectorType(components, data_type);

  shader.AdditionalImplementation()
      << "fn mm_readA(batch: i32, row: i32, colIn: i32 "
      << (batch_dims
              ? ", batch_indices: batch_dims_indices_t"
              : "")
      << ") -> " << type_string << " {\n"
      << "  var value = " << type_string << "(0);\n"
      << "  let col = colIn * " << components << ";\n";
  if (transA) {
    shader.AdditionalImplementation() << "  if(row < i32(uniforms.dim_inner) && col < i32(uniforms.dim_a_outer)) {\n";
  } else {
    shader.AdditionalImplementation() << "  if(row < i32(uniforms.dim_a_outer) && col < i32(uniforms.dim_inner)) {\n";
  }
  shader.AdditionalImplementation() << "    var a_indices: a_indices_t;\n";

  if (batch_dims) {
    shader.AdditionalImplementation() << ConvertOutputBatchIndicesToInputBatchIndices("a", a, a.Rank() - 2, batch_dims ? batch_dims->Rank() : 0, " batch_indices ") << "\n";
  }
  shader.AdditionalImplementation() << "    " << a.IndicesSet("a_indices", a.Rank() - 2, "u32(row)") << "\n"
                                    << "    " << a.IndicesSet("a_indices", a.Rank() - 1, "u32(colIn)") << "\n"
                                    << "    value = " << a.GetByIndices("a_indices") << ";\n"
                                    << "  }\n"
                                    << "  return value;\n"
                                    << "}\n\n";

  components = is_vec4 ? 4 : 1;
  type_string = MakeScalarOrVectorType(components, data_type);
  // Add the mm_readB function
  shader.AdditionalImplementation()
      << "fn mm_readB(batch: i32, row: i32, colIn: i32 "
      << (batch_dims
              ? ", batch_indices: batch_dims_indices_t"
              : "")
      << ") -> " << type_string << " {\n"
      << "  var value = " << type_string << "(0);\n"
      << "  let col = colIn * " << components << ";\n";

  if (transB) {
    shader.AdditionalImplementation() << "  if(row < i32(uniforms.dim_b_outer) && col < i32(uniforms.dim_inner)) {\n";
  } else {
    shader.AdditionalImplementation() << "  if(row < i32(uniforms.dim_inner) && col < i32(uniforms.dim_b_outer)) {\n";
  }

  shader.AdditionalImplementation() << "    var b_indices: b_indices_t;\n"
                                    << ConvertOutputBatchIndicesToInputBatchIndices("b", b, b.Rank() - 2, batch_dims ? batch_dims->Rank() : 0, "batch_indices")
                                    << "    " << b.IndicesSet("b_indices", b.Rank() - 2, "u32(row)") << "\n"
                                    << "    " << b.IndicesSet("b_indices", b.Rank() - 1, "u32(colIn)") << "\n"
                                    << "    value = " << b.GetByIndices("b_indices") << ";\n"
                                    << "  }\n"
                                    << "  return value;\n"
                                    << "}\n\n";
}

void MatMulWriteFnSourceIntel(ShaderHelper& shader,
                              const ShaderVariableHelper& output,
                              bool has_bias,
                              bool is_gemm,
                              int c_components,
                              int output_components,
                              bool c_is_scalar,
                              std::string activation_snippet,
                              bool is_channels_last) {
  shader.AdditionalImplementation()
      << "fn mm_write(batch: i32, row: i32, colIn: i32, valueIn: output_value_t) { \n";

  shader.AdditionalImplementation() << "  let col = colIn * " << output_components << ";\n";

  shader.AdditionalImplementation() << "  if(row < i32(uniforms.dim_a_outer) && col < i32(uniforms.dim_b_outer)) { \n"
                                    << "    var value = valueIn; \n";

  if (is_gemm) {
    HanldeMaybeHaveBiasForGEMM(shader, output, has_bias, c_components, output_components, c_is_scalar);
  } else {
    HandleMaybeBiasForMatMul(shader, output, has_bias, activation_snippet, is_channels_last);
  }

  shader.AdditionalImplementation()
      << "  }\n"
      << "}\n\n";
}

}  // namespace webgpu
}  // namespace onnxruntime
