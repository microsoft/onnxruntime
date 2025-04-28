// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/gemm_vec4.h"

#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace webgpu {

void GemmVec4Program::MatMulReadFnSource(ShaderHelper& shader) const {
  // We can’t treat `output_value_t` as the type of A and B, because output might not be a vec4, while A or B is.
  const std::string data_type = "output_element_t";
  const std::string type_string = MakeScalarOrVectorType(4 /*components */, data_type);

  shader.AdditionalImplementation()
      << "fn mm_readA(row: u32, col: u32, total_rows: u32, total_cols: u32) -> " << type_string << " { \n"
      << " if(col < total_cols && row < total_rows) {\n"
      << "    return A[row * total_cols + col];\n"
      << "  } else {\n"
      << "    return " << type_string << "(0);\n"
      << "  }\n"
      << "}\n\n";

  shader.AdditionalImplementation()
      << "fn mm_readB(row: u32, col: u32, total_rows: u32, total_cols: u32) -> " << type_string << "{ \n"
      << "  if(col < total_cols && row < total_rows) {\n"
      << "    return B[row * total_cols + col];\n"
      << "  } else {\n"
      << "    return " << type_string << "(0);\n"
      << "  }\n"
      << "}\n\n";
}

void GemmVec4Program::MatMulWriteFnSource(ShaderHelper& shader, const ShaderVariableHelper& output) const {
  shader.AdditionalImplementation()
      << "fn mm_write(row: u32, col: u32, valuesIn: output_value_t) { \n";

  if (output_components_ == 1) {
    shader.AdditionalImplementation() << "  let total_cols = uniforms.N; \n";
  } else {
    shader.AdditionalImplementation() << "  let total_cols = uniforms.N4; \n";
  }

  shader.AdditionalImplementation() << "var values = valuesIn; \n"
                                    << "if(col < total_cols && row < uniforms.M) { \n";
  if (need_handle_bias_) {
    const ShaderVariableHelper& C = shader.AddInput("C", ShaderUsage::UseUniform);
    shader.AdditionalImplementation() << "    values += output_element_t(uniforms.beta) * ";
    // We can be allowed to use broadcasting only when both components are equal.
    // There is only one case for c_components_ is not equal output_components_.
    // I.g. the former is `1` and the latter is `4`.
    // That means the shape of C is either {M,1} or {1,1}
    if (c_components_ == output_components_) {
      shader.AdditionalImplementation() << "output_value_t("
                                        << C.GetByOffset(C.BroadcastedIndicesToOffset("vec2(row, col)", output)) << ");\n";
    } else if (c_is_scalar_) {
      shader.AdditionalImplementation() << "output_value_t(C[0]);\n";
    } else {
      shader.AdditionalImplementation() << "output_value_t(C[row]);\n";
    }
  }
  shader.AdditionalImplementation() << "    output[row * total_cols + col] = values;\n"
                                    << "  }\n"
                                    << "}\n";
}

Status GemmVec4Program::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  // We can’t treat `output_value_t` as the type of A and B, because output might not be a vec4, while A or B is.
  const std::string data_type = "output_element_t";
  const std::string type_string = MakeScalarOrVectorType(4 /*components */, data_type);

  shader.MainFunctionBody() << "  var values = " << type_string << "(0);\n\n"
                            << "  let tile_col_start = (workgroup_idx % uniforms.num_tile_n) * 8u;\n"
                            << "  let tile_row_start = (workgroup_idx / uniforms.num_tile_n) * 32u;\n";

  if (need_handle_matmul_) {
    shader.AddInput("A", ShaderUsage::UseUniform);
    shader.AddInput("B", ShaderUsage::UseUniform);

    MatMulReadFnSource(shader);

    // Add shared memory arrays for tiling
    shader.AdditionalImplementation() << "var<workgroup> tile_a: array<array< " << type_string << ", 8 >, 32 >;\n "
                                      << "var<workgroup> tile_b: array<array< " << type_string << ", 8 >, 32 >;\n ";

    shader.MainFunctionBody()
        << "  var k_start_a = 0u;\n"
        << "  var k_start_b = 0u;\n\n"
        << "  let num_tiles = (uniforms.K + 32 - 1) / 32;\n";

    // Main loop for matrix multiplication
    shader.MainFunctionBody()
        << "  for (var t = 0u; t < num_tiles; t = t + 1u) {\n";
    // Load TILE_A
    if (transA_) {
      shader.MainFunctionBody() << R"TILE_A(
        var row = k_start_a + (local_idx / 8u);
        var col =  tile_row_start/4 + local_idx % 8u;
        tile_a[local_idx / 8u][local_idx % 8u] = mm_readA(row, col, uniforms.K, uniforms.M4);
        )TILE_A";
    } else {
      shader.MainFunctionBody() << R"TILE_A(
        var row = tile_row_start + local_idx / 8u;
        var col = k_start_a + (local_idx % 8u);
        tile_a[local_idx / 8u][local_idx % 8u] = mm_readA(row, col, uniforms.M, uniforms.K4);
        )TILE_A";
    }
    // Load TILE_B
    if (transB_) {
      shader.MainFunctionBody() << R"TILE_B(
        row = tile_col_start * 4 + (local_idx / 8u);
        col = k_start_b + (local_idx % 8u);
        // load 1 vec4 into tile_b
        tile_b[local_idx / 8u][local_idx % 8u] = mm_readB(row, col, uniforms.N, uniforms.K4);
        )TILE_B";
    } else {
      shader.MainFunctionBody() << R"TILE_B(
        row = k_start_b + (local_idx / 8u);
        col = tile_col_start + (local_idx % 8u);
        // load 1 vec4 into tile_b
        tile_b[local_idx / 8u][local_idx % 8u] = mm_readB(row, col, uniforms.K, uniforms.N4);
        )TILE_B";
    }

    shader.MainFunctionBody() << "    workgroupBarrier();\n\n";

    if (transA_) {
      shader.MainFunctionBody() << "k_start_a = k_start_a + 32u; \n";
    } else {
      shader.MainFunctionBody() << "k_start_a = k_start_a + 8u; \n";
    }

    if (transB_) {
      shader.MainFunctionBody() << "k_start_b = k_start_b + 8u; \n";
    } else {
      shader.MainFunctionBody() << "k_start_b = k_start_b + 32u; \n";
    }

    // Calculate output according to TILE_A and TILE_B
    if (transA_ && transB_) {
      shader.MainFunctionBody() << R"CALC(
        // Calculate 4 output for each thread
        // We read 32 vec4 from tile_a and 32 vec4 from tile_b in total.
        for (var i = 0u; i < 32; i = i + 4u) {
            let a1 = tile_a[i][local_idx / 32u];
            let a2 = tile_a[i + 1u][local_idx / 32u];
            let a3 = tile_a[i + 2u][local_idx / 32u];
            let a4 = tile_a[i + 3u][local_idx / 32u];
            let b1 = tile_b[(local_idx % 8) * 4][i / 4u];
            let b2 = tile_b[(local_idx % 8) * 4 + 1u][i / 4u];
            let b3 = tile_b[(local_idx % 8) * 4 + 2u][i / 4u];
            let b4 = tile_b[(local_idx % 8) * 4 + 3u][i / 4u];

            var vec_idx = local_idx / 8u % 4;

            values[0] += a1[vec_idx] * b1[0] + a2[vec_idx] * b1[1] + a3[vec_idx] * b1[2] + a4[vec_idx] * b1[3];
            values[1] += a1[vec_idx] * b2[0] + a2[vec_idx] * b2[1] + a3[vec_idx] * b2[2] + a4[vec_idx] * b2[3];
            values[2] += a1[vec_idx] * b3[0] + a2[vec_idx] * b3[1] + a3[vec_idx] * b3[2] + a4[vec_idx] * b3[3];
            values[3] += a1[vec_idx] * b4[0] + a2[vec_idx] * b4[1] + a3[vec_idx] * b4[2] + a4[vec_idx] * b4[3];
        }
        )CALC";
    } else if (transA_ && !transB_) {
      shader.MainFunctionBody() << R"CALC(
        // Calculate 4 output for each thread
        // We read 32 vec4 from tile_a and 32 vec4 from tile_b in total.
        for (var i = 0u; i < 32; i = i + 1u) {
            let a = tile_a[i][local_idx / 32u];
            let b = tile_b[i][local_idx % 8u];
            values += a[(local_idx / 8u) % 4] * b;
        })CALC";
    } else if (!transA_ && transB_) {
      shader.MainFunctionBody() << R"CALC(
         for (var i = 0u; i < 32; i = i + 4u) {
            let a = tile_a[local_idx / 8u][i/4u];
            let b1 = tile_b[(local_idx % 8) * 4][i / 4u];
            let b2 = tile_b[(local_idx % 8) * 4 + 1u][i / 4u];
            let b3 = tile_b[(local_idx % 8) * 4 + 2u][i / 4u];
            let b4 = tile_b[(local_idx % 8) * 4 + 3u][i / 4u];

            values += vec4<output_element_t>(
                dot(a, b1),
                dot(a, b2),
                dot(a, b3),
                dot(a, b4)
            );
        }
            )CALC";
    } else {
      shader.MainFunctionBody() << R"CALC(
        for (var i = 0u; i < 32; i = i + 4u) {
            let a = tile_a[local_idx / 8u][i/4u];
            let b1 = tile_b[i][local_idx % 8u];
            let b2 = tile_b[i+1][local_idx % 8u];
            let b3 = tile_b[i+2][local_idx % 8u];
            let b4 = tile_b[i+3][local_idx % 8u];

            values += a.x * b1 + a.y * b2 + a.z * b3 + a.w * b4;
        }
        )CALC";
    }
    shader.MainFunctionBody() << "    workgroupBarrier();\n"
                              << "  }\n\n";

    // Calculate alpha
    if (alpha_ != 1.0f) {
      shader.MainFunctionBody() << "  values = output_element_t(uniforms.alpha) * values;\n";
    }
  }

  MatMulWriteFnSource(shader, output);
  shader.MainFunctionBody() << "  let m = tile_row_start + local_idx / 8u;\n"
                            << "  let n = tile_col_start + local_idx % 8u;\n\n";

  // Write output
  if (output_components_ == 1) {
    shader.MainFunctionBody() << " for (var i = 0u; i < 4u; i = i + 1u) {\n"
                              << "    mm_write(m, 4 * n + i, values[i]);\n"
                              << "  }\n";
  } else {
    shader.MainFunctionBody() << "  mm_write(m, n, values);\n";
  }

  return Status::OK();
}

bool CanApplyGemmVec4(const Tensor* a,
                      const Tensor* b) {
  const auto& a_shape = a->Shape();
  const auto& b_shape = b->Shape();

  // When the number of columns in A and B is divisible by 4, we apply vec4 optimization to A and B.
  // However, this doesn't necessarily mean that C and Y will use vec4.
  // For example, C/output won't be vec4 if B is transposed and N is not divisible by 4.
  // Also, C won't use vec4 when it's a scalar.
  // The code would be simpler if we avoided vec4 optimization for C/output.
  // But to maximize performance, we still apply vec4 when possible — even though it adds some complexity.
  // I've added detailed comments explaining this logic.
  // See MatMulReadFnSource and MatMulWriteFnSource, especially the parts related to broadcasting.
  return a_shape[1] % 4 == 0 && b_shape[1] % 4 == 0;
}

Status ApplyGemmVec4(const Tensor* a,
                     const Tensor* b,
                     const Tensor* c,
                     bool transA,
                     bool transB,
                     float alpha,
                     float beta,
                     ComputeContext& context,
                     Tensor* y) {
  const auto& a_shape = a->Shape();
  const auto& b_shape = b->Shape();

  uint32_t M = onnxruntime::narrow<uint32_t>(transA ? a_shape[1] : a_shape[0]);
  uint32_t K = onnxruntime::narrow<uint32_t>(transA ? a_shape[0] : a_shape[1]);
  uint32_t N = onnxruntime::narrow<uint32_t>(transB ? b_shape[0] : b_shape[1]);

  // WebGPU doesn't support binding a zero-sized buffer, so we need to check if A or B is empty.
  bool need_handle_matmul = a_shape.Size() > 0 && b_shape.Size() > 0;
  bool need_handle_bias = c && beta;

  int c_components = 4;
  bool c_is_scalar = false;

  // We use vec4 for C when its last dimension equals N and N is divisible by 4.
  if (need_handle_bias) {
    const auto& c_shape = c->Shape();
    int64_t c_last_dim = c_shape[c_shape.NumDimensions() - 1];
    c_components = (c_last_dim == N && N % 4 == 0) ? 4 : 1;
    c_is_scalar = c_shape.Size() == 1;
  }

  // We use vec4 for Y when N is divisible by 4.
  const int output_components = N % 4 == 0 ? 4 : 1;

  GemmVec4Program program{transA, transB, alpha, need_handle_bias, need_handle_matmul, c_components, c_is_scalar, output_components};

  const int components = 4;

  if (need_handle_matmul) {
    program.AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {b, ProgramTensorMetadataDependency::TypeAndRank, components}});
  }

  if (need_handle_bias) {
    program.AddInput({c, ProgramTensorMetadataDependency::TypeAndRank, c_components});
  }

  const uint32_t TILE_SIZE = 32;
  const uint32_t num_tile_n = (N + TILE_SIZE - 1) / TILE_SIZE;
  const uint32_t num_tile_m = (M + TILE_SIZE - 1) / TILE_SIZE;

  program.CacheHint(alpha, transA, transB, c_is_scalar)
      .AddOutputs({{y, ProgramTensorMetadataDependency::TypeAndRank, output_components}})
      .SetDispatchGroupSize(num_tile_n * num_tile_m)
      .SetWorkgroupSize(256, 1, 1)
      .AddUniformVariables({{num_tile_n},
                            {M},
                            {N},
                            {K},
                            {M / 4},
                            {N / 4},
                            {K / 4},
                            {alpha},
                            {beta}});

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
