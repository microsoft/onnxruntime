// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/gemm_vec4.h"

namespace onnxruntime {
namespace webgpu {

void GemmVec4Program::MatMulReadFnSource(ShaderHelper& shader) const {
  shader.AdditionalImplementation()
      << R"READA(fn mm_readA(row: u32, col: u32, total_rows: u32, total_cols: u32) -> output_value_t {

    if(col < total_cols&& row < total_rows) {
        return A[row * total_cols+ col];
    } else {
        return output_value_t(0);
    }
}
)READA";

  shader.AdditionalImplementation()
      << R"READB(fn mm_readB(row: u32, col: u32, total_rows: u32, total_cols: u32) -> output_value_t {
    if(col < total_cols && row < total_rows) {
        return B[row * total_cols + col];
    } else {
        return output_value_t(0);
    }
    })READB";
}

void GemmVec4Program::MatMulWriteFnSource(ShaderHelper& shader, const ShaderVariableHelper& output) const {
  shader.AdditionalImplementation()
      << "fn mm_write(row: u32, col: u32, valuesIn: output_value_t) { \n"
      << "var values = valuesIn; \n"
      << "if(col < uniforms.N4 && row < uniforms.M) {";
  if (need_handle_bias_) {
    const ShaderVariableHelper& C = shader.AddInput("C", ShaderUsage::UseUniform);
    shader.AdditionalImplementation() << "    values += uniforms.beta * ";
    if (c_component_ == 1 && c_is_scalar_) {
      shader.AdditionalImplementation() << "output_value_t(C[0]);\n";
    } else if (c_component_ == 1) {
      shader.AdditionalImplementation() << "output_value_t(C[row]);\n";
    } else {
      shader.AdditionalImplementation() << C.GetByOffset(C.BroadcastedIndicesToOffset("vec2(row, col)", output)) << ";\n";
    }
  }
  shader.AdditionalImplementation() << "    output[row * uniforms.N4 + col] = values;\n"
                                    << "  }\n"
                                    << "}\n";
}

Status GemmVec4Program::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  shader.MainFunctionBody() << "  var values = output_value_t(0);\n\n"
                            << "  let tile_col_start = (workgroup_idx % uniforms.num_tile_n) * 8u;\n"
                            << "  let tile_row_start = (workgroup_idx / uniforms.num_tile_n) * 32u;\n";

  if (need_handle_matmul_) {
    shader.AddInput("A", ShaderUsage::UseUniform);
    shader.AddInput("B", ShaderUsage::UseUniform);

    MatMulReadFnSource(shader);

    // Add shared memory arrays for tiling
    shader.AdditionalImplementation() << "var<workgroup> tile_a: array<array<output_value_t, 8>, 32>;\n"
                                      << "var<workgroup> tile_b: array<array<output_value_t, 8>, 32>;\n\n";

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
        var row = tile_row_start + local_idx / 8u; // row is A index
        var col = k_start_a + (local_idx % 8u); //
        tile_a[local_idx / 8u][local_idx % 8u] = mm_readA(row, col, uniforms.M, uniforms.K4);
        )TILE_A";
    }
    // Load TILE_B
    if (transB_) {
      shader.MainFunctionBody() << R"TILE_B(
        row = tile_col_start * 4 + (local_idx / 8u);
        col = k_start_b + (local_idx % 8u); // col is B index
        // load 1 vec4 into tile_b
        tile_b[local_idx / 8u][local_idx % 8u] = mm_readB(row, col, uniforms.N, uniforms.K4);
        )TILE_B";
    } else {
      shader.MainFunctionBody() << R"TILE_B(
        row = k_start_b + (local_idx / 8u); // row is B index
        col = tile_col_start + (local_idx % 8u); // col is B index
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
        var a1 = output_value_t(0);
        var a2 = output_value_t(0);
        var a3 = output_value_t(0);
        var a4 = output_value_t(0);

        var b1 = output_value_t(0);
        var b2 = output_value_t(0);
        var b3 = output_value_t(0);
        var b4 = output_value_t(0);

        // Calculate 4 output for each thread
        // We read 32 vec4 from tile_a and 32 vec4 from tile_b in total.
        for (var i = 0u; i < 32; i = i + 4u) {
            a1 = tile_a[i][local_idx / 32u];
            a2 = tile_a[i + 1u][local_idx / 32u];
            a3 = tile_a[i + 2u][local_idx / 32u];
            a4 = tile_a[i + 3u][local_idx / 32u];
            b1 = tile_b[(local_idx % 8) * 4][i / 4u];
            b2 = tile_b[(local_idx % 8) * 4 + 1u][i / 4u];
            b3 = tile_b[(local_idx % 8) * 4 + 2u][i / 4u];
            b4 = tile_b[(local_idx % 8) * 4 + 3u][i / 4u];

            var vec_idx = local_idx / 8u % 4;

            values[0] += a1[vec_idx] * b1[0] + a2[vec_idx] * b1[1] + a3[vec_idx] * b1[2] + a4[vec_idx] * b1[3];
            values[1] += a1[vec_idx] * b2[0] + a2[vec_idx] * b2[1] + a3[vec_idx] * b2[2] + a4[vec_idx] * b2[3];
            values[2] += a1[vec_idx] * b3[0] + a2[vec_idx] * b3[1] + a3[vec_idx] * b3[2] + a4[vec_idx] * b3[3];
            values[3] += a1[vec_idx] * b4[0] + a2[vec_idx] * b4[1] + a3[vec_idx] * b4[2] + a4[vec_idx] * b4[3];
        }
        )CALC";
    } else if (transA_ && !transB_) {
      shader.MainFunctionBody() << R"CALC(
        var a = output_value_t(0);
        var b = output_value_t(0);

        // Calculate 4 output for each thread
        // We read 32 vec4 from tile_a and 32 vec4 from tile_b in total.
        for (var i = 0u; i < 32; i = i + 1u) {
            a = tile_a[i][local_idx / 32u];
            b = tile_b[i][local_idx % 8u];
            values += a[(local_idx / 8u) % 4] * b;
        })CALC";
    } else if (!transA_ && transB_) {
      // !transA_ && !transB_
      shader.MainFunctionBody() << R"CALC(
        var a = output_value_t(0);
        var b1 = output_value_t(0);
        var b2 = output_value_t(0);
        var b3 = output_value_t(0);
        var b4 = output_value_t(0);

         for (var i = 0u; i < 32; i = i + 4u) {
            a = tile_a[local_idx / 8u][i/4u];
            b1 = tile_b[(local_idx % 8) * 4][i / 4u];
            b2 = tile_b[(local_idx % 8) * 4 + 1u][i / 4u];
            b3 = tile_b[(local_idx % 8) * 4 + 2u][i / 4u];
            b4 = tile_b[(local_idx % 8) * 4 + 3u][i / 4u];

            values += vec4<f32>(
                dot(a, b1),
                dot(a, b2),
                dot(a, b3),
                dot(a, b4)
            );
        }
            )CALC";
    } else {
      shader.MainFunctionBody() << R"CALC(
        var a = output_value_t(0);
        var b1 = output_value_t(0);
        var b2 = output_value_t(0);
        var b3 = output_value_t(0);
        var b4 = output_value_t(0);
        for (var i = 0u; i < 32; i = i + 4u) {
            a = tile_a[local_idx / 8u][i/4u];
            b1 = tile_b[i][local_idx % 8u];
            b2 = tile_b[i+1][local_idx % 8u];
            b3 = tile_b[i+2][local_idx % 8u];
            b4 = tile_b[i+3][local_idx % 8u];

            values += a.x * b1 + a.y * b2 + a.z * b3 + a.w * b4;
        }
        )CALC";
    }
    shader.MainFunctionBody() << "    workgroupBarrier();\n"
                              << "  }\n\n";

    // Calculate alpha
    if (alpha_ != 1.0f) {
      shader.MainFunctionBody() << "  values = uniforms.alpha * values;\n";
    }
  }

  MatMulWriteFnSource(shader, output);
  shader.MainFunctionBody() << "  let m = tile_row_start + local_idx / 8u;\n"
                            << "  let n = tile_col_start + local_idx % 8u;\n\n";

  shader.MainFunctionBody() << " mm_write(m, n, values);\n";

  return Status::OK();
}

bool CanApplyGemmVec4(const Tensor* a,
                      const Tensor* b) {
  const auto& a_shape = a->Shape();
  const auto& b_shape = b->Shape();

  uint32_t A_rows = onnxruntime::narrow<uint32_t>(a_shape[0]);
  uint32_t A_cols = onnxruntime::narrow<uint32_t>(a_shape[1]);
  uint32_t B_rows = onnxruntime::narrow<uint32_t>(b_shape[0]);
  uint32_t B_cols = onnxruntime::narrow<uint32_t>(b_shape[1]);

  return A_rows % 4 == 0 && A_cols % 4 == 0 && B_rows % 4 == 0 && B_cols % 4 == 0;
}

Status ApplyGemmVec4(const Tensor* a,
                     const Tensor* b,
                     const Tensor* c,
                     bool transA_,
                     bool transB_,
                     float alpha,
                     float beta,
                     ComputeContext& context,
                     Tensor* y) {
  const auto& a_shape = a->Shape();
  const auto& b_shape = b->Shape();

  uint32_t M = onnxruntime::narrow<uint32_t>(transA_ ? a_shape[1] : a_shape[0]);
  uint32_t K = onnxruntime::narrow<uint32_t>(transA_ ? a_shape[0] : a_shape[1]);
  uint32_t N = onnxruntime::narrow<uint32_t>(transB_ ? b_shape[0] : b_shape[1]);

  // WebGPU doesn't support binding a zero-sized buffer, so we need to check if A or B is empty.
  bool need_handle_matmul = a_shape.Size() > 0 && b_shape.Size() > 0;
  bool need_handle_bias = c && beta;

  int c_component = 4;
  bool c_is_scalar = false;
  if (need_handle_bias) {
    const auto& c_shape = c->Shape();
    int64_t c_last_dim = c_shape[c_shape.NumDimensions() - 1];
    c_component = c_last_dim == N ? 4 : 1;
    c_is_scalar = c_shape.Size() == 1;
  }

  GemmVec4Program program{transA_, transB_, alpha, need_handle_bias, need_handle_matmul, c_component, c_is_scalar};

  const int components = 4;

  if (need_handle_matmul) {
    program.AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {b, ProgramTensorMetadataDependency::TypeAndRank, components}});
  }

  if (need_handle_bias) {
    program.AddInput({c, ProgramTensorMetadataDependency::TypeAndRank, c_component});
  }

  const uint32_t TILE_SIZE = 32;
  const uint32_t num_tile_n = (N + TILE_SIZE - 1) / TILE_SIZE;
  const uint32_t num_tile_m = (M + TILE_SIZE - 1) / TILE_SIZE;

  program.CacheHint(alpha, transA_, transB_)
      .AddOutputs({{y, ProgramTensorMetadataDependency::TypeAndRank, components}})
      .SetDispatchGroupSize(num_tile_n * num_tile_m)
      .SetWorkgroupSize(256, 1, 1)
      .AddUniformVariables({
          {num_tile_n},  // num_tile_n
          {M},           // M
          {N},           // N
          {K},           // K
          {M / 4},       // M4
          {N / 4},       // N4
          {K / 4},       // K4
          {alpha},       // alpha
          {beta}         // beta
      });

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
