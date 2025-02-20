// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

Status SubgroupMatrixMatMulNBitsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  shader.AddInput("input_b", ShaderUsage::UseUniform);
  shader.AddInput("scales_b", ShaderUsage::UseUniform);
  shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  // tile/subtile sizes and work distribution are inspired from metal shaders in llama.cpp (kernel_mul_mm)
  // https://github.com/ggml-org/llama.cpp/blob/d04e7163c85a847bc61d58c22f2c503596db7aa8/ggml/src/ggml-metal/ggml-metal.metal#L6066
  shader.AdditionalImplementation() << R"ADDNL_FN(
        const tile_cols = 64;
        const tile_rows = 32;
        const tile_k = 32;
        const subtile_cols = 32;
        const subtile_rows = 16;
        const quantization_block_size = 32;
        alias compute_precision = output_element_t;

        var<workgroup> tile_A: array<compute_precision, tile_rows * tile_k>;       // 32 x 32 - RxC
        var<workgroup> tile_B: array<compute_precision, tile_cols * tile_k>;       // 64 x 32 - RxC
        var<workgroup> scratch: array<array<array<compute_precision, 64>, 4>, 4>;  // 64 * 4 * 4

        fn loadSHMA(tile_base: u32, k_idx: u32, row: u32, c_idx:u32) {
            let a_global = tile_base + row;
            if (a_global >= uniforms.M) {
                return;
            }
            // Each call loads 8 columns, starting at col.
            var col = c_idx * 8;
            // 128 threads need to load 32 x 32. 4 threads per row or 8 col per thread.
            for (var col_offset:u32 = 0; col_offset < 8; col_offset++)
            {
                tile_A[row * tile_k + col + col_offset] = compute_precision(input_a[a_global*uniforms.K + k_idx + col + col_offset]);
            }
        }

        fn loadSHMB(tile_base: u32, k_idx: u32, row: u32, c_idx: u32) {
            let b_global = tile_base + row;
            if (b_global >= uniforms.N) {
                return;
            }
            // Each call loads 16 columns, starting at col.
            var col = c_idx * 16;
            // 128 threads need to load 64 x 32. 2 threads per row or 16 col per thread.
            // Stored in column major fashion.
            let b_idx = u32((b_global*uniforms.K + k_idx + col)/8);
            let scale   = compute_precision(scales_b[(b_global*uniforms.K + k_idx + col)/quantization_block_size]);
            for (var step:u32 = 0; step < 2; step++)
            {
                var b_value = input_b[b_idx+step];
                var b_value_lower = (vec4<compute_precision>(unpack4xU8(b_value & 0x0F0F0F0Fu)) - vec4<compute_precision>(8)) * scale;
                var b_value_upper = (vec4<compute_precision>(unpack4xU8((b_value >> 4) & 0x0F0F0F0Fu)) - vec4<compute_precision>(8)) * scale;
                let tile_b_base = row * tile_k + col + step * 8;
                tile_B[tile_b_base]     = b_value_lower[0];
                tile_B[tile_b_base + 1] = b_value_upper[0];
                tile_B[tile_b_base + 2] = b_value_lower[1];
                tile_B[tile_b_base + 3] = b_value_upper[1];
                tile_B[tile_b_base + 4] = b_value_lower[2];
                tile_B[tile_b_base + 5] = b_value_upper[2];
                tile_B[tile_b_base + 6] = b_value_lower[3];
                tile_B[tile_b_base + 7] = b_value_upper[3];
            }
        }

        fn storeOutput(offset:u32, row: u32, col:u32, src_slot:u32, row_limit:i32) {
            if (row_limit > 0 && row < u32(row_limit))
            {
                output[offset + row * uniforms.N + col] = output_element_t(scratch[src_slot][0][row * 8 + col]);
                output[offset + row * uniforms.N + col + 8] = output_element_t(scratch[src_slot][1][row * 8 + col]);
                output[offset + row * uniforms.N + col + 16] = output_element_t(scratch[src_slot][2][row * 8 + col]);
                output[offset + row * uniforms.N + col + 24] = output_element_t(scratch[src_slot][3][row * 8 + col]);
                let col2 = col + 1;
                output[offset + row * uniforms.N + col2] = output_element_t(scratch[src_slot][0][row * 8 + col2]);
                output[offset + row * uniforms.N + col2 + 8] = output_element_t(scratch[src_slot][1][row * 8 + col2]);
                output[offset + row * uniforms.N + col2 + 16] = output_element_t(scratch[src_slot][2][row * 8 + col2]);
                output[offset + row * uniforms.N + col2 + 24] = output_element_t(scratch[src_slot][3][row * 8 + col2]);
            }
        }
    )ADDNL_FN";

  shader.MainFunctionBody() << R"MAIN_FN(
        let a_global_base = workgroup_id.y * tile_rows;
        let b_global_base = workgroup_id.x * tile_cols;

        let subtile_id =  u32(local_idx / sg_size);
        let subtile_idx = u32(subtile_id / 2);
        let subtile_idy = subtile_id % 2;
        let base_A = subtile_idy * subtile_rows;
        let base_B = subtile_idx * subtile_cols;

        var matC00: subgroup_matrix_result<compute_precision, 8, 8>;
        var matC01: subgroup_matrix_result<compute_precision, 8, 8>;
        var matC02: subgroup_matrix_result<compute_precision, 8, 8>;
        var matC03: subgroup_matrix_result<compute_precision, 8, 8>;
        var matC10: subgroup_matrix_result<compute_precision, 8, 8>;
        var matC11: subgroup_matrix_result<compute_precision, 8, 8>;
        var matC12: subgroup_matrix_result<compute_precision, 8, 8>;
        var matC13: subgroup_matrix_result<compute_precision, 8, 8>;
        for (var kidx: u32 = 0; kidx < uniforms.K; kidx += tile_k) {
            // Load Phase
            loadSHMA(a_global_base, kidx, local_idx/4, local_idx%4);
            loadSHMB(b_global_base, kidx, local_idx/2, local_idx%2);
            workgroupBarrier();

            for (var step: u32 = 0; step < tile_k; step+=8)
            {
                // Load to local memory phase
                let matrix_a_offset = subtile_idy * subtile_rows * tile_k + step;
                // Syntax: subgroupMatrixLoad src_ptr,src_offset,is_col_major,src_stride
                var matA0: subgroup_matrix_left<compute_precision, 8, 8> = subgroupMatrixLoad<subgroup_matrix_left<compute_precision, 8, 8>>(&tile_A, matrix_a_offset, false, tile_k);
                var matA1: subgroup_matrix_left<compute_precision, 8, 8> = subgroupMatrixLoad<subgroup_matrix_left<compute_precision, 8, 8>>(&tile_A, matrix_a_offset + 8 * tile_k, false, tile_k);

                // tile_B is stored as column major.
                // [col0-0:32][col1-0:32][col2-0:32]..[col63-0:32]
                var matrix_b_offset = subtile_idx * subtile_cols * tile_k + step;
                var matB0: subgroup_matrix_right<compute_precision, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<compute_precision, 8, 8>>(&tile_B, matrix_b_offset, true, tile_k);
                var matB1: subgroup_matrix_right<compute_precision, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<compute_precision, 8, 8>>(&tile_B, matrix_b_offset +  8 * tile_k, true, tile_k);
                var matB2: subgroup_matrix_right<compute_precision, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<compute_precision, 8, 8>>(&tile_B, matrix_b_offset + 16 * tile_k, true, tile_k);
                var matB3: subgroup_matrix_right<compute_precision, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<compute_precision, 8, 8>>(&tile_B, matrix_b_offset + 24 * tile_k, true, tile_k);

                // Compute Phase
                // Syntax: subgroupMatrixMultiplyAccumulate left, right, accumulate -> accumulate
                matC00 = subgroupMatrixMultiplyAccumulate(matA0, matB0, matC00);
                matC01 = subgroupMatrixMultiplyAccumulate(matA0, matB1, matC01);
                matC02 = subgroupMatrixMultiplyAccumulate(matA0, matB2, matC02);
                matC03 = subgroupMatrixMultiplyAccumulate(matA0, matB3, matC03);

                matC10 = subgroupMatrixMultiplyAccumulate(matA1, matB0, matC10);
                matC11 = subgroupMatrixMultiplyAccumulate(matA1, matB1, matC11);
                matC12 = subgroupMatrixMultiplyAccumulate(matA1, matB2, matC12);
                matC13 = subgroupMatrixMultiplyAccumulate(matA1, matB3, matC13);
            }

            workgroupBarrier();
        }

        // Write out
        // Write out top block
        subgroupMatrixStore(&scratch[subtile_id][0], 0, matC00, false, 8);
        subgroupMatrixStore(&scratch[subtile_id][1], 0, matC01, false, 8);
        subgroupMatrixStore(&scratch[subtile_id][2], 0, matC02, false, 8);
        subgroupMatrixStore(&scratch[subtile_id][3], 0, matC03, false, 8);
        workgroupBarrier();
        let row = u32(sg_id / 4);
        var col = u32(sg_id % 4) * 2;
        var matrix_c_offset = (a_global_base+base_A) * uniforms.N + b_global_base + base_B;
        var row_limit:i32 = i32(uniforms.M) - i32(a_global_base + base_A);
        storeOutput(matrix_c_offset, row, col, subtile_id, row_limit);
        workgroupBarrier();

        // Write out bottom block
        subgroupMatrixStore(&scratch[subtile_id][0], 0, matC10, false, 8);
        subgroupMatrixStore(&scratch[subtile_id][1], 0, matC11, false, 8);
        subgroupMatrixStore(&scratch[subtile_id][2], 0, matC12, false, 8);
        subgroupMatrixStore(&scratch[subtile_id][3], 0, matC13, false, 8);
        workgroupBarrier();
        matrix_c_offset = matrix_c_offset + 8 * uniforms.N;
        row_limit = i32(uniforms.M) - i32(a_global_base + base_A + 8);
        storeOutput(matrix_c_offset, row, col, subtile_id, row_limit);
    )MAIN_FN";

  return Status::OK();
}

Status ApplySubgroupMatrixMatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales,
                                      uint32_t M,
                                      uint32_t N,
                                      uint32_t K,
                                      onnxruntime::webgpu::ComputeContext& context,
                                      Tensor* y) {
  constexpr uint32_t kTileSizeA = 32;
  constexpr uint32_t kTileSizeB = 64;
  constexpr uint32_t kU32Components = 4;
  TensorShape y_shape{1, M, N};
  SubgroupMatrixMatMulNBitsProgram mul_program;
  mul_program.SetWorkgroupSize(128);
  mul_program.SetDispatchGroupSize(
      (N + kTileSizeB - 1) / kTileSizeB,
      (M + kTileSizeA - 1) / kTileSizeA, 1);
  mul_program.AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, gsl::narrow<int>(1)},
                         {b, ProgramTensorMetadataDependency::TypeAndRank, gsl::narrow<int>(kU32Components)},
                         {scales, ProgramTensorMetadataDependency::TypeAndRank, gsl::narrow<int>(1)}})
      .AddUniformVariables({{static_cast<uint32_t>(M)},
                            {static_cast<uint32_t>(N)},
                            {static_cast<uint32_t>(K)}})
      .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, y_shape, gsl::narrow<int>(1)});
  return context.RunProgram(mul_program);
}

bool CanApplySubgroupMatrixMatMulNBits(onnxruntime::webgpu::ComputeContext& context,
                                       uint64_t accuracy_level,
                                       uint32_t block_size,
                                       uint32_t batch_count,
                                       uint32_t N,
                                       uint32_t K,
                                       bool has_zero_points) {
#if !defined(__wasm__)
  const bool has_subgroup_matrix = context.Device().HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix);
#else
  const bool has_subgroup_matrix = false;
#endif
  // For now SubgroupMatrixMatMulNBits is only supported for accuracy level 4, because with Fp16 there are
  // some precision issues with subgroupMatrixMultiplyAccumulate. It is possible to support higher accuracy
  // by setting compute_precision to Fp32, but that will be slower. For 1K token prefill FP16 Phi 3.5 is around 5s,
  // FP322 is around 7s.
  return context.AdapterInfo().backendType == wgpu::BackendType::Metal &&
         has_subgroup_matrix &&
         accuracy_level == 4 &&
         block_size == 32 &&
         batch_count == 1 &&
         K % 32 == 0 &&
         N % 64 == 0 &&
         !has_zero_points;
}
}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
