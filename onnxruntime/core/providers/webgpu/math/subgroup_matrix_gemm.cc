// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(__wasm__)

#include "core/providers/webgpu/math/subgroup_matrix_gemm.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace webgpu {

Status SubgroupMatrixGemmProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  shader.AddInput("input_b", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  // Add common shader header
  shader.AdditionalImplementation() << R"(
const tile_cols = 64;
const tile_rows = 32;
const tile_k = 32;
const subtile_cols = 32;
const subtile_rows = 16;
alias compute_precision = output_value_t;

var<workgroup> tile_A: array<compute_precision, tile_rows * tile_k>;
// 32 x 32 - RxC
var<workgroup> tile_B: array<compute_precision, tile_cols * tile_k>;
// If B is transposed, 64 x 32 - RxC. Else 32 x 64 - RxC
var<workgroup> scratch: array<array<array<compute_precision, 64>, 4>, 4>;
// 64 * 4 * 4
)";

  if (transA_) {
    shader.AdditionalImplementation() << R"(
fn loadSHMA(tile_base: u32, k_idx: u32, row: u32, c_idx: u32) {
    var col = c_idx * 8;
    let a_global = tile_base + col;
    if (a_global >= uniforms.M) {
        return;
    }
    // 128 threads need to load 32 x 32. 4 threads per row or 8 col per thread
    for (var col_offset: u32 = 0; col_offset < 8; col_offset++) {
        tile_A[row * tile_k + col + col_offset] = input_a[(k_idx + row) * uniforms.M + a_global + col_offset];
    }
}
)";
  } else {
    shader.AdditionalImplementation() << R"(
fn loadSHMA(tile_base: u32, k_idx: u32, row: u32, c_idx: u32) {
    let a_global = tile_base + row;
    if (a_global >= uniforms.M) {
        return;
    }
    var col = c_idx * 8;
    // 128 threads need to load 32 x 32. 4 threads per row or 8 col per thread
    for (var col_offset: u32 = 0; col_offset < 8; col_offset++) {
        tile_A[row * tile_k + col + col_offset] = input_a[a_global * uniforms.K + k_idx + col + col_offset];
    }
}
)";
  }

  if (transB_) {
    shader.AdditionalImplementation() << R"(
fn loadSHMB(tile_base: u32, k_idx: u32, row: u32, c_idx: u32) {
    let b_global = tile_base + row;
    if (b_global >= uniforms.N) {
        return;
    }
    var col = c_idx * 16;
    // 128 threads need to load 32 x 64. 4 threads per row or 16 col per thread
    for (var col_offset: u32 = 0; col_offset < 16; col_offset++) {
        tile_B[row * tile_k + col + col_offset] = input_b[b_global * uniforms.K + k_idx + col + col_offset];
    }
}
)";
  } else {
    shader.AdditionalImplementation() << R"(
fn loadSHMB(tile_base: u32, k_idx: u32, row: u32, c_idx: u32) {
    var col = c_idx * 16;
    let b_global = tile_base + col;
    if (b_global >= uniforms.N) {
        return;
    }
    // 128 threads need to load 32 x 64. 4 threads per row or 16 col per thread
    for (var col_offset: u32 = 0; col_offset < 16; col_offset++) {
        tile_B[row * 64 + col + col_offset] = input_b[k_idx * uniforms.N + row * uniforms.N + b_global + col_offset];
    }
}
)";
  }

  // Generate storeOutput function dynamically
  shader.AdditionalImplementation() << R"(
fn storeOutput(offset: u32, row: u32, col: u32, src_slot: u32, row_limit: i32) {
    if (row_limit > 0 && row < u32(row_limit)) {
        var value0 = scratch[src_slot][0][row * 8 + col];
        var value1 = scratch[src_slot][1][row * 8 + col];
        var value2 = scratch[src_slot][2][row * 8 + col];
        var value3 = scratch[src_slot][3][row * 8 + col];

        var value0_2 = scratch[src_slot][0][row * 8 + col + 1];
        var value1_2 = scratch[src_slot][1][row * 8 + col + 1];
        var value2_2 = scratch[src_slot][2][row * 8 + col + 1];
        var value3_2 = scratch[src_slot][3][row * 8 + col + 1];
)";

  // Handle alpha scaling
  if (alpha_ != 1.0f) {
    shader.AdditionalImplementation() << R"(
        value0 = value0 * output_value_t(uniforms.alpha);
        value1 = value1 * output_value_t(uniforms.alpha);
        value2 = value2 * output_value_t(uniforms.alpha);
        value3 = value3 * output_value_t(uniforms.alpha);
        value0_2 = value0_2 * output_value_t(uniforms.alpha);
        value1_2 = value1_2 * output_value_t(uniforms.alpha);
        value2_2 = value2_2 * output_value_t(uniforms.alpha);
        value3_2 = value3_2 * output_value_t(uniforms.alpha);
)";
  }

  // Add bias computation if needed
  if (need_handle_bias_) {
    const ShaderVariableHelper& input_c = shader.AddInput("input_c", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
    shader.AdditionalImplementation() << "        value0 = value0 + output_value_t(uniforms.beta) * "
                                      << input_c.GetByOffset(input_c.BroadcastedIndicesToOffset("vec2((offset + row * uniforms.N + col) / uniforms.N, (offset + row * uniforms.N + col) % uniforms.N)", output)) << ";\n"
                                      << "        value1 = value1 + output_value_t(uniforms.beta) * "
                                      << input_c.GetByOffset(input_c.BroadcastedIndicesToOffset("vec2((offset + row * uniforms.N + col + 8) / uniforms.N, (offset + row * uniforms.N + col + 8) % uniforms.N)", output)) << ";\n"
                                      << "        value2 = value2 + output_value_t(uniforms.beta) * "
                                      << input_c.GetByOffset(input_c.BroadcastedIndicesToOffset("vec2((offset + row * uniforms.N + col + 16) / uniforms.N, (offset + row * uniforms.N + col + 16) % uniforms.N)", output)) << ";\n"
                                      << "        value3 = value3 + output_value_t(uniforms.beta) * "
                                      << input_c.GetByOffset(input_c.BroadcastedIndicesToOffset("vec2((offset + row * uniforms.N + col + 24) / uniforms.N, (offset + row * uniforms.N + col + 24) % uniforms.N)", output)) << ";\n"
                                      << "        value0_2 = value0_2 + output_value_t(uniforms.beta) * "
                                      << input_c.GetByOffset(input_c.BroadcastedIndicesToOffset("vec2((offset + row * uniforms.N + col + 1) / uniforms.N, (offset + row * uniforms.N + col + 1) % uniforms.N)", output)) << ";\n"
                                      << "        value1_2 = value1_2 + output_value_t(uniforms.beta) * "
                                      << input_c.GetByOffset(input_c.BroadcastedIndicesToOffset("vec2((offset + row * uniforms.N + col + 1 + 8) / uniforms.N, (offset + row * uniforms.N + col + 1 + 8) % uniforms.N)", output)) << ";\n"
                                      << "        value2_2 = value2_2 + output_value_t(uniforms.beta) * "
                                      << input_c.GetByOffset(input_c.BroadcastedIndicesToOffset("vec2((offset + row * uniforms.N + col + 1 + 16) / uniforms.N, (offset + row * uniforms.N + col + 1 + 16) % uniforms.N)", output)) << ";\n"
                                      << "        value3_2 = value3_2 + output_value_t(uniforms.beta) * "
                                      << input_c.GetByOffset(input_c.BroadcastedIndicesToOffset("vec2((offset + row * uniforms.N + col + 1 + 24) / uniforms.N, (offset + row * uniforms.N + col + 1 + 24) % uniforms.N)", output)) << ";\n";
  }

  shader.AdditionalImplementation() << R"(
        output[offset + row * uniforms.N + col] = value0;
        output[offset + row * uniforms.N + col + 8] = value1;
        output[offset + row * uniforms.N + col + 16] = value2;
        output[offset + row * uniforms.N + col + 24] = value3;

        output[offset + row * uniforms.N + col + 1] = value0_2;
        output[offset + row * uniforms.N + col + 1 + 8] = value1_2;
        output[offset + row * uniforms.N + col + 1 + 16] = value2_2;
        output[offset + row * uniforms.N + col + 1 + 24] = value3_2;
    }
}
)";

  // Add main function start
  shader.MainFunctionBody() << R"(
        let a_global_base = workgroup_id.y * tile_rows;
        let b_global_base = workgroup_id.x * tile_cols;

        let subtile_id = u32(local_idx / sg_size);
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
    )";

  // Add load phase
  if (transB_) {
    shader.MainFunctionBody() << R"(
        // Load Phase
        loadSHMA(a_global_base, kidx, local_idx / 4, local_idx % 4);
        loadSHMB(b_global_base, kidx, local_idx / 2, local_idx % 2);
)";
  } else {
    shader.MainFunctionBody() << R"(
        // Load Phase
        loadSHMA(a_global_base, kidx, local_idx / 4, local_idx % 4);
        loadSHMB(b_global_base, kidx, local_idx / 4, local_idx % 4);
)";
  }

  shader.MainFunctionBody() << R"(
        workgroupBarrier();

        for (var step: u32 = 0; step < tile_k; step += 8) {
            // Load to local memory phase
)";

  if (transA_) {
    shader.MainFunctionBody() << R"(            let matrix_a_offset = step * 8 * 4 + subtile_idy * subtile_rows;
            var matA0: subgroup_matrix_left<compute_precision, 8, 8> = subgroupMatrixLoad<subgroup_matrix_left<compute_precision, 8, 8>>(&tile_A, matrix_a_offset, true, tile_k);
            var matA1: subgroup_matrix_left<compute_precision, 8, 8> = subgroupMatrixLoad<subgroup_matrix_left<compute_precision, 8, 8>>(&tile_A, matrix_a_offset + 8, true, tile_k);
)";
  } else {
    shader.MainFunctionBody() << R"(            let matrix_a_offset = subtile_idy * subtile_rows * tile_k + step;
            var matA0: subgroup_matrix_left<compute_precision, 8, 8> = subgroupMatrixLoad<subgroup_matrix_left<compute_precision, 8, 8>>(&tile_A, matrix_a_offset, false, tile_k);
            var matA1: subgroup_matrix_left<compute_precision, 8, 8> = subgroupMatrixLoad<subgroup_matrix_left<compute_precision, 8, 8>>(&tile_A, matrix_a_offset + 8 * tile_k, false, tile_k);
)";
  }

  if (transB_) {
    shader.MainFunctionBody() << R"(            var matrix_b_offset = subtile_idx * subtile_cols * tile_k + step;

            var matB0: subgroup_matrix_right<compute_precision, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<compute_precision, 8, 8>>(&tile_B, matrix_b_offset, true, tile_k);
            var matB1: subgroup_matrix_right<compute_precision, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<compute_precision, 8, 8>>(&tile_B, matrix_b_offset + 8 * tile_k, true, tile_k);
            var matB2: subgroup_matrix_right<compute_precision, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<compute_precision, 8, 8>>(&tile_B, matrix_b_offset + 16 * tile_k, true, tile_k);
            var matB3: subgroup_matrix_right<compute_precision, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<compute_precision, 8, 8>>(&tile_B, matrix_b_offset + 24 * tile_k, true, tile_k);
)";
  } else {
    shader.MainFunctionBody() << R"(            var matrix_b_offset = step * 8 * 8 + subtile_idx * tile_k;

            var matB0: subgroup_matrix_right<compute_precision, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<compute_precision, 8, 8>>(&tile_B, matrix_b_offset, false, tile_k * 2);
            var matB1: subgroup_matrix_right<compute_precision, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<compute_precision, 8, 8>>(&tile_B, matrix_b_offset + 8 , false, tile_k * 2);
            var matB2: subgroup_matrix_right<compute_precision, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<compute_precision, 8, 8>>(&tile_B, matrix_b_offset + 16 , false, tile_k * 2);
            var matB3: subgroup_matrix_right<compute_precision, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<compute_precision, 8, 8>>(&tile_B, matrix_b_offset + 24 , false, tile_k * 2);
)";
  }

  shader.MainFunctionBody() << R"(
            // Compute Phase
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
)";

  shader.MainFunctionBody() << R"(
    // Write out top block
    subgroupMatrixStore(&scratch[subtile_id][0], 0, matC00, false, 8);
    subgroupMatrixStore(&scratch[subtile_id][1], 0, matC01, false, 8);
    subgroupMatrixStore(&scratch[subtile_id][2], 0, matC02, false, 8);
    subgroupMatrixStore(&scratch[subtile_id][3], 0, matC03, false, 8);
    workgroupBarrier();

    let row = u32(sg_id / 4);
    var col = u32(sg_id % 4) * 2;
    var matrix_c_offset = (a_global_base + base_A) * uniforms.N + b_global_base + base_B;

    var row_limit = i32(uniforms.M) - i32(a_global_base + base_A);
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
)";

  return Status::OK();
}

// std::tuple<architecture, backendType, M, N, K, subgroupMinSize, subgroupMaxSize>
static const std::tuple<std::string_view, wgpu::BackendType, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    metal_supported_subgroup_matrix_configs[] = {
        {"metal-3", wgpu::BackendType::Metal, 8, 8, 8, 4, 64}};

bool IsSubgroupMatrixConfigSupported(ComputeContext& context, int32_t number_type) {
  const wgpu::AdapterInfo& adapter_info = context.AdapterInfo();
  const wgpu::AdapterPropertiesSubgroupMatrixConfigs& subgroup_matrix_configs = context.SubgroupMatrixConfigs();

  // Convert number type to wgpu::SubgroupMatrixComponentType
  wgpu::SubgroupMatrixComponentType type;
  switch (number_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      type = wgpu::SubgroupMatrixComponentType::F16;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      type = wgpu::SubgroupMatrixComponentType::F32;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      type = wgpu::SubgroupMatrixComponentType::I32;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      type = wgpu::SubgroupMatrixComponentType::U32;
      break;
    default:
      return false;  // Unsupported type for WebGPU
  }

  for (auto& supported_config : metal_supported_subgroup_matrix_configs) {
    for (size_t i = 0; i < subgroup_matrix_configs.configCount; i++) {
      auto& subgroup_matrix_config = subgroup_matrix_configs.configs[i];
      auto&& config = std::make_tuple(adapter_info.architecture, adapter_info.backendType,
                                      subgroup_matrix_config.M, subgroup_matrix_config.N, subgroup_matrix_config.K,
                                      adapter_info.subgroupMinSize, adapter_info.subgroupMaxSize);
      if (config == supported_config &&
          type == subgroup_matrix_config.componentType &&
          type == subgroup_matrix_config.resultComponentType) {
        return true;
      }
    }
  }
  return false;
}

bool CanApplySubgroupMatrixGemm(ComputeContext& context, uint32_t K, uint32_t N, int32_t number_type) {
  bool has_subgroup_matrix = context.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix);
  bool is_subgroup_matrix_config_supported = IsSubgroupMatrixConfigSupported(context, number_type);

  return has_subgroup_matrix && is_subgroup_matrix_config_supported && K % 32 == 0 && N % 64 == 0;
}

Status ApplySubgroupMatrixGemm(const Tensor* a,
                               const Tensor* b,
                               const Tensor* c,
                               bool transA,
                               bool transB,
                               float alpha,
                               float beta,
                               ComputeContext& context) {
  const auto& a_shape = a->Shape();
  const auto& b_shape = b->Shape();

  uint32_t M = onnxruntime::narrow<uint32_t>(transA ? a_shape[1] : a_shape[0]);
  uint32_t K = onnxruntime::narrow<uint32_t>(transA ? a_shape[0] : a_shape[1]);
  uint32_t N = onnxruntime::narrow<uint32_t>(transB ? b_shape[0] : b_shape[1]);

  std::vector<int64_t> output_dims{M, N};
  auto* y = context.Output(0, output_dims);
  int64_t output_size = y->Shape().Size();

  if (output_size == 0) {
    return Status::OK();
  }

  bool need_handle_bias = c && beta != 0.0f;

  SubgroupMatrixGemmProgram program{transA, transB, alpha, need_handle_bias};

  program.AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank},
                     {b, ProgramTensorMetadataDependency::TypeAndRank}});

  if (need_handle_bias) {
    program.AddInput({c, ProgramTensorMetadataDependency::TypeAndRank});
  }

  const uint32_t tile_rows = 32;
  const uint32_t tile_cols = 64;
  const uint32_t num_tile_n = (N + tile_cols - 1) / tile_cols;
  const uint32_t num_tile_m = (M + tile_rows - 1) / tile_rows;

  program.CacheHint(alpha, transA, transB, need_handle_bias)
      .AddOutputs({{y, ProgramTensorMetadataDependency::TypeAndRank}})
      .SetDispatchGroupSize(num_tile_n, num_tile_m, 1)
      .SetWorkgroupSize(SubgroupMatrixGemmProgram::SUBGROUP_MATRIX_WORKGROUP_SIZE_X,
                        SubgroupMatrixGemmProgram::SUBGROUP_MATRIX_WORKGROUP_SIZE_Y,
                        SubgroupMatrixGemmProgram::SUBGROUP_MATRIX_WORKGROUP_SIZE_Z)
      .AddUniformVariables({{alpha},
                            {beta},
                            {M},
                            {N},
                            {K}});

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime

#endif
