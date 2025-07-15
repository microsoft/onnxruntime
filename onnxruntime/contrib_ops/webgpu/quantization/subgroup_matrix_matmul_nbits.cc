// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(__wasm__)
#include <tuple>

#include "contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits_common.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

constexpr std::string_view ComponentTypeName[] = {"unknown", "f32", "f16", "u32", "i32"};
template <std::size_t N>
constexpr bool ValidateComponentTypeName(const std::array<wgpu::SubgroupMatrixComponentType, N>& component_type) {
  bool matched = true;
  for (auto type : component_type) {
    switch (type) {
      case wgpu::SubgroupMatrixComponentType::F32:
        matched = ComponentTypeName[static_cast<uint32_t>(wgpu::SubgroupMatrixComponentType::F32)] == "f32";
        break;
      case wgpu::SubgroupMatrixComponentType::F16:
        matched = ComponentTypeName[static_cast<uint32_t>(wgpu::SubgroupMatrixComponentType::F16)] == "f16";
        break;
      case wgpu::SubgroupMatrixComponentType::U32:
        matched = ComponentTypeName[static_cast<uint32_t>(wgpu::SubgroupMatrixComponentType::U32)] == "u32";
        break;
      case wgpu::SubgroupMatrixComponentType::I32:
        matched = ComponentTypeName[static_cast<uint32_t>(wgpu::SubgroupMatrixComponentType::I32)] == "i32";
        break;
      default:
        return false;
    }

    if (!matched) {
      return matched;
    }
  }

  return matched;
}
static_assert(ValidateComponentTypeName<4>({wgpu::SubgroupMatrixComponentType::F32,
                                            wgpu::SubgroupMatrixComponentType::F16, wgpu::SubgroupMatrixComponentType::U32,
                                            wgpu::SubgroupMatrixComponentType::I32}),
              "The elements' sequence of ComponentTypeName array do not match wgpu::SubgroupMatrixComponentType");

// std::tuple<architecture, backendType, componentType, resultComponentType, M, N, K, subgroupMinSize, subgroupMaxSize>
static const std::tuple<std::string_view, wgpu::BackendType, wgpu::SubgroupMatrixComponentType, wgpu::SubgroupMatrixComponentType,
                        uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    intel_supported_subgroup_matrix_configs[] = {
        {"xe-2lpg", wgpu::BackendType::Vulkan, wgpu::SubgroupMatrixComponentType::F16, wgpu::SubgroupMatrixComponentType::F16, 8, 16, 16, 16, 32},
        {"xe-2lpg", wgpu::BackendType::Vulkan, wgpu::SubgroupMatrixComponentType::F16, wgpu::SubgroupMatrixComponentType::F32, 8, 16, 16, 16, 32}};

bool IsSubgroupMatrixConfigSupportedOnIntel(onnxruntime::webgpu::ComputeContext& context, int32_t& config_index) {
  const wgpu::AdapterInfo& adapter_info = context.AdapterInfo();
  const wgpu::AdapterPropertiesSubgroupMatrixConfigs& subgroup_matrix_configs = context.SubgroupMatrixConfigs();
  int32_t index = 0;
  for (auto& supported_config : intel_supported_subgroup_matrix_configs) {
    for (size_t i = 0; i < subgroup_matrix_configs.configCount; i++) {
      auto& subgroup_matrix_config = subgroup_matrix_configs.configs[i];
      auto&& config = std::make_tuple(adapter_info.architecture, adapter_info.backendType,
                                      subgroup_matrix_config.componentType, subgroup_matrix_config.resultComponentType,
                                      subgroup_matrix_config.M, subgroup_matrix_config.N, subgroup_matrix_config.K,
                                      adapter_info.subgroupMinSize, adapter_info.subgroupMaxSize);
      if (config == supported_config) {
        config_index = index;
        return true;
      }
    }
    index++;
  }
  return false;
}

// This program optimizes the layout of input matrix A(MxK) for SubgroupMatrixLoad, so that all elements of each
// subgroup matrix(mxk) are arranged continuously in memory.
// Take "M = 4, K = 4, m = 2, k = 2" as an example, the input matrix A is arranged in row-major order as follows:
// d00, d01, | d02, d03,
// d10, d11, | d12, d13,
// ---------------------
// d20, d21, | d22, d23,
// d30, d31, | d32, d33,
//
// The layout program rearranges the input matrix A to be in the following order:
// d00, d01,
// d10, d11,
// ---------
// d02, d03,
// d12, d13,
// ---------
// d20, d21,
// d30, d31,
// ---------
// d22, d23,
// d32, d33,
class LayoutProgram final : public Program<LayoutProgram> {
 public:
  LayoutProgram(uint32_t m, uint32_t k, std::string_view component_type) : Program{"SubgroupMatrixMatMulLayout"},
                                                                        m_(m), k_(k), component_type_(component_type) {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"M", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32});
 private:
    uint32_t m_;
    uint32_t k_;
    std::string_view component_type_;
};

Status LayoutProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseUniform);
  shader.AddOutput("output_a", ShaderUsage::UseUniform);
  shader.AdditionalImplementation() << "alias component_type = " << component_type_ << ";\n"
                                    << "const m_dim: u32 = " << m_ << ";\n"
                                    << "const k_dim: u32 = " << k_ << ";\n";

  shader.MainFunctionBody() << R"MAIN_FN(
  let M = uniforms.M;
  let K = uniforms.K;
  let in_offset = workgroup_id.x * m_dim * K + workgroup_id.y * k_dim;
  let out_offset = (workgroup_id.x * K / k_dim + workgroup_id.y) * m_dim * k_dim;

  // Syntax: subgroupMatrixLoad src_ptr, src_offset, is_col_major, src_stride
  var mat: subgroup_matrix_left<component_type, k_dim, m_dim> =
    subgroupMatrixLoad<subgroup_matrix_left<component_type, k_dim, m_dim>>(&input_a, in_offset, false, uniforms.K);
  subgroupMatrixStore(&output_a, out_offset, mat, false, k_dim);
  )MAIN_FN";
  return Status::OK();
}


Status GenerateShaderCodeOnIntel(ShaderHelper& shader, uint32_t nbits, int32_t config_index, bool has_zero_points) {
  auto& config = intel_supported_subgroup_matrix_configs[config_index];
  shader.AdditionalImplementation() << "alias component_type = " << ComponentTypeName[static_cast<uint32_t>(std::get<2>(config))] << ";\n"
                                    << "alias result_component_type = " << ComponentTypeName[static_cast<uint32_t>(std::get<3>(config))] << ";\n"
                                    << "const m_dim: u32 = " << std::get<4>(config) << ";\n"
                                    << "const n_dim: u32 = " << std::get<5>(config) << ";\n"
                                    << "const k_dim: u32 = " << std::get<6>(config) << ";\n";

  shader.AdditionalImplementation() << R"ADDNL_FN(
  const tile_cols: u32 = 64;
  const tile_rows: u32 = 64;
  const tile_k: u32 = 32;
  const subtile_rows: u32 = 8;
  const quantization_block_size: u32 = 32;

  var<workgroup> tile_B: array<component_type, tile_cols * tile_k>;       // 64 x 32 - RxC
  )ADDNL_FN" << GenerateZeroPointReadingCode(nbits, has_zero_points, "component_type");
  if (nbits == 4) {
    shader.AdditionalImplementation() << R"ADDNL_FN(
        fn loadSHMB(tile_base: u32, k_idx: u32, row: u32, c_idx: u32) {
            let b_global = tile_base + row;
            if (b_global >= uniforms.N) {
                return;
            }
            // Each call loads 8 columns, starting at col.
            let col = c_idx * 8;
            // 256 threads need to load 64 x 32. 4 threads per row or 8 col per thread.
            // Stored in column major fashion.
            let b_idx = u32((b_global * uniforms.K + k_idx + col) / 8);
            let scale = component_type(scales_b[(b_global * uniforms.K + k_idx + col) / quantization_block_size]);
            let zero = mm_read_zero(b_global, (k_idx + col) / quantization_block_size, uniforms.N, uniforms.zero_blocks_per_col);
            let b_value = input_b[b_idx];
            let b_value_lower = (vec4<component_type>(unpack4xU8(b_value & 0x0F0F0F0Fu)) - vec4<component_type>(zero)) * scale;
            let b_value_upper = (vec4<component_type>(unpack4xU8((b_value >> 4) & 0x0F0F0F0Fu)) - vec4<component_type>(zero)) * scale;
            let tile_b_base = row * tile_k + col;
            tile_B[tile_b_base]     = b_value_lower[0];
            tile_B[tile_b_base + 1] = b_value_upper[0];
            tile_B[tile_b_base + 2] = b_value_lower[1];
            tile_B[tile_b_base + 3] = b_value_upper[1];
            tile_B[tile_b_base + 4] = b_value_lower[2];
            tile_B[tile_b_base + 5] = b_value_upper[2];
            tile_B[tile_b_base + 6] = b_value_lower[3];
            tile_B[tile_b_base + 7] = b_value_upper[3];
        }
    )ADDNL_FN";
  } else {
    ORT_ENFORCE(nbits == 8, "Only 4/8 bits are supported for webgpu matmulnbits");
    shader.AdditionalImplementation() << R"ADDNL_FN(
        fn loadSHMB(tile_base: u32, k_idx: u32, row: u32, c_idx: u32) {
            let b_global = tile_base + row;
            if (b_global >= uniforms.N) {
                return;
            }
            // Each call loads 8 columns, starting at col.
            let col = c_idx * 8;
            // 256 threads need to load 64 x 32. 4 threads per row or 8 col per thread.
            // Stored in column major fashion.
            let b_idx = u32((b_global * uniforms.K + k_idx + col) / 8);
            let scale   = component_type(scales_b[(b_global * uniforms.K + k_idx + col) / quantization_block_size]);
            let zero = mm_read_zero(b_global, (k_idx + col) / quantization_block_size, uniforms.N, uniforms.zero_blocks_per_col);
            let b_value = input_b[b_idx];
            let b_value0 = (vec4<component_type>(unpack4xU8(b_value[0])) - vec4<component_type>(zero)) * scale;
            let b_value1 = (vec4<component_type>(unpack4xU8(b_value[1])) - vec4<component_type>(zero)) * scale;
            let tile_b_base = row * tile_k + col;
            tile_B[tile_b_base]     = b_value0[0];
            tile_B[tile_b_base + 1] = b_value0[1];
            tile_B[tile_b_base + 2] = b_value0[2];
            tile_B[tile_b_base + 3] = b_value0[3];
            tile_B[tile_b_base + 4] = b_value1[0];
            tile_B[tile_b_base + 5] = b_value1[1];
            tile_B[tile_b_base + 6] = b_value1[2];
            tile_B[tile_b_base + 7] = b_value1[3];
        }
    )ADDNL_FN";
  }

  shader.MainFunctionBody() << R"MAIN_FN(
        let a_global_base = workgroup_id.y * tile_rows;
        let b_global_base = workgroup_id.x * tile_cols;

        let subtile_id =  u32(local_idx / sg_size);
        let subtile_a_num_per_tensor_row = u32(uniforms.K / k_dim);
        let subtile_a_num_per_tile_col = u32(tile_rows / m_dim);
        let subtile_a_id = (workgroup_id.y * subtile_a_num_per_tile_col + subtile_id) * subtile_a_num_per_tensor_row;

        let subtile_a_size = m_dim * k_dim;
        var matrix_a_offset = subtile_a_id * subtile_a_size;

        var matC00: subgroup_matrix_result<result_component_type, n_dim, m_dim>;
        var matC01: subgroup_matrix_result<result_component_type, n_dim, m_dim>;
        var matC02: subgroup_matrix_result<result_component_type, n_dim, m_dim>;
        var matC03: subgroup_matrix_result<result_component_type, n_dim, m_dim>;
        for (var kidx: u32 = 0; kidx < uniforms.K; kidx += tile_k) {
            // Load Phase
            loadSHMB(b_global_base, kidx, local_idx / 4, local_idx % 4);
            workgroupBarrier();

            for (var step: u32 = 0; step < tile_k; step += k_dim)
            {
                // Load A from global memory.
                // Syntax: subgroupMatrixLoad src_ptr, src_offset, is_col_major, src_stride
                var matA0: subgroup_matrix_left<component_type, k_dim, m_dim> = subgroupMatrixLoad<subgroup_matrix_left<component_type, k_dim, m_dim>>(&input_a, matrix_a_offset, false, k_dim);
                matrix_a_offset += subtile_a_size;

                // Load B from shared local memory.
                // tile_B is stored as column major.
                // [col0-0:32][col1-0:32][col2-0:32]..[col63-0:32]
                var matrix_b_offset = step;
                var matB0: subgroup_matrix_right<component_type, n_dim, k_dim> = subgroupMatrixLoad<subgroup_matrix_right<component_type, n_dim, k_dim>>(&tile_B, matrix_b_offset, true, tile_k);
                var matB1: subgroup_matrix_right<component_type, n_dim, k_dim> = subgroupMatrixLoad<subgroup_matrix_right<component_type, n_dim, k_dim>>(&tile_B, matrix_b_offset + n_dim * tile_k, true, tile_k);
                var matB2: subgroup_matrix_right<component_type, n_dim, k_dim> = subgroupMatrixLoad<subgroup_matrix_right<component_type, n_dim, k_dim>>(&tile_B, matrix_b_offset + 2 * n_dim * tile_k, true, tile_k);
                var matB3: subgroup_matrix_right<component_type, n_dim, k_dim> = subgroupMatrixLoad<subgroup_matrix_right<component_type, n_dim, k_dim>>(&tile_B, matrix_b_offset + 3 * n_dim * tile_k, true, tile_k);

                // Compute Phase
                // Syntax: subgroupMatrixMultiplyAccumulate left, right, accumulate -> accumulate
                matC00 = subgroupMatrixMultiplyAccumulate(matA0, matB0, matC00);
                matC01 = subgroupMatrixMultiplyAccumulate(matA0, matB1, matC01);
                matC02 = subgroupMatrixMultiplyAccumulate(matA0, matB2, matC02);
                matC03 = subgroupMatrixMultiplyAccumulate(matA0, matB3, matC03);
            }
            workgroupBarrier();
        }

        // Write out
        let matrix_c_offset = (a_global_base) * uniforms.N + b_global_base;
        subgroupMatrixStore(&output, matrix_c_offset + subtile_id * m_dim * uniforms.N, matC00, false, uniforms.N);
        subgroupMatrixStore(&output, matrix_c_offset + subtile_id * m_dim * uniforms.N + n_dim, matC01, false, uniforms.N);
        subgroupMatrixStore(&output, matrix_c_offset + subtile_id * m_dim * uniforms.N + 2 * n_dim, matC02, false, uniforms.N);
        subgroupMatrixStore(&output, matrix_c_offset + subtile_id * m_dim * uniforms.N + 3 * n_dim, matC03, false, uniforms.N);
    )MAIN_FN";

  return Status::OK();
}

Status GenerateShaderCodeOnApple(ShaderHelper& shader, uint32_t nbits, bool has_zero_points) {
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
    )ADDNL_FN"
                                    << GenerateZeroPointReadingCode(nbits, has_zero_points, "compute_precision");
  if (nbits == 4) {
    shader.AdditionalImplementation() << R"ADDNL_FN(
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
            let scale = compute_precision(scales_b[(b_global*uniforms.K + k_idx + col)/quantization_block_size]);
            let zero = mm_read_zero(b_global, (k_idx + col) / quantization_block_size, uniforms.N, uniforms.zero_blocks_per_col);
            for (var step:u32 = 0; step < 2; step++)
            {
                var b_value = input_b[b_idx+step];
                var b_value_lower = (vec4<compute_precision>(unpack4xU8(b_value & 0x0F0F0F0Fu)) - vec4<compute_precision>(zero)) * scale;
                var b_value_upper = (vec4<compute_precision>(unpack4xU8((b_value >> 4) & 0x0F0F0F0Fu)) - vec4<compute_precision>(zero)) * scale;
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
    )ADDNL_FN";
  } else {
    ORT_ENFORCE(nbits == 8, "Only 4/8 bits are supported for webgpu matmulnbits");
    shader.AdditionalImplementation() << R"ADDNL_FN(
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
            let scale = compute_precision(scales_b[(b_global*uniforms.K + k_idx + col)/quantization_block_size]);
            let zero = mm_read_zero(b_global, (k_idx + col) / quantization_block_size, uniforms.N, uniforms.zero_blocks_per_col);
            for (var step:u32 = 0; step < 2; step++)
            {
                var b_value = input_b[b_idx+step];
                var b_value0 = (vec4<compute_precision>(unpack4xU8(b_value[0])) - vec4<compute_precision>(zero)) * scale;
                var b_value1 = (vec4<compute_precision>(unpack4xU8(b_value[1])) - vec4<compute_precision>(zero)) * scale;
                let tile_b_base = row * tile_k + col + step * 8;
                tile_B[tile_b_base]     = b_value0[0];
                tile_B[tile_b_base + 1] = b_value0[1];
                tile_B[tile_b_base + 2] = b_value0[2];
                tile_B[tile_b_base + 3] = b_value0[3];
                tile_B[tile_b_base + 4] = b_value1[0];
                tile_B[tile_b_base + 5] = b_value1[1];
                tile_B[tile_b_base + 6] = b_value1[2];
                tile_B[tile_b_base + 7] = b_value1[3];
            }
        }
    )ADDNL_FN";
  }
  shader.AdditionalImplementation() << R"ADDNL_FN(
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

Status SubgroupMatrixMatMulNBitsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  shader.AddInput("input_b", ShaderUsage::UseUniform);
  shader.AddInput("scales_b", ShaderUsage::UseUniform);
  if (has_zero_points_) {
    shader.AddInput("zero_points", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  if (!vendor_.compare("apple")) {
    return GenerateShaderCodeOnApple(shader, nbits_, has_zero_points_);
  } else if (!vendor_.compare("intel")) {
    return GenerateShaderCodeOnIntel(shader, nbits_, config_index_, has_zero_points_);
  } else {
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::NOT_IMPLEMENTED,
                  "onnxruntime does not support subgroup matrix on this verdor.");
  }
}

Status ApplySubgroupMatrixMatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales,
                                      const Tensor* zero_points,
                                      uint32_t M,
                                      uint32_t N,
                                      uint32_t K,
                                      uint32_t nbits,
                                      uint32_t zero_blocks_per_col,
                                      int32_t config_index,
                                      onnxruntime::webgpu::ComputeContext& context,
                                      Tensor* y) {
  const auto& config = intel_supported_subgroup_matrix_configs[config_index];
  const auto component_type = ComponentTypeName[static_cast<uint32_t>(std::get<2>(config))];
  const auto m = std::get<4>(config);
  const auto k = std::get<6>(config);

  // Optimize the layout of input matrix A(MxK) for SubgroupMatrixLoad.
  LayoutProgram layout_program{m, k, component_type};
  constexpr uint32_t kSubgroupSize = 32;
  layout_program.SetWorkgroupSize(kSubgroupSize);

  const auto dispatch_group_size_x = (M + m - 1) / m;
  ORT_ENFORCE(K % k == 0, "K must be a multiple of ", k);
  const auto dispatch_group_size_y = K / k;
  // Each workgroup will process one subgroup matrix of size m x k.
  layout_program.SetDispatchGroupSize(dispatch_group_size_x, dispatch_group_size_y, 1);

  TensorShape a_layout_shape{dispatch_group_size_x * m, K};
  Tensor a_layout = context.CreateGPUTensor(a->DataType(), a_layout_shape);
  layout_program.AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, 1}})
      .AddOutputs({{&a_layout, ProgramTensorMetadataDependency::Rank, a_layout.Shape(), 1}})
      .AddUniformVariables({{static_cast<uint32_t>(M)},
                            {static_cast<uint32_t>(K)}});
  ORT_RETURN_IF_ERROR(context.RunProgram(layout_program));

  uint32_t tile_size_a = 32;
  uint32_t work_group_size = 128;
  constexpr uint32_t kTileSizeB = 64;
  constexpr uint32_t kU32Components = 4;
  TensorShape y_shape{1, M, N};
  const bool has_zero_points = zero_points != nullptr;
  SubgroupMatrixMatMulNBitsProgram mul_program{nbits, config_index, context.AdapterInfo().vendor, has_zero_points};
  if (context.AdapterInfo().vendor == std::string_view{"intel"}) {
    tile_size_a = 64;
    work_group_size = 256;
  }
  mul_program.SetWorkgroupSize(work_group_size);
  mul_program.SetDispatchGroupSize(
      (N + kTileSizeB - 1) / kTileSizeB,
      (M + tile_size_a - 1) / tile_size_a, 1);
  mul_program.AddInputs({{&a_layout, ProgramTensorMetadataDependency::TypeAndRank, 1},
                         {b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(nbits == 4 ? kU32Components : 2 * kU32Components)},
                         {scales, ProgramTensorMetadataDependency::TypeAndRank, 1}})
      .AddUniformVariables({{static_cast<uint32_t>(M)},
                            {static_cast<uint32_t>(N)},
                            {static_cast<uint32_t>(K)},
                            {zero_blocks_per_col}})
      .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, y_shape, 1})
      .CacheHint(nbits, has_zero_points);
  if (has_zero_points) {
    mul_program.AddInput({zero_points, ProgramTensorMetadataDependency::None, {(zero_points->Shape().Size() + 3) / 4}, 4});
  }
  return context.RunProgram(mul_program);
}

bool CanApplySubgroupMatrixMatMulNBits(onnxruntime::webgpu::ComputeContext& context,
                                       uint64_t accuracy_level,
                                       uint32_t block_size,
                                       uint32_t batch_count,
                                       uint32_t N,
                                       uint32_t K,
                                       int32_t& config_index) {
  bool has_subgroup_matrix = context.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix);
  if (has_subgroup_matrix) {
    if (context.AdapterInfo().vendor == std::string_view{"apple"}) {
      // For now SubgroupMatrixMatMulNBits is only supported for accuracy level 4, because with Fp16 there are
      // some precision issues with subgroupMatrixMultiplyAccumulate. It is possible to support higher accuracy
      // by setting compute_precision to Fp32, but that will be slower. For 1K token prefill FP16 Phi 3.5 is around 5s,
      // FP32 is around 7s.
      has_subgroup_matrix = accuracy_level == 4;
    } else if (context.AdapterInfo().vendor == std::string_view{"intel"}) {
      has_subgroup_matrix = IsSubgroupMatrixConfigSupportedOnIntel(context, config_index);
    }
  }

  return has_subgroup_matrix &&
         block_size == 32 &&
         batch_count == 1 &&
         K % 32 == 0 &&
         N % 64 == 0;
}
}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime

#endif
