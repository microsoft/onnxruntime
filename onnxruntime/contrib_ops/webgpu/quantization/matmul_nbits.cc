// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <string_view>

#include "contrib_ops/webgpu/quantization/matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits_common.h"
#include "contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/dp4a_matmul_nbits.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

namespace {

constexpr unsigned int kMinMForTileOptimization = 4;

#define CEIL_DIV(numerator, denominator) \
  (((numerator) + (denominator) - 1) / (denominator))

}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    MatMulNBits,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", WebGpuSupportedFloatTypes())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulNBits);

Status MatMulNBitsWideTileProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias);
  shader.AddInput("input_b", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("scales", ShaderUsage::UseElementTypeAlias);
  if (has_zero_points_) {
    shader.AddInput("zero_points", ShaderUsage::UseElementTypeAlias);
  }
  shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);

  constexpr uint32_t KAVecSizeForBlock32 = 8;

  const uint32_t workgroup_size = WorkgroupSizeX() * WorkgroupSizeY();
  ORT_ENFORCE(tile_m_ == workgroup_size / 8, "tile_m must be workgroup_size / 8.");
  ORT_ENFORCE(tile_n_ == workgroup_size, "tile_n must be workgroup_size.");

  shader.AdditionalImplementation() << R"(
fn load_a(batch : u32, row : u32, col : u32) -> input_a_value_t {
  if (batch < uniforms.Batch && row < uniforms.M && col < uniforms.K_div_4) {
    let offset = batch * uniforms.M * uniforms.K_div_4 + row * uniforms.K_div_4 + col;
    return input_a[offset];
  }
  return input_a_value_t();
}

fn load_scale(row : u32, block_idx : u32) -> output_element_t {
  if (row < uniforms.N && block_idx < uniforms.K_div_32) {
    let offset = row * uniforms.K_div_32 + block_idx;
    return scales[offset];
  }
  return output_element_t();
}

fn mm_write_y(batch : u32, row : u32, col : u32, value : output_element_t) {
  if (batch < uniforms.Batch && row < uniforms.M && col < uniforms.N) {
    let offset = batch * uniforms.M * uniforms.N + row * uniforms.N + col;
    output[offset] = value;
  }
}
)";

  if (nbits_ == 4) {
    shader.AdditionalImplementation() << R"(
fn load_b(row : u32, block_idx : u32) -> vec4<input_b_element_t> {
  if (row < uniforms.N && block_idx < uniforms.K_div_32) {
    let offset = row * uniforms.K_div_32 + block_idx;
    return input_b[offset];
  }
  return vec4<input_b_element_t>();
}

// packed8xU4
fn dequantize(packed_data : u32, zero_point : output_element_t, scale : output_element_t) -> mat2x4<output_element_t> {
  let lower: vec4<u32> = unpack4xU8(packed_data & 0x0F0F0F0Fu);
  let upper: vec4<u32> = unpack4xU8((packed_data >> 4u) & 0x0F0F0F0Fu);

  let zero_matrix: mat2x4<output_element_t> = mat2x4<output_element_t>(
      zero_point, zero_point, zero_point, zero_point,
      zero_point, zero_point, zero_point, zero_point
  );

  var dequantized_values: mat2x4<output_element_t> = mat2x4<output_element_t>(
      output_element_t(lower[0]), output_element_t(upper[0]),
      output_element_t(lower[1]), output_element_t(upper[1]),
      output_element_t(lower[2]), output_element_t(upper[2]),
      output_element_t(lower[3]), output_element_t(upper[3])
  );

  dequantized_values = (dequantized_values - zero_matrix) * scale;
  return dequantized_values;
}
)";
  } else {
    ORT_ENFORCE(nbits_ == 8, "Only 4/8 bits are supported for webgpu matmulnbits");

    shader.AdditionalImplementation() << R"(
fn load_b(row : u32, block_idx : u32) -> array<vec2<u32>, 4> {
  if (row < uniforms.N && block_idx < uniforms.K_div_32) {
    let offset = row * uniforms.K_div_16 + 2 * block_idx;
    let b_data_0 = input_b[offset];
    let b_data_1 = input_b[offset + 1];

    let b_data = array<vec2<u32>, 4>(
      vec2<u32>(b_data_0[0], b_data_0[1]),
      vec2<u32>(b_data_0[2], b_data_0[3]),
      vec2<u32>(b_data_1[0], b_data_1[1]),
      vec2<u32>(b_data_1[2], b_data_1[3])
    );
    return b_data;
  }
  return array<vec2<u32>, 4>();
}

// 2x packed4xU8
fn dequantize(packed_data : vec2<u32>, zero_point : output_element_t, scale : output_element_t) -> mat2x4<output_element_t> {
  let lower: vec4<u32> = unpack4xU8(packed_data[0]);
  let upper: vec4<u32> = unpack4xU8(packed_data[1]);

  let zero_matrix: mat2x4<output_element_t> = mat2x4<output_element_t>(
      zero_point, zero_point, zero_point, zero_point,
      zero_point, zero_point, zero_point, zero_point
  );

  var dequantized_values: mat2x4<output_element_t> = mat2x4<output_element_t>(
      output_element_t(lower[0]), output_element_t(lower[1]),
      output_element_t(lower[2]), output_element_t(lower[3]),
      output_element_t(upper[0]), output_element_t(upper[1]),
      output_element_t(upper[2]), output_element_t(upper[3])
  );

  dequantized_values = (dequantized_values - zero_matrix) * scale;
  return dequantized_values;
}
)";
  }

  shader.AdditionalImplementation() << GenerateZeroPointReadingCode(nbits_, has_zero_points_);

  // declare const variables
  shader.AdditionalImplementation() << "\n"
                                    << "const KAVecSizeForBlock32 = " << KAVecSizeForBlock32 << "u;\n"
                                    << "const kTileM = " << tile_m_ << "u;\n"
                                    << "const kTileN = " << tile_n_ << "u;\n";

  // declare workgroup memory
  shader.AdditionalImplementation() << "\n"
                                    << "var<workgroup> a_data_tile: array<array<input_a_value_t, KAVecSizeForBlock32>, kTileM>;\n"
                                    << "\n";

  // main
  shader.MainFunctionBody() << R"MAIN_FN(
  let batch = workgroup_idx / (uniforms.num_M_tile * uniforms.num_N_tile);
  let row = ((workgroup_idx / uniforms.num_N_tile) % uniforms.num_M_tile) * kTileM;
  let col = (workgroup_idx % uniforms.num_N_tile) * kTileN;

  // Utilizing an f32 accumulator mitigated precision loss with minimal
  // performance impact compared to an f16 accumulator.
  var results : array<f32, kTileM>;
  for (var block_idx = 0u; block_idx < uniforms.K_div_32; block_idx++) {
    // Load `a` elements into workgroup memory, TileM x KAVecSizeForBlock32 (block32)
    let a_row_idx = local_idx / KAVecSizeForBlock32;
    let a_col_idx = local_idx % KAVecSizeForBlock32;
    a_data_tile[a_row_idx][a_col_idx] = load_a(batch, row + a_row_idx, block_idx * KAVecSizeForBlock32 + a_col_idx);
    workgroupBarrier();

    let b_row = col + local_idx;
    let scale = load_scale(b_row, block_idx);
    let zero_point = mm_read_zero(b_row, block_idx, uniforms.N, uniforms.zero_blocks_per_col);
    let b_data = load_b(b_row, block_idx);

    for (var b_idx = 0u; b_idx < 4u; b_idx++) {
      let b_dequantized = dequantize(b_data[b_idx], zero_point, scale);
      for (var m_idx = 0u; m_idx < kTileM; m_idx++) {
        let a_data0 = a_data_tile[m_idx][b_idx * 2u];
        let a_data1 = a_data_tile[m_idx][b_idx * 2u + 1u];

        results[m_idx] += f32(dot(a_data0, b_dequantized[0])) +
                          f32(dot(a_data1, b_dequantized[1]));
      }
    }
    workgroupBarrier();
  }

  // Write the results.
  for (var m_idx = 0u; m_idx < kTileM; m_idx++) {
    mm_write_y(batch, row + m_idx, col + local_idx, output_element_t(results[m_idx]));
  }
)MAIN_FN";

  return Status::OK();
}

// Apply similar idea with DP4AMatMulNBitsSmallMProgram algorithm.
Status MatMulNBitsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias);
  const auto& b = shader.AddInput("input_b");
  shader.AddInput("scales_b");
  if (has_zero_points_) {
    shader.AddInput("zero_points", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
  const uint32_t components_a = a.NumComponents();
  const uint32_t components_b = b.NumComponents() / 4;  // b is stored as uint32 which includs 4 uint8.
  constexpr uint32_t tile_size_k_vec = 16;
  uint32_t elements_in_value_b = components_b * (32 / nbits_);
  uint32_t tile_k_size = tile_size_k_vec * elements_in_value_b;
  const uint32_t a_length_per_tile = tile_k_size / components_a;

  shader.AdditionalImplementation() << "const a_length_per_tile = " << a_length_per_tile << "u;\n"
                                    << "const tile_size_k_vec = " << tile_size_k_vec << ";\n"
                                    << "const tile_size_k = " << tile_k_size << "u;\n"
                                    << "const tile_size = " << tile_size_ << "u;\n"
                                    << "const elements_in_value_b = " << elements_in_value_b << "u;\n"
                                    << "const sub_tile_count = " << WorkgroupSizeX() / tile_size_k_vec << "u;\n"
                                    << "const component_a = " << components_a << "u;\n"
                                    << "const component_b = " << components_b << "u;\n";
  shader.AdditionalImplementation() << R"ADDNL_FN(
  // Shared memory
  var<workgroup> tile_A : array<input_a_value_t, a_length_per_tile>;
  var<workgroup> inter_results: array<array<output_element_t, tile_size_k_vec>, tile_size>;
  fn loadSHMA(batch: u32, a_global: u32, kidx: u32, col: u32)
  {
    let k_offset = kidx / component_a + col;
    if (batch < uniforms.batch_count && k_offset < uniforms.K_of_a) {
      tile_A[col] = input_a[batch * uniforms.M * uniforms.K_of_a + a_global * uniforms.K_of_a + k_offset];
    } else {
      tile_A[col] = input_a_value_t(0);
    }
  }
)ADDNL_FN"
                                    << GenerateZeroPointReadingCode(nbits_, has_zero_points_);

  shader.MainFunctionBody() << R"MAIN_FN(
  let batch = workgroup_idx / (uniforms.M * uniforms.num_N_tile);
  let a_global = (workgroup_idx / uniforms.num_N_tile) % uniforms.M;
  let b_global_base = (workgroup_idx % uniforms.num_N_tile) * tile_size;

  let idx = local_idx % tile_size_k_vec;
  let idy = local_idx / tile_size_k_vec;

  for (var kidx = 0u; kidx < uniforms.K; kidx += tile_size_k)
  {
    for (var id = local_idx; id < a_length_per_tile; id += workgroup_size_x)
    {
      loadSHMA(batch, a_global, kidx, id);
    }
    workgroupBarrier();

    for (var local_row_offset = 0u; local_row_offset < tile_size; local_row_offset += sub_tile_count)
    {
      var b_global = b_global_base + local_row_offset + idy;
      var k_offset = kidx / elements_in_value_b + idx;
      if (b_global < uniforms.N && k_offset < uniforms.K_of_b)
      {
        let block_idx = (kidx + idx * elements_in_value_b) / uniforms.block_size;
        let scale_b = scales_b[b_global * uniforms.blocks_per_col + block_idx];
        let zero = mm_read_zero(b_global, block_idx, uniforms.N, uniforms.zero_blocks_per_col);
        var b_value = input_b[b_global * uniforms.K_of_b + k_offset];
)MAIN_FN";

  if (nbits_ == 4) {
    shader.MainFunctionBody() << R"MAIN_FN(
        var sum = output_element_t(0);
        var a_offset = idx * (8 / component_a) * component_b;
        for (var i = 0u; i < component_b; i++) {
          let b_value_lower = vec4<output_element_t>(unpack4xU8(b_value[i] & 0x0F0F0F0Fu)) - vec4<output_element_t>(zero);
          let b_value_upper = vec4<output_element_t>(unpack4xU8((b_value[i] >> 4) & 0x0F0F0F0Fu)) - vec4<output_element_t>(zero);
          let b0 = vec4<output_element_t>(b_value_lower[0], b_value_upper[0], b_value_lower[1], b_value_upper[1]) * scale_b;
          let b1 = vec4<output_element_t>(b_value_lower[2], b_value_upper[2], b_value_lower[3], b_value_upper[3]) * scale_b;
)MAIN_FN";
    switch (components_a) {
      case 1:
        shader.MainFunctionBody() << "          sum += dot(vec4<output_element_t>(tile_A[a_offset], tile_A[a_offset + 1], tile_A[a_offset + 2], tile_A[a_offset + 3]), b0) +"
                                     " dot(vec4<output_element_t>(tile_A[a_offset + 4], tile_A[a_offset + 5], tile_A[a_offset + 6], tile_A[a_offset + 7]), b1);\n"
                                     "          a_offset += 8;\n";
        break;
      case 2:
        shader.MainFunctionBody() << "          sum += dot(vec4<output_element_t>(tile_A[a_offset], tile_A[a_offset + 1]), b0) +"
                                     "dot(vec4<output_element_t>(tile_A[a_offset + 2], tile_A[a_offset + 3]), b1);\n"
                                     "          a_offset += 4;\n";
        break;
      case 4:
        shader.MainFunctionBody() << "          sum += dot(tile_A[a_offset], b0) + dot(tile_A[a_offset + 1], b1);\n"
                                     "          a_offset += 2;\n";
        break;
      default:
        break;
    }
    shader.MainFunctionBody() << "        }\n";
  } else {
    shader.MainFunctionBody() << R"MAIN_FN(
        var sum = output_element_t(0);
        var a_offset = idx * (4 / component_a) * component_b;
        for (var i = 0u; i < component_b; i++) {
          let b_value = (vec4<output_element_t>(unpack4xU8(b_value[i])) - vec4<output_element_t>(zero)) * scale_b;
)MAIN_FN";
    switch (components_a) {
      case 1:
        shader.MainFunctionBody() << "          sum += dot(vec4<output_element_t>(tile_A[a_offset], tile_A[a_offset + 1], tile_A[a_offset + 2], tile_A[a_offset + 3]), b_value);\n"
                                     "          a_offset += 4;\n";
        break;
      case 2:
        shader.MainFunctionBody() << "          sum += dot(vec4<output_element_t>(tile_A[a_offset], tile_A[a_offset + 1]), b_value);\n"
                                     "          a_offset += 2;\n";
        break;
      case 4:
        shader.MainFunctionBody() << "          sum += dot(tile_A[a_offset], b_value);\n"
                                     "          a_offset += 1;\n";
        break;
      default:
        break;
    }
    shader.MainFunctionBody() << "        }\n";
  }

  shader.MainFunctionBody() << R"MAIN_FN(
        inter_results[local_row_offset + idy][idx] += sum;
      }
    }
    workgroupBarrier();
  }

  if (batch >= uniforms.batch_count) {
    return;
  }

  if (local_idx < tile_size) {
    var output_value = output_element_t(0);
    for (var b = 0u; b < tile_size_k_vec; b++) {
      output_value += inter_results[local_idx][b];
    }
    let b_global =  b_global_base + local_idx;
    let output_idx = batch * uniforms.M * uniforms.N + a_global * uniforms.N + b_global;
    if (b_global < uniforms.N) {
      output[output_idx] = output_value;
    }
  }
)MAIN_FN";

  return Status::OK();
}

Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const Tensor* a = context.Input(0);
  const Tensor* b = context.Input(1);
  const Tensor* scales = context.Input(2);
  const Tensor* zero_points = context.Input(3);
  const Tensor* g_idx = context.Input(4);
  const Tensor* bias = context.Input(5);

  ORT_ENFORCE(g_idx == nullptr, "group_idx as input is not supported yet.");
  ORT_ENFORCE(bias == nullptr, "bias as input is not supported yet.");

  const bool has_zero_points = zero_points != nullptr;
  if (has_zero_points) {
    ORT_ENFORCE(zero_points->DataType() == DataTypeImpl::GetType<uint8_t>(), "Currently, only uint8 is supported for zero points, but got ", zero_points->DataType());
  }

  MatMulComputeHelper helper;
  TensorShape b_shape({N_, K_});
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));
  auto* y = context.Output(0, helper.OutputShape());
  const uint32_t data_size = onnxruntime::narrow<uint32_t>(y->Shape().Size());
  if (data_size == 0) {
    return Status::OK();
  }

  const uint32_t batch_count = onnxruntime::narrow<uint32_t>(helper.OutputOffsets().size());
  const uint32_t M = onnxruntime::narrow<uint32_t>(helper.M());
  const uint32_t N = onnxruntime::narrow<uint32_t>(helper.N());
  const uint32_t K = onnxruntime::narrow<uint32_t>(helper.K());
  const uint32_t block_size = onnxruntime::narrow<uint32_t>(block_size_);
  const uint32_t nbits = onnxruntime::narrow<uint32_t>(bits_);

  const uint32_t n_blocks_per_col = (K + block_size - 1) / block_size;
  const uint32_t blob_size = (block_size / 8) * nbits;
  const uint32_t blob_size_in_words = blob_size / 4;
  const uint32_t components_a = GetMaxComponents(K);
  const uint32_t components_b = GetMaxComponents(blob_size_in_words);
  uint32_t components = GetMaxComponents(N);
  // zero_points has shape[N * CeilDiv(n_blocks_per_col * bits, 8)]. So here we need to check whether n_blocks_per_col is divisible by 8/nbits.
  // For bits==4, this is counted by elements of uint4. Need add 1 if not divisible by 2.
  uint32_t zero_blocks_per_col = n_blocks_per_col % (8 / nbits) == 0 ? n_blocks_per_col : n_blocks_per_col + 1;

#if !defined(__wasm__)
  int32_t subgroup_matrix_config_index = -1;
  // apple|intel - Experimental dawn support for subgroup matrix matmul.
  if (M >= kMinMForTileOptimization && (context.AdapterInfo().vendor == std::string_view{"apple"} || context.AdapterInfo().vendor == std::string_view{"intel"}) &&
      CanApplySubgroupMatrixMatMulNBits(context, accuracy_level_, block_size, batch_count, N, K, subgroup_matrix_config_index)) {
    return ApplySubgroupMatrixMatMulNBits(a, b, scales, zero_points, M, N, K, nbits, zero_blocks_per_col, subgroup_matrix_config_index, context, y);
  }
#endif

  // On FP32 only GPUs, integer math is faster than FP32 therefore always use DP4A independent of length of M.
  if ((M >= kMinMForTileOptimization || y->DataType() == DataTypeImpl::GetType<float>() ||
       context.AdapterInfo().vendor == std::string_view{"qualcomm"}) &&
      CanApplyDP4AMatrixMatMulNBits(context, accuracy_level_, block_size, batch_count, N, K, components_a)) {
    return ApplyDP4AMatrixMatMulNBits(a, b, scales, zero_points, M, N, K, block_size, zero_blocks_per_col, kMinMForTileOptimization, nbits, context, y);
  }

  // WideTileProgram
  // This program is optimized for Block32 prefill using Tile16x128.
  const bool use_wide_tile_program = block_size == 32 && components_a == 4 && components_b == 4 && M >= kMinMForTileOptimization;
  if (use_wide_tile_program) {
    // Enforce output components to 1.
    components = 1;

    constexpr uint32_t workgroup_size = 128;
    constexpr uint32_t tile_m = workgroup_size / 8;
    constexpr uint32_t tile_n = workgroup_size;
    const uint32_t num_N_tile = CEIL_DIV(N, tile_n);
    const uint32_t num_M_tile = CEIL_DIV(M, tile_m);

    MatMulNBitsWideTileProgram program{has_zero_points, tile_m, tile_n, nbits};
    program.SetWorkgroupSize(workgroup_size);
    program.SetDispatchGroupSize(num_N_tile, num_M_tile, batch_count);

    TensorShape reshaped_a_shape{batch_count, M, CEIL_DIV(K, components_a)};
    TensorShape reshaped_b_shape{N, n_blocks_per_col, CEIL_DIV(blob_size_in_words, components_b)};
    TensorShape reshaped_y_shape{batch_count, M, CEIL_DIV(N, components)};

    program
        .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, reshaped_a_shape, onnxruntime::narrow<int>(components_a)},
                    {b, ProgramTensorMetadataDependency::TypeAndRank, reshaped_b_shape, onnxruntime::narrow<int>(components_b * 4)},
                    {scales, ProgramTensorMetadataDependency::None}})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, reshaped_y_shape, onnxruntime::narrow<int>(components)})
        .AddUniformVariables({{batch_count}, {M}, {N}, {K}, {CEIL_DIV(K, 4)}, {CEIL_DIV(K, 16)}, {CEIL_DIV(K, 32)}, {zero_blocks_per_col}, {num_N_tile}, {num_M_tile}})
        .CacheHint(nbits, has_zero_points);
    if (has_zero_points) {
      program.AddInput({zero_points, ProgramTensorMetadataDependency::None, {CEIL_DIV(zero_points->Shape().Size(), 4)}, 4});
    }

    return context.RunProgram(program);
  }

  constexpr uint32_t workgroup_size = 128;
  constexpr uint32_t tile_size = 8;
  constexpr uint32_t kU32Components = 4;
  uint32_t components_b_with_u32 = components_b * kU32Components;
  uint32_t num_N_tile = (N + tile_size - 1) / tile_size;
  MatMulNBitsProgram program{tile_size, nbits, has_zero_points};
  program.SetWorkgroupSize(workgroup_size);
  program.SetDispatchGroupSize((N + tile_size - 1) / tile_size, M, batch_count);
  program
      .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_a)},
                  {b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_b_with_u32)},
                  {scales, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank})
      .AddUniformVariables({{M}, {N}, {K}, {K / components_a}, {n_blocks_per_col * blob_size / components_b_with_u32}, {block_size}, {n_blocks_per_col}, {zero_blocks_per_col}, {num_N_tile}, {batch_count}})
      .CacheHint(nbits, has_zero_points);
  if (has_zero_points) {
    program.AddInput({zero_points, ProgramTensorMetadataDependency::None, {(zero_points->Shape().Size() + 3) / 4}, 4});
  }
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
