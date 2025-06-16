// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <string_view>

#include "contrib_ops/webgpu/quantization/matmul_nbits.h"
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

std::string ReadZeroPoint(uint32_t nbits, bool has_zero_points) {
  ORT_ENFORCE(nbits == 8 || nbits == 4, "Only 4/8 bits are supported for webgpu matmulnbits");
  std::stringstream ss;
  if (has_zero_points) {
    ss << "const elements_in_uint32 = " << (32 / nbits) << "u;\n"
       << "const bits = " << nbits << "u;\n";
    ss << R"(
fn mm_read_zero(row : u32, col : u32, r_dim: u32, c_dim: u32) -> output_element_t {
  if (row < r_dim && col < c_dim) {
    let offset = row * c_dim + col;

    // u32 holds elements_in_uint32 packed nbits.
    let array_index = offset / elements_in_uint32;
    let component_index = offset % elements_in_uint32;
    let packed_value = zero_points[array_index];

    // Extract the nbits component
    let shift_amount = component_index * bits;
)";
    ss << "    let masked_value = (packed_value >> shift_amount) & " << (nbits == 4 ? "0xFu" : "0xFF") << ";\n";
    ss << R"(
    return output_element_t(masked_value);
  }
  return output_element_t(0);
}
)";
  } else {
    ss << "const default_zero_point = " << (nbits == 4 ? 8 : 128) << ";\n";
    ss << R"(
fn mm_read_zero(row : u32, col : u32, r_dim: u32, c_dim: u32) -> output_element_t {
  return output_element_t(default_zero_point);
}
)";
  }
  return ss.str();
}

constexpr unsigned int kMinMForTileOptimization = 4;
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
  const auto& a = shader.AddInput("input_a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& b = shader.AddInput("input_b", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  shader.AddInput("scales", ShaderUsage::UseUniform);
  if (has_zero_points_) {
    shader.AddInput("zero_points", ShaderUsage::UseUniform);
  }
  const auto& y = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias | ShaderUsage::UseIndicesTypeAlias);

  // Bock size 32, `a` component size 4, 8 `a` components per block.
  constexpr uint32_t kAComponentsForBlock32 = 8;

  const uint32_t workgroup_size = WorkgroupSizeX() * WorkgroupSizeY();
  ORT_ENFORCE(tile_m_ == workgroup_size / 8, "tile_m must be workgroup_size / 8.");
  ORT_ENFORCE(tile_n_ == workgroup_size, "tile_n must be workgroup_size.");

  // memory read/write helpers
  shader.AdditionalImplementation() << "fn mm_read_a(batch : u32, row : u32, col : u32) -> input_a_value_t {\n"
                                    << "  if (batch < uniforms.input_a_shape[0] && row < uniforms.input_a_shape[1] && col < uniforms.input_a_shape[2]) {\n"
                                    << "    return " << a.GetByIndices("input_a_indices_t(batch, row, col)") << ";\n"
                                    << "  }\n"
                                    << "  return input_a_value_t(0);\n"
                                    << "}\n";
  if (nbits_ == 4) {
    shader.AdditionalImplementation() << "\n"
                                      << "fn mm_read_b(row : u32, col : u32) -> input_b_value_t {\n"
                                      << "  if (row < uniforms.input_b_shape[0] && col < uniforms.input_b_shape[1]) {\n"
                                      << "    return " << b.GetByIndices("input_b_indices_t(row, col, 0)") << ";\n"
                                      << "  }\n"
                                      << "  return input_b_value_t(0);\n"
                                      << "}\n";

    shader.AdditionalImplementation() << R"(
fn dequantize_packed8xU4(packed_value : u32, zero_point : output_element_t, scale : output_element_t) -> mat2x4<output_element_t> {
  let lower_values: vec4<u32> = unpack4xU8(packed_value & 0x0F0F0F0Fu);
  let upper_values: vec4<u32> = unpack4xU8((packed_value >> 4u) & 0x0F0F0F0Fu);

  let zero_matrix: mat2x4<output_element_t> = mat2x4<output_element_t>(
      zero_point, zero_point, zero_point, zero_point,
      zero_point, zero_point, zero_point, zero_point
  );

  var dequantized_values: mat2x4<output_element_t> = mat2x4<output_element_t>(
      output_element_t(lower_values[0]), output_element_t(upper_values[0]),
      output_element_t(lower_values[1]), output_element_t(upper_values[1]),
      output_element_t(lower_values[2]), output_element_t(upper_values[2]),
      output_element_t(lower_values[3]), output_element_t(upper_values[3])
  );

  dequantized_values = (dequantized_values - zero_matrix) * scale;
  return dequantized_values;
}
)";
  }

  shader.AdditionalImplementation() << "\n"
                                    << "fn mm_read_scale(row : u32, col : u32) -> output_element_t {\n"
                                    << "  if (row < uniforms.input_b_shape[0] && col < uniforms.input_b_shape[1]) {\n"
                                    << "    return scales[row * uniforms.input_b_shape[1] + col];\n"
                                    << "  }\n"
                                    << "  return output_element_t(0);\n"
                                    << "}\n"
                                    << ReadZeroPoint(nbits_, has_zero_points_);

  shader.AdditionalImplementation() << "\n"
                                    << "fn mm_write_y(batch : u32, row : u32, col : u32, value : output_value_t) {\n"
                                    << "  if (row < uniforms.output_shape[1] && col < uniforms.output_shape[2]) {\n"
                                    << "    " << y.SetByIndices("output_indices_t(batch, row, col)", "value") << "\n"
                                    << "  }\n"
                                    << "}\n";

  // declare const variables
  shader.AdditionalImplementation() << "\n"
                                    << "// A block32 containing 8 components of `a`." << "\n"
                                    << "const kAComponentsForBlock32 = " << kAComponentsForBlock32 << "u;\n"
                                    << "const kTileM = " << tile_m_ << "u;\n"
                                    << "const kTileN = " << tile_n_ << "u;\n";

  // declare workgroup memory
  shader.AdditionalImplementation() << "\n"
                                    << "var<workgroup> a_data_tile: array<array<input_a_value_t, kAComponentsForBlock32>, kTileM>;\n"
                                    << "\n";

  // main
  shader.MainFunctionBody() << R"MAIN_FN(
  let batch = workgroup_idx / (uniforms.num_M_tile * uniforms.num_N_tile);
  let row = ((workgroup_idx / uniforms.num_N_tile) % uniforms.num_M_tile) * kTileM;
  let col = (workgroup_idx % uniforms.num_N_tile) * kTileN;

  let a_elements_per_col = uniforms.input_a_shape[2];
  let a_blocks_per_col = (a_elements_per_col + kAComponentsForBlock32 - 1) / kAComponentsForBlock32;

  // Utilizing an f32 accumulator mitigated precision loss with minimal
  // performance impact compared to an f16 accumulator.
  var results : array<f32, kTileM>;
  for (var a_block_idx = 0u; a_block_idx < a_blocks_per_col; a_block_idx++) {
    // Load `a` elements into workgroup memory, TileM x kAComponentsForBlock32 (block32)
    let a_row_idx = local_idx / kAComponentsForBlock32;
    let a_col_idx = local_idx % kAComponentsForBlock32;
    a_data_tile[a_row_idx][a_col_idx] = mm_read_a(batch, row + a_row_idx, a_block_idx * kAComponentsForBlock32 + a_col_idx);
    workgroupBarrier();

    let b_row = col + local_idx;
    let b_col = a_block_idx;

    let scale = mm_read_scale(b_row, b_col);
    let zero_point = mm_read_zero(b_row, b_col, uniforms.input_b_shape[0], uniforms.zero_blocks_per_col);
)MAIN_FN";

  if (nbits_ == 4) {
    shader.MainFunctionBody() << R"MAIN_FN(
    let b_data = mm_read_b(b_row, b_col);
    // `b` component size is 4.
    for (var b_idx = 0u; b_idx < 4u; b_idx++) {
      let b_dequantized = dequantize_packed8xU4(b_data[b_idx], zero_point, scale);
      for (var m_idx = 0u; m_idx < kTileM; m_idx++) {
        let a_data0 = a_data_tile[m_idx][b_idx * 2u];
        let a_data1 = a_data_tile[m_idx][b_idx * 2u + 1u];

        results[m_idx] += f32(dot(a_data0, b_dequantized[0])) + f32(dot(a_data1, b_dequantized[1]));
      }
    }
)MAIN_FN";
  } else {
    shader.MainFunctionBody() << "    var b_data0 = vec4<u32>(0);\n"
                                 "    var b_data1 = vec4<u32>(0);\n"
                                 "    if (b_row < uniforms.input_b_shape[0] && b_col < uniforms.input_b_shape[1]) {\n"
                              << "      b_data0 = " << b.GetByIndices("input_b_indices_t(b_row, b_col, 0)") << ";\n"
                              << "      b_data1 = " << b.GetByIndices("input_b_indices_t(b_row, b_col, 1)") << ";\n"
                                                                                                               "    }"
                              << R"MAIN_FN(
    for (var b_idx = 0u; b_idx < 4u; b_idx++) {
      let b_dequantized0 = (vec4<output_element_t>(unpack4xU8(b_data0[b_idx])) - vec4<output_element_t>(zero_point)) * scale;
      let b_dequantized1 = (vec4<output_element_t>(unpack4xU8(b_data1[b_idx])) - vec4<output_element_t>(zero_point)) * scale;
      for (var m_idx = 0u; m_idx < kTileM; m_idx++) {
        let a_data0 = a_data_tile[m_idx][b_idx];
        let a_data1 = a_data_tile[m_idx][b_idx + 4u];

        results[m_idx] += f32(dot(a_data0, b_dequantized0)) + f32(dot(a_data1, b_dequantized1));
      }
    }
)MAIN_FN";
  }

  shader.MainFunctionBody() << R"MAIN_FN(

    workgroupBarrier();
  }

  if (batch >= uniforms.input_a_shape[0]) {
    return;
  }

  // Write the results.
  for (var m_idx = 0u; m_idx < kTileM; m_idx++) {
    mm_write_y(batch, row + m_idx, col + local_idx, output_value_t(results[m_idx]));
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
                                    << ReadZeroPoint(nbits_, has_zero_points_);

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

  const bool has_zero_points = zero_points != nullptr;
#if !defined(__wasm__)
  int32_t subgroup_matrix_config_index = -1;
  // apple|intel - Experimental dawn support for subgroup matrix matmul.
  if (M >= kMinMForTileOptimization && (context.AdapterInfo().vendor == std::string_view{"apple"} || context.AdapterInfo().vendor == std::string_view{"intel"}) &&
      CanApplySubgroupMatrixMatMulNBits(context, accuracy_level_, block_size, batch_count, N, K, has_zero_points, subgroup_matrix_config_index)) {
    return ApplySubgroupMatrixMatMulNBits(a, b, scales, M, N, K, nbits, subgroup_matrix_config_index, context, y);
  }
#endif

  // On FP32 only GPUs, integer math is faster than FP32 therefore always use DP4A independent of length of M.
  if ((M >= kMinMForTileOptimization || y->DataType() == DataTypeImpl::GetType<float>() ||
       context.AdapterInfo().vendor == std::string_view{"qualcomm"}) &&
      CanApplyDP4AMatrixMatMulNBits(context, accuracy_level_, block_size, batch_count, N, K, components_a, has_zero_points)) {
    return ApplyDP4AMatrixMatMulNBits(a, b, scales, M, N, K, block_size, kMinMForTileOptimization, nbits, context, y);
  }

  // zero_points has shape[N * CeilDiv(n_blocks_per_col * bits, 8)]. So here we need to check whether n_blocks_per_col is divisible by 8/nbits.
  // For bits==4, this is counted by elements of uint4. Need add 1 if not divisible by 2.
  uint32_t zero_blocks_per_col = n_blocks_per_col % (8 / nbits) == 0 ? n_blocks_per_col : n_blocks_per_col + 1;

  // WideTileProgram
  // This program is optimized for Block32 prefill using Tile16x128.
  const bool use_wide_tile_program = block_size == 32 && components_a == 4 && components_b == 4 && M >= kMinMForTileOptimization;
  if (use_wide_tile_program) {
    // Enforce output components to 1.
    components = 1;

    constexpr uint32_t workgroup_size = 128;
    constexpr uint32_t tile_m = workgroup_size / 8;
    constexpr uint32_t tile_n = workgroup_size;
    uint32_t num_N_tile = (N + tile_n - 1) / tile_n;
    uint32_t num_M_tile = (M + tile_m - 1) / tile_m;

    MatMulNBitsWideTileProgram program{has_zero_points, tile_m, tile_n, nbits};
    program.SetWorkgroupSize(workgroup_size);
    program.SetDispatchGroupSize((N + tile_n - 1) / tile_n,
                                 (M + tile_m - 1) / tile_m,
                                 batch_count);
    program.CacheHint("Tile" + std::to_string(tile_m) + "x" + std::to_string(tile_n) + "_Block32");

    TensorShape reshaped_a_shape{batch_count, M, K / components_a};
    TensorShape reshaped_b_shape{N, n_blocks_per_col, blob_size_in_words / components_b};
    TensorShape reshaped_y_shape{batch_count, M, N / components};

    program
        .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, reshaped_a_shape, onnxruntime::narrow<int>(components_a)},
                    {b, ProgramTensorMetadataDependency::TypeAndRank, reshaped_b_shape, onnxruntime::narrow<int>(components_b * 4)},
                    {scales, ProgramTensorMetadataDependency::None}})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, reshaped_y_shape, onnxruntime::narrow<int>(components)})
        .AddUniformVariables({{block_size}, {zero_blocks_per_col}, {num_N_tile}, {num_M_tile}})
        .CacheHint(nbits, has_zero_points);
    if (has_zero_points) {
      program.AddInput({zero_points, ProgramTensorMetadataDependency::None, {(zero_points->Shape().Size() + 3) / 4}, 4});
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
