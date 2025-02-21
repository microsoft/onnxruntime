// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string_view>

#include "contrib_ops/webgpu/quantization/matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

namespace {
// Put it to a common place?
uint32_t GetMaxComponents(uint32_t size) {
  // we cannot use vec3 type since it has alignment of 16 bytes
  if (size % 4 == 0) {
    return 4;
  } else if (size % 2 == 0) {
    return 2;
  }

  return 1;
}

std::string QuantizedDataType(int components) {
  switch (components) {
    case 1:
      return "array<output_element_t, 8>";
    case 2:
      return "mat4x2<output_element_t>";
    case 4:
      return "mat2x4<output_element_t>";
    default:
      return "array<output_element_t, 8>";
  }
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

Status MatMulNBitsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("input_a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& b = shader.AddInput("input_b", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& scales = shader.AddInput("scales", ShaderUsage::UseUniform);
  const auto& y = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias | ShaderUsage::UseIndicesTypeAlias);

  if (block_size_ == 32) {
    const uint32_t workgroup_size = WorkgroupSizeX() * WorkgroupSizeY();
    const uint32_t tile_size = WorkgroupSizeX() * components_b_ * 8;  // each uint32 has 8 data.
    const uint32_t a_length_per_tile = tile_size / a.NumComponents();
    const uint32_t blocks_per_tile = tile_size / block_size_;
    if (tile_m_ > 1 && use_subgroup_) {
      ORT_ENFORCE(a.NumComponents() == 4, "input a's components must be equal to 4.");
      ORT_ENFORCE(components_b_ == 4, "input b's components must be equal to 4.");
      shader.AdditionalImplementation() << "fn mm_readA(batch : u32, row : u32, col : u32) -> input_a_value_t {\n"
                                           "  if (row < uniforms.input_a_shape[1] && col < uniforms.input_a_shape[2]) {\n"
                                        << "    return " << a.GetByIndices("input_a_indices_t(batch, row, col)") << ";\n"
                                        << "  } else {\n"
                                           "    return input_a_value_t(0);\n"
                                           "  }\n"
                                           "}\n"
                                        << "var<workgroup> sub_b: array<array<input_b_value_t, " << WorkgroupSizeX() << ">, " << WorkgroupSizeY() << ">;\n"
                                        << "var<workgroup> sub_scale: array<array<output_value_t, " << WorkgroupSizeX() << ">, " << WorkgroupSizeY() << ">;\n"
                                        << "var<workgroup> inter_results: array<array<array<output_value_t, " << WorkgroupSizeX() << ">, " << WorkgroupSizeY() << ">," << tile_m_ << ">;\n";
      shader.MainFunctionBody() << "  let col = workgroup_id.x * " << WorkgroupSizeY() << ";\n"
                                << "  let row = workgroup_id.y * " << tile_m_ << ";\n"
                                << "  let batch = workgroup_id.z;\n";
      shader.MainFunctionBody() << "  let n_blocks_per_col = uniforms.input_b_shape[1];\n"
                                << "  let num_tiles =  (n_blocks_per_col - 1) / " << blocks_per_tile << " + 1;\n"
                                // Loop over shared dimension.
                                << "  for (var tile: u32 = 0; tile < num_tiles; tile += 1) {\n"
                                << "    // load one tile B/scale data into shared memory.\n"
                                   // Each thread processes one block.
                                   "    let b_col = col + local_id.y;\n"
                                << "    let block = tile * " << blocks_per_tile << " + local_id.x;\n"
                                << "    if (b_col < uniforms.input_b_shape[0] && block < n_blocks_per_col) {\n"
                                << "      sub_b[local_id.y][local_id.x] = " << b.GetByIndices("input_b_indices_t(b_col, block, 0)") << ";\n"
                                << "      sub_scale[local_id.y][local_id.x] = " << scales.GetByOffset("b_col * n_blocks_per_col + block") << ";\n"
                                << "    } else {\n"
                                   "      sub_b[local_id.y][local_id.x] = input_b_value_t(0);\n"
                                   "      sub_scale[local_id.y][local_id.x] = output_value_t(0);\n"
                                   "    }\n"
                                   "    workgroupBarrier();\n"
                                << "    var in_y = (local_idx % 32) / 4;\n"
                                   "    var in_x = (local_idx / 32) * 4 + local_idx % 4;\n"
                                << "    var word_offset = (local_idx % 4) * " << block_size_ / a.NumComponents() << ";\n"
                                << "    if (sg_size == 8u) {\n"
                                   "      in_y = local_idx % 8;\n"
                                   "      in_x = local_idx / 8;\n"
                                << "      word_offset = 0u;\n"
                                   "    } else if (sg_size == 16u) {\n"
                                   "      in_y = (local_idx % 16) / 2;\n"
                                   "      in_x = (local_idx / 16) * 2 + local_idx % 2;\n"
                                << "      word_offset = (local_idx % 2) * " << block_size_ / a.NumComponents() << ";\n"
                                << "    } else if (sg_size == 32u) {\n"
                                   "      in_y = (local_idx % 32) / 4;\n"
                                   "      in_x = (local_idx / 32) * 4 + local_idx % 4;\n"
                                << "      word_offset = (local_idx % 4) * " << block_size_ / a.NumComponents() << ";\n"
                                << "    } else if (sg_size == 64u) {\n"
                                   "      in_y = local_idx / 8;\n"
                                   "      in_x = local_idx % 8;\n"
                                << "      word_offset = (local_idx % 8) * " << block_size_ / a.NumComponents() << ";\n"
                                << "    }\n";
      if (has_zero_points_) {
        const auto& zero_points = shader.AddInput("zero_points", ShaderUsage::UseUniform);
        shader.MainFunctionBody() << "    let zero_point_bytes_per_col = (n_blocks_per_col + 1) / 2;\n"
                                     "    let zero_point_byte_count = b_col * zero_point_bytes_per_col + (block >> 0x1u);\n"
                                     "    let zero_point_word_index = zero_point_byte_count >> 0x2u;\n"
                                     "    let zero_point_byte_offset = zero_point_byte_count & 0x3u;\n"
                                     "    let zero_point_nibble_offset: u32 = block & 0x1u;\n"
                                     "    let zero_point_bits_offset = (zero_point_byte_offset << 3) + (zero_point_nibble_offset << 2);\n"
                                  << "    let zero_point_word = " << zero_points.GetByOffset("zero_point_word_index") << " >> zero_point_bits_offset;\n"
                                  << "    let zero_point = output_element_t((zero_point_word) & 0xFu);\n";
      } else {
        // The default zero point is 8 for unsigned 4-bit quantization.
        shader.MainFunctionBody() << "    let zero_point = output_element_t(8.0);\n";
      }
      shader.MainFunctionBody() << "    let scale = sub_scale[in_y][in_x];\n"
                                   "    let b_data = sub_b[in_y][in_x];\n";
      shader.MainFunctionBody() << "    let a_col_start = tile * " << a_length_per_tile << ";\n";
      for (uint32_t i = 0; i < tile_m_; i++) {
        shader.MainFunctionBody() << "    let a_data" << i << " = mm_readA(batch, row + " << i << ", a_col_start + local_idx);\n";
      }

      shader.MainFunctionBody() << "    if (sg_size == 8u) {\n";
      shader.MainFunctionBody() << "      for (var i: u32 = 0; i < 4; i++) {\n";
      shader.MainFunctionBody() << "        let b_value = b_data[i];\n"
                                   "        let b_value_lower = unpack4xU8(b_value & 0x0F0F0F0Fu);\n"
                                   "        let b_value_upper = unpack4xU8((b_value >> 4) & 0x0F0F0F0Fu);\n"
                                   "        let b_quantized_values = mat2x4<output_element_t>(output_element_t(b_value_lower[0]), output_element_t(b_value_upper[0]), output_element_t(b_value_lower[1]), output_element_t(b_value_upper[1]), output_element_t(b_value_lower[2]), output_element_t(b_value_upper[2]), output_element_t(b_value_lower[3]), output_element_t(b_value_upper[3]));\n"
                                   "        let b_dequantized_values = (b_quantized_values - mat2x4<output_element_t>(zero_point, zero_point, zero_point, zero_point, zero_point, zero_point, zero_point, zero_point)) * scale;\n";
      for (uint32_t i = 0; i < tile_m_; i++) {
        if (i == 0) {
          shader.MainFunctionBody() << "        var ";
        }
        shader.MainFunctionBody() << "        a0 = subgroupShuffle(a_data" << i << ", i * 2);\n";
        if (i == 0) {
          shader.MainFunctionBody() << "        var ";
        }
        shader.MainFunctionBody() << "        a1 = subgroupShuffle(a_data" << i << ", i * 2 + 1);\n";
        shader.MainFunctionBody() << "        inter_results[" << i << "][in_y][in_x] += dot(a0, b_dequantized_values[0]) + dot(a1, b_dequantized_values[1]);\n";
      }
      shader.MainFunctionBody() << "      }\n";
      shader.MainFunctionBody() << "    } else if (sg_size == 16u) {\n";
      shader.MainFunctionBody() << "      for (var i: u32 = 0; i < 4; i++) {\n";
      shader.MainFunctionBody() << "        let b_value = b_data[i];\n"
                                   "        let b_value_lower = unpack4xU8(b_value & 0x0F0F0F0Fu);\n"
                                   "        let b_value_upper = unpack4xU8((b_value >> 4) & 0x0F0F0F0Fu);\n"
                                   "        let b_quantized_values = mat2x4<output_element_t>(output_element_t(b_value_lower[0]), output_element_t(b_value_upper[0]), output_element_t(b_value_lower[1]), output_element_t(b_value_upper[1]), output_element_t(b_value_lower[2]), output_element_t(b_value_upper[2]), output_element_t(b_value_lower[3]), output_element_t(b_value_upper[3]));\n"
                                   "        let b_dequantized_values = (b_quantized_values - mat2x4<output_element_t>(zero_point, zero_point, zero_point, zero_point, zero_point, zero_point, zero_point, zero_point)) * scale;\n";
      for (uint32_t i = 0; i < tile_m_; i++) {
        if (i == 0) {
          shader.MainFunctionBody() << "        var ";
        }
        shader.MainFunctionBody() << "        a0 = subgroupShuffle(a_data" << i << ", i * 2);\n";
        if (i == 0) {
          shader.MainFunctionBody() << "        var ";
        }
        shader.MainFunctionBody() << "        a00 = subgroupShuffle(a_data" << i << ", i * 2 + 8);\n";
        if (i == 0) {
          shader.MainFunctionBody() << "        var ";
        }
        shader.MainFunctionBody() << "        a1 = subgroupShuffle(a_data" << i << ", i * 2 + 1);\n";
        if (i == 0) {
          shader.MainFunctionBody() << "        var ";
        }
        shader.MainFunctionBody() << "        a11 = subgroupShuffle(a_data" << i << ", i * 2 + 9);\n";
        shader.MainFunctionBody() << "        inter_results[" << i << "][in_y][in_x] += dot(select(a00, a0, local_idx % 2 == 0), b_dequantized_values[0]) + dot(select(a11, a1, local_idx % 2 == 0), b_dequantized_values[1]);\n";
      }
      shader.MainFunctionBody() << "        word_offset += " << 8 / a.NumComponents() << ";\n"
                                << "      }\n";
      shader.MainFunctionBody() << "    } else {\n";
      shader.MainFunctionBody() << "      for (var i: u32 = 0; i < 4; i++) {\n";
      shader.MainFunctionBody() << "        let b_value = b_data[i];\n"
                                   "        let b_value_lower = unpack4xU8(b_value & 0x0F0F0F0Fu);\n"
                                   "        let b_value_upper = unpack4xU8((b_value >> 4) & 0x0F0F0F0Fu);\n"
                                   "        let b_quantized_values = mat2x4<output_element_t>(output_element_t(b_value_lower[0]), output_element_t(b_value_upper[0]), output_element_t(b_value_lower[1]), output_element_t(b_value_upper[1]), output_element_t(b_value_lower[2]), output_element_t(b_value_upper[2]), output_element_t(b_value_lower[3]), output_element_t(b_value_upper[3]));\n"
                                   "        let b_dequantized_values = (b_quantized_values - mat2x4<output_element_t>(zero_point, zero_point, zero_point, zero_point, zero_point, zero_point, zero_point, zero_point)) * scale;\n";
      for (uint32_t i = 0; i < tile_m_; i++) {
        if (i == 0) {
          shader.MainFunctionBody() << "        var ";
        }
        shader.MainFunctionBody() << "        a0 = subgroupShuffle(a_data" << i << ", word_offset);\n";
        if (i == 0) {
          shader.MainFunctionBody() << "        var ";
        }
        shader.MainFunctionBody() << "        a1 = subgroupShuffle(a_data" << i << ", word_offset + 1);\n";
        shader.MainFunctionBody() << "        inter_results[" << i << "][in_y][in_x] += dot(a0, b_dequantized_values[0]) + dot(a1, b_dequantized_values[1]);\n";
      }
      shader.MainFunctionBody() << "        word_offset += " << 8 / a.NumComponents() << ";\n";
      shader.MainFunctionBody() << "      }\n";
      shader.MainFunctionBody() << "    }\n";
      shader.MainFunctionBody() << "    workgroupBarrier();\n";

      shader.MainFunctionBody() << "  }\n";
      shader.MainFunctionBody() << "  if (local_idx < " << WorkgroupSizeY() * tile_m_ << ") {\n"
                                << "    let inner_row = local_idx / " << WorkgroupSizeY() << ";\n"
                                << "    let inner_col = local_idx % " << WorkgroupSizeY() << ";\n"
                                << "    var output_value = output_value_t(0);\n"
                                << "    for (var b = 0u; b < " << WorkgroupSizeX() << "; b++) {\n"
                                << "      output_value += inter_results[inner_row][inner_col][b];\n"
                                   "    }\n"
                                   "    if (row + inner_row < uniforms.output_shape[1] && col + inner_col < uniforms.output_shape[2]) {\n"
                                << "      " << y.SetByIndices("output_indices_t(batch, row + inner_row, col + inner_col)", "output_value") << ";\n"
                                << "    }\n"
                                   "  }\n";
    } else {
      if (tile_m_ == 1) {
        shader.AdditionalImplementation() << "fn mm_readA(batch : u32, row : u32, col : u32) -> input_a_value_t {\n"
                                             "  if (col < uniforms.input_a_shape[2]) {\n"
                                          << "    return " << a.GetByIndices("input_a_indices_t(batch, row, col)") << ";\n"
                                          << "  } else {\n"
                                             "    return input_a_value_t(0);\n"
                                             "  }\n"
                                             "}\n"
                                          << "var<workgroup> sub_a: array<input_a_value_t, " << a_length_per_tile << ">;\n"
                                          << "var<workgroup> inter_results: array<array<output_value_t, " << WorkgroupSizeX() << ">, " << WorkgroupSizeY() << ">;\n";
        std::string offset = "workgroup_idx * " + std::to_string(WorkgroupSizeY());
        shader.MainFunctionBody() << "  let output_indices = " << y.OffsetToIndices(offset) << ";\n"
                                  << "  let col = output_indices[2];\n"
                                     "  let row = output_indices[1];\n"
                                     "  let batch = output_indices[0];\n";
      } else {
        ORT_ENFORCE(tile_m_ < WorkgroupSizeY(), "tile_m must be less than or equal to WorkgroupSizeY.");
        ORT_ENFORCE(WorkgroupSizeX() == WorkgroupSizeY(), "WorkgroupSizeX must be equal to WorkgroupSizeY.");

        shader.AdditionalImplementation() << "fn mm_readA(batch : u32, row : u32, col : u32) -> input_a_value_t {\n"
                                             "  if (row < uniforms.input_a_shape[1] && col < uniforms.input_a_shape[2]) {\n"
                                          << "    return " << a.GetByIndices("input_a_indices_t(batch, row, col)") << ";\n"
                                          << "  } else {\n"
                                             "    return input_a_value_t(0);\n"
                                             "  }\n"
                                             "}\n"
                                          << "var<workgroup> sub_a: array<array<input_a_value_t, " << a_length_per_tile << ">," << tile_m_ << ">;\n"
                                          << "var<workgroup> inter_results: array<array<array<output_value_t, " << WorkgroupSizeX() << ">, " << WorkgroupSizeY() << ">," << tile_m_ << ">;\n";
        shader.MainFunctionBody() << "  let col = workgroup_id.x * " << WorkgroupSizeY() << ";\n"
                                  << "  let row = workgroup_id.y * " << tile_m_ << ";\n"
                                  << "  let batch = workgroup_id.z;\n";
      }
      shader.MainFunctionBody() << "  let n_blocks_per_col = uniforms.input_b_shape[1];\n"
                                << "  let num_tiles =  (n_blocks_per_col - 1) / " << blocks_per_tile << " + 1;\n"
                                // Loop over shared dimension.
                                << "  for (var tile: u32 = 0; tile < num_tiles; tile += 1) {\n"
                                << "    let a_col_start = tile * " << a_length_per_tile << ";\n"
                                << "    // load one tile A data into shared memory.\n"
                                << "    for (var a_offset = local_idx; a_offset < " << a_length_per_tile << "; a_offset += " << workgroup_size << ") {\n"
                                << "      let a_col = a_col_start + a_offset;\n";
      if (tile_m_ == 1) {
        shader.MainFunctionBody() << "      sub_a[a_offset] = mm_readA(batch, row, a_col);\n";
      } else {
        for (uint32_t i = 0; i < tile_m_; i++) {
          shader.MainFunctionBody() << "      sub_a[" << i << "][a_offset] = mm_readA(batch, row + " << i << ", a_col);\n";
        }
      }
      shader.MainFunctionBody() << "    }\n"
                                   "    workgroupBarrier();\n"
                                   // Each thread processes one block.
                                   "    let b_row = col + local_id.y;\n"
                                << "    let block = tile * " << blocks_per_tile << " + local_id.x;\n";
      if (has_zero_points_) {
        const auto& zero_points = shader.AddInput("zero_points", ShaderUsage::UseUniform);
        shader.MainFunctionBody() << "    let zero_point_bytes_per_col = (n_blocks_per_col + 1) / 2;\n"
                                     "    let zero_point_byte_count = b_row * zero_point_bytes_per_col + (block >> 0x1u);\n"
                                     "    let zero_point_word_index = zero_point_byte_count >> 0x2u;\n"
                                     "    let zero_point_byte_offset = zero_point_byte_count & 0x3u;\n"
                                     "    let zero_point_nibble_offset: u32 = block & 0x1u;\n"
                                     "    let zero_point_bits_offset = (zero_point_byte_offset << 3) + (zero_point_nibble_offset << 2);\n"
                                  << "    let zero_point_word = " << zero_points.GetByOffset("zero_point_word_index") << " >> zero_point_bits_offset;\n"
                                  << "    let zero_point = output_element_t((zero_point_word) & 0xFu);\n";
      } else {
        // The default zero point is 8 for unsigned 4-bit quantization.
        shader.MainFunctionBody() << "    let zero_point = output_element_t(8.0);\n";
      }
      shader.MainFunctionBody() << "    var scale = output_element_t(0);\n"
                                   "    var b_data = input_b_value_t(0);\n"
                                << "    if (block < n_blocks_per_col) {\n"
                                << "      scale = " << scales.GetByOffset("b_row * n_blocks_per_col + block") << ";\n"
                                << "      b_data = " << b.GetByIndices("input_b_indices_t(b_row, block, 0)") << ";\n"
                                << "    }\n"
                                << "    var word_offset = local_id.x * " << block_size_ / a.NumComponents() << ";\n"
                                << "    for (var i: u32 = 0; i < " << components_b_ << "; i++) {\n";
      shader.MainFunctionBody() << "      let b_value = b_data";
      if (components_b_ > 1) {
        shader.MainFunctionBody() << "[i]";
      }
      shader.MainFunctionBody() << ";\n"
                                   "      let b_value_lower = unpack4xU8(b_value & 0x0F0F0F0Fu);\n"
                                   "      let b_value_upper = unpack4xU8((b_value >> 4) & 0x0F0F0F0Fu);\n"
                                   "      let b_quantized_values = mat2x4<output_element_t>(output_element_t(b_value_lower[0]), output_element_t(b_value_upper[0]), output_element_t(b_value_lower[1]), output_element_t(b_value_upper[1]), output_element_t(b_value_lower[2]), output_element_t(b_value_upper[2]), output_element_t(b_value_lower[3]), output_element_t(b_value_upper[3]));\n"
                                   "      let b_dequantized_values = (b_quantized_values - mat2x4<output_element_t>(";
      for (int i = 0; i < 8; i++) {
        shader.MainFunctionBody() << "zero_point";
        if (i < 7) {
          shader.MainFunctionBody() << ", ";
        }
      }
      shader.MainFunctionBody() << ")) * scale;\n";
      if (tile_m_ == 1) {
        switch (a.NumComponents()) {
          case 1:
            shader.MainFunctionBody() << "      inter_results[local_id.y][local_id.x] += dot(vec4<output_element_t>(sub_a[word_offset], sub_a[word_offset + 1], sub_a[word_offset + 2], sub_a[word_offset + 3]), b_dequantized_values[0]) + dot(vec4<output_element_t>(sub_a[word_offset + 4], sub_a[word_offset + 5], sub_a[word_offset + 6], sub_a[word_offset + 7]), b_dequantized_values[1]);\n";
            break;
          case 2:
            shader.MainFunctionBody() << "      inter_results[local_id.y][local_id.x] += dot(vec4<output_element_t>(sub_a[word_offset], sub_a[word_offset + 1]), b_dequantized_values[0]) + dot(vec4<output_element_t>(sub_a[word_offset + 2], sub_a[word_offset + 3]), b_dequantized_values[1]);\n";
            break;
          case 4:
            shader.MainFunctionBody() << "      inter_results[local_id.y][local_id.x] += dot(sub_a[word_offset], b_dequantized_values[0]) + dot(sub_a[word_offset + 1], b_dequantized_values[1]);\n";
            break;
          default:
            break;
        }
      } else {
        for (uint32_t i = 0; i < tile_m_; i++) {
          switch (a.NumComponents()) {
            case 1:
              shader.MainFunctionBody() << "      inter_results[" << i << "][local_id.y][local_id.x] += dot(vec4<output_element_t>(sub_a[" << i << "][word_offset], sub_a[" << i << "][word_offset + 1], sub_a[" << i << "][word_offset + 2], sub_a[" << i << "][word_offset + 3]), b_dequantized_values[0]) + dot(vec4<output_element_t>(sub_a[" << i << "][word_offset + 4], sub_a[" << i << "][word_offset + 5], sub_a[" << i << "][word_offset + 6], sub_a[" << i << "][word_offset + 7]), b_dequantized_values[1]);\n";
              break;
            case 2:
              shader.MainFunctionBody() << "      inter_results[" << i << "][local_id.y][local_id.x] += dot(vec4<output_element_t>(sub_a[" << i << "][word_offset], sub_a[" << i << "][word_offset + 1]), b_dequantized_values[0]) + dot(vec4<output_element_t>(sub_a[" << i << "][word_offset + 2], sub_a[" << i << "][word_offset + 3]), b_dequantized_values[1]);\n";
              break;
            case 4:
              shader.MainFunctionBody() << "      inter_results[" << i << "][local_id.y][local_id.x] += dot(sub_a[" << i << "][word_offset], b_dequantized_values[0]) + dot(sub_a[" << i << "][word_offset + 1], b_dequantized_values[1]);\n";
              break;
            default:
              break;
          }
        }
      }
      shader.MainFunctionBody() << "      word_offset += " << 8 / a.NumComponents() << ";\n"
                                << "    }\n"
                                   "    workgroupBarrier();\n"
                                   "  }\n";
      if (tile_m_ == 1) {
        shader.MainFunctionBody() << "  if (local_idx < " << WorkgroupSizeY() << ") {\n"
                                  << "    var output_value = output_value_t(0);\n"
                                  << "    for (var b = 0u; b < " << WorkgroupSizeX() << "; b++) {\n"
                                  << "      output_value += inter_results[local_idx][b];\n"
                                     "    }\n"
                                     "    if (col + local_idx < uniforms.output_shape[2]) {\n"
                                  << "      " << y.SetByIndices("output_indices_t(batch, row, col + local_idx)", "output_value") << ";\n"
                                  << "    }\n"
                                     "  }\n";
      } else {
        shader.MainFunctionBody() << "  if (local_id.y < " << tile_m_ << ") {\n"
                                  << "    var output_value = output_value_t(0);\n"
                                  << "    for (var b = 0u; b < " << WorkgroupSizeX() << "; b++) {\n"
                                  << "      output_value += inter_results[local_id.y][local_id.x][b];\n"
                                     "    }\n"
                                     "    if (row + local_id.y < uniforms.output_shape[1] && col + local_id.x < uniforms.output_shape[2]) {\n"
                                  << "      " << y.SetByIndices("output_indices_t(batch, row + local_id.y, col + local_id.x)", "output_value") << ";\n"
                                  << "    }\n"
                                     "  }\n";
      }
    }
  } else {
    const std::string quantized_data_type = QuantizedDataType(a.NumComponents());
    const int output_element_number = y.NumComponents() * gsl::narrow<int>(output_number_);

    const uint32_t shared_memory_size = output_number_ * WORKGROUP_SIZE;
    std::string offset = "workgroup_idx * " + std::to_string(output_number_);
    shader.AdditionalImplementation() << "var<workgroup> workgroup_shared : array<output_value_t," << shared_memory_size << ">;\n";
    shader.MainFunctionBody() << "  let output_indices = " << y.OffsetToIndices(offset) << ";\n"
                              << "  let col = output_indices[2];\n"
                                 "  let row = output_indices[1];\n"
                                 "  let batch = output_indices[0];\n"
                                 "  let n_blocks_per_col = uniforms.input_b_shape[1];\n"
                                 "  let blob_size = uniforms.input_b_shape[2];\n"
                                 "  for (var block = local_id.x; block < n_blocks_per_col; block += workgroup_size_x) {\n"
                              << "    var word_offset = block * uniforms.block_size / " << a.NumComponents() << ";\n";

    // prepare scale and zero point
    shader.MainFunctionBody() << "    var col_index = col * " << y.NumComponents() << ";\n";
    if (has_zero_points_) {
      const auto& zero_points = shader.AddInput("zero_points", ShaderUsage::UseUniform);
      shader.MainFunctionBody() << "    let zero_point_bytes_per_col = (n_blocks_per_col + 1) / 2;\n"
                                   "    var zero_point_byte_count: u32;\n"
                                   "    var zero_point_word_index: u32;\n"
                                   "    var zero_point_byte_offset: u32;\n"
                                   "    let zero_point_nibble_offset: u32 = block & 0x1u;\n"
                                   "    var zero_point_bits_offset: u32;\n"
                                   "    var zero_point_word: u32;\n";
      for (int c = 0; c < output_element_number; c++) {
        shader.MainFunctionBody() << "    let scale" << c << " = " << scales.GetByOffset("col_index * n_blocks_per_col + block") << ";\n"
                                  << "    zero_point_byte_count = col_index * zero_point_bytes_per_col + (block >> 0x1u);\n"
                                     "    zero_point_word_index = zero_point_byte_count >> 0x2u;\n"
                                     "    zero_point_byte_offset = zero_point_byte_count & 0x3u;\n"
                                     "    zero_point_bits_offset = (zero_point_byte_offset << 3) + (zero_point_nibble_offset << 2);\n"
                                  << "    zero_point_word = " << zero_points.GetByOffset("zero_point_word_index") << " >> zero_point_bits_offset;\n"
                                  << "    let zero_point" << c << " = output_element_t((zero_point_word) & 0xFu);\n"
                                  << "    col_index += 1;\n";
      }
    } else {
      shader.MainFunctionBody() << "    let zero_point = output_element_t(8.0);\n";
      for (int c = 0; c < output_element_number; c++) {
        shader.MainFunctionBody() << "    let scale" << c << " = " << scales.GetByOffset("col_index * n_blocks_per_col + block") << ";\n"
                                  << "    col_index += 1;\n";
      }
    }

    shader.MainFunctionBody() << "    for (var word: u32 = 0; word < blob_size; word += 1) {\n";

    // prepare b data
    shader.MainFunctionBody() << "      col_index = col * " << y.NumComponents() << ";\n";
    for (int c = 0; c < output_element_number; c++) {
      shader.MainFunctionBody() << "      let b" << c << "_data = " << b.GetByIndices("input_b_indices_t(col_index, block, word)") << ";\n"
                                << "      col_index += 1;\n";
    }
    shader.MainFunctionBody() << "      var b_value : u32;\n"
                                 "      let b_mask : u32 = 0x0F0F0F0Fu;\n"
                                 "      var b_value_lower : vec4<u32>;\n"
                                 "      var b_value_upper : vec4<u32>;\n"
                              << "      var b_quantized_values : " << quantized_data_type << ";\n"
                              << "      var b_dequantized_values : " << quantized_data_type << ";\n";

    shader.MainFunctionBody() << "      for (var i: u32 = 0; i < " << components_b_ << "; i++) {\n";

    // process one word
    shader.MainFunctionBody() << "        var input_offset = " << a.IndicesToOffset("input_a_indices_t(batch, row, word_offset)") << ";\n"
                              << "        var a_data: " << quantized_data_type << ";\n"
                              << "        for (var j: u32 = 0; j < " << (8 / a.NumComponents()) << "; j++) {\n"
                              << "          if (word_offset + j < uniforms.input_a_shape[2]) {\n"
                              << "            a_data[j] = " << a.GetByOffset("input_offset") << ";\n"
                              << "            input_offset++;\n"
                                 "          } else {\n"
                                 "            a_data[j] = input_a_value_t(0);\n"
                                 "          }\n"
                                 "        }\n";
    for (int c = 0; c < output_element_number; c++) {
      shader.MainFunctionBody() << "        b_value = b" << c << "_data";
      if (components_b_ > 1) {
        shader.MainFunctionBody() << "[i]";
      }
      shader.MainFunctionBody() << ";\n"
                                   "        b_value_lower = unpack4xU8(b_value & b_mask);\n"
                                   "        b_value_upper = unpack4xU8((b_value >> 4) & b_mask);\n"
                                << "        b_quantized_values = " << quantized_data_type << "(output_element_t(b_value_lower[0]), output_element_t(b_value_upper[0]), output_element_t(b_value_lower[1]), output_element_t(b_value_upper[1]), output_element_t(b_value_lower[2]), output_element_t(b_value_upper[2]), output_element_t(b_value_lower[3]), output_element_t(b_value_upper[3]));\n"
                                << "        b_dequantized_values = ";
      if (a.NumComponents() == 1) {
        if (has_zero_points_) {
          shader.MainFunctionBody() << quantized_data_type << "((b_quantized_values[0] - zero_point" << c << ") * scale" << c << ", "
                                    << "(b_quantized_values[1] - zero_point" << c << ") * scale" << c << ", "
                                    << "(b_quantized_values[2] - zero_point" << c << ") * scale" << c << ", "
                                    << "(b_quantized_values[3] - zero_point" << c << ") * scale" << c << ", "
                                    << "(b_quantized_values[4] - zero_point" << c << ") * scale" << c << ", "
                                    << "(b_quantized_values[5] - zero_point" << c << ") * scale" << c << ", "
                                    << "(b_quantized_values[6] - zero_point" << c << ") * scale" << c << ", "
                                    << "(b_quantized_values[7] - zero_point" << c << ") * scale" << c << ");\n";
        } else {
          shader.MainFunctionBody() << quantized_data_type << "((b_quantized_values[0] - zero_point) * scale" << c << ", "
                                    << "(b_quantized_values[1] - zero_point) * scale" << c << ","
                                    << "(b_quantized_values[2] - zero_point) * scale" << c << ","
                                    << "(b_quantized_values[3] - zero_point) * scale" << c << ","
                                    << "(b_quantized_values[4] - zero_point) * scale" << c << ","
                                    << "(b_quantized_values[5] - zero_point) * scale" << c << ","
                                    << "(b_quantized_values[6] - zero_point) * scale" << c << ","
                                    << "(b_quantized_values[7] - zero_point) * scale" << c << ");\n";
        }
      } else {
        shader.MainFunctionBody() << "(b_quantized_values - " << quantized_data_type << "(";
        for (int i = 0; i < 8; i++) {
          if (has_zero_points_) {
            shader.MainFunctionBody() << "zero_point" << c;
          } else {
            shader.MainFunctionBody() << "zero_point";
          }
          if (i < 7) {
            shader.MainFunctionBody() << ", ";
          }
        }
        shader.MainFunctionBody() << ")) * scale" << c << ";\n";
      }

      shader.MainFunctionBody() << "        workgroup_shared[local_id.x * " << output_number_ << " + " << c / y.NumComponents() << "]";
      if (y.NumComponents() > 1) {
        shader.MainFunctionBody() << "[" << c % y.NumComponents() << "]";
      }
      shader.MainFunctionBody() << " += ";
      if (a.NumComponents() == 1) {
        shader.MainFunctionBody() << "a_data[0] * b_dequantized_values[0] + "
                                     "a_data[1] * b_dequantized_values[1] + "
                                     "a_data[2] * b_dequantized_values[2] + "
                                     "a_data[3] * b_dequantized_values[3] + "
                                     "a_data[4] * b_dequantized_values[4] + "
                                     "a_data[5] * b_dequantized_values[5] + "
                                     "a_data[6] * b_dequantized_values[6] + "
                                     "a_data[7] * b_dequantized_values[7];\n";
      } else if (a.NumComponents() == 2) {
        shader.MainFunctionBody() << "dot(a_data[0], b_dequantized_values[0]) + "
                                     "dot(a_data[1], b_dequantized_values[1]) + "
                                     "dot(a_data[2], b_dequantized_values[2]) + "
                                     "dot(a_data[3], b_dequantized_values[3]);\n";
      } else if (a.NumComponents() == 4) {
        shader.MainFunctionBody() << "dot(a_data[0], b_dequantized_values[0]) + "
                                     "dot(a_data[1], b_dequantized_values[1]);\n";
      }
    }

    shader.MainFunctionBody() << "        word_offset += " << 8 / a.NumComponents() << ";\n"
                              << "      }\n"
                                 "    }\n"
                                 "  }\n"
                                 "  workgroupBarrier();\n"
                              << "  if (local_id.x < " << output_number_ << ") {\n"
                              << "    var output_value = output_value_t(0);\n"
                                 "    var workgroup_shared_offset = local_id.x;\n"
                              << "    let blocks_num = min(" << shared_memory_size << ", n_blocks_per_col);\n"
                              << "    for (var b = 0u; b < blocks_num; b++) {\n"
                                 "      output_value += workgroup_shared[workgroup_shared_offset];\n"
                              << "      workgroup_shared_offset += " << output_number_ << ";\n"
                              << "    }\n"
                              << "    " << y.SetByIndices("output_indices_t(batch, row, col + local_id.x)", "output_value") << "\n"
                              << "  }\n";
  }

  return Status::OK();
}

Status DP4AMatMulQuantizeProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("output", ShaderUsage::UseUniform);
  shader.AddOutput("scales", ShaderUsage::UseUniform);
  shader.AdditionalImplementation() << R"ADDNL_FN(
  fn readInput(offset: u32) -> input_a_value_t
  {
    if (offset > uniforms.input_size) {
      return input_a_value_t(0);
    }
    return input_a[offset];
  }
)ADDNL_FN";
  shader.MainFunctionBody() << R"MAIN_FN(
  var local_a : array<vec4<input_a_element_t>, 32>;
  var max_value:vec4<input_a_element_t> = vec4<input_a_element_t>(0);
  for (var idx:u32=0;idx<32;idx+=1)
  {
    local_a[idx] = readInput(workgroup_idx*32 + idx);
    max_value = max(max_value, abs(local_a[idx]));
  }
  var scale = max(max_value.x, max_value.y);
  scale = max(scale, max_value.z);
  scale = max(scale, max_value.w);
  for (var idx:u32=0;idx<32;idx+=1)
  {
    output[workgroup_idx*32+idx] = pack4x8snorm(vec4<f32>(local_a[idx]/scale));
  }
  // 127 is the max value of signed int8 [-127,127] used by pack4x8snorm for 1.0f.
  scales[workgroup_idx] = scale/127;
)MAIN_FN";
  return Status::OK();
}

Status DP4AMatMulNBitsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  shader.AddInput("scales_a", ShaderUsage::UseUniform);
  shader.AddInput("input_b", ShaderUsage::UseUniform);
  shader.AddInput("scales_b", ShaderUsage::UseUniform);
  shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  // This shader implements co-operative matrix multiply. The key idea here is to
  // assume there is a primitive for medium size matrix multiply a subgroup can perform,
  // using all its lanes and pooling all its registers to keep the values in registry.
  //
  // The entire workgroup which has N subgroups first loads a tile into shared memory,
  // Then each subgroup loads a subtile from shared memory into registers and uses
  // the medium size matrix multiply primitive to perform the math.
  // The values for tile/subtile size are chosen to conform to the resource limits
  // of an alderlake/tiger lake gpu. A tile is 64x64, workgroup is 256 threads -
  // therefore there are 16 subgroups and 16 lanes in each subgroup.
  // K the hidden dimension is paged in from RAM at k tile size which is 64.
  // All this puts the shared memory requirement slightly above 16KB.
  // WebGPU limit is 16KB, output is moved to registers instead of SHM to make
  // everything fit in shared memory.
  //
  // Each subgroup performs a 16 x 64 x 16 multiply which is implemented with
  // subgroup shuffle as a placeholder for the day the medium matrix mul primitive
  // becomes available in WGSL. The registry requirements is ~2KB per subgroup, on
  // Alderlake/Tigerlake subgroup has 8KB of registry space pooling the
  // 512B of registry from each lane.
  //
  // The medium size matmul is implemented using dot4I8Packed, so the inputs for
  // this shader require A to be int8 quantized with block size 64. B is regular
  // matmulnbits input with block size 32.

  shader.AdditionalImplementation() << R"ADDNL_FN(
  const tile_size = 64;
  const subtile_size = 16;
  const tile_size_k =  32;
  const vec_factor = 4;
  const u32_factor = 4;
  const tile_size_k_vec = 2;
  const block_size = 32;

  // Shared memory
  var<workgroup> tile_A : array<array<vec4<u32>, tile_size>, tile_size_k_vec>;                     // 64 x 32
  var<workgroup> scale_A : array<output_element_t, tile_size>;                                     // 64 x 1
  var<workgroup> tile_B : array<array<vec4<u32>, tile_size>, tile_size_k_vec>;                     // 64 x 32
  var<workgroup> scale_B : array<output_element_t, tile_size>;                                     // 64 x 1

  fn loadSHMA(a_global_base:u32, kidx_v:u32, row: u32, col: u32)
  {
    let a_global = a_global_base + row;
    if (a_global >= uniforms.M)
    {
      return;
    }
    tile_A[col][row] = input_a[a_global*uniforms.K16+kidx_v+col];
    if (col == 0)
    {
      // kidx_v - covers 16 values of k
      scale_A[row] = scales_a[a_global*(uniforms.K/128) + kidx_v/8];
    }
  }

  fn loadSHMB(b_global_base:u32, kidx_v:u32, row: u32, col: u32)
  {
      let b_global = b_global_base + row;
      if (b_global >= uniforms.N)
      {
        return;
      }

      let b_value = input_b[b_global*uniforms.K16+kidx_v+col];
      var b_value_lower = vec4<i32>(unpack4xU8(b_value[0] & 0x0F0F0F0Fu)) - vec4<i32>(8);
      var b_value_upper = vec4<i32>(unpack4xU8((b_value[0] >> 4) & 0x0F0F0F0Fu)) - vec4<i32>(8);
      tile_B[col][row][0] = pack4xI8(vec4<i32>(b_value_lower[0], b_value_upper[0], b_value_lower[1], b_value_upper[1]));
      tile_B[col][row][1] = pack4xI8(vec4<i32>(b_value_lower[2], b_value_upper[2], b_value_lower[3], b_value_upper[3]));
      b_value_lower = vec4<i32>(unpack4xU8(b_value[1] & 0x0F0F0F0Fu)) - vec4<i32>(8);
      b_value_upper = vec4<i32>(unpack4xU8((b_value[1] >> 4) & 0x0F0F0F0Fu)) - vec4<i32>(8);
      tile_B[col][row][2] = pack4xI8(vec4<i32>(b_value_lower[0], b_value_upper[0], b_value_lower[1], b_value_upper[1]));
      tile_B[col][row][3] = pack4xI8(vec4<i32>(b_value_lower[2], b_value_upper[2], b_value_lower[3], b_value_upper[3]));
      if (col == 0)
      {
        // kidx_v - each kidx_v covers 16 values of k
        scale_B[row] = scales_b[b_global*(uniforms.K/32) + kidx_v/2];
      }
  }

  // Scaled dot product of 8 packed unsigned integers.
  fn SDP8AI(a1:vec4<u32>, b1:vec4<u32>, a2:vec4<u32>, b2:vec4<u32>, scale:output_element_t) -> output_element_t
  {
      var local_sum = dot4I8Packed(a1[0], b1[0]);
      local_sum += dot4I8Packed(a1[1], b1[1]);
      local_sum += dot4I8Packed(a1[2], b1[2]);
      local_sum += dot4I8Packed(a1[3], b1[3]);
      local_sum += dot4I8Packed(a2[0], b2[0]);
      local_sum += dot4I8Packed(a2[1], b2[1]);
      local_sum += dot4I8Packed(a2[2], b2[2]);
      local_sum += dot4I8Packed(a2[3], b2[3]);
      return output_element_t(local_sum) * scale;
  }
)ADDNL_FN";

  shader.MainFunctionBody() << R"MAIN_FN(
  // During the load phase we use all 256 threads to load 64 rows of A/B.
  // For each row we load tile_size_k_vec (2) vectorized elements, which are 32 elements of K.
  let a_global_base = workgroup_id.x * tile_size;
  let b_global_base = workgroup_id.y * tile_size;
  let load_AorB = u32(local_idx/128);
  let load_row = u32((local_idx%128)/2);
  let load_col = u32(local_idx%2);

  // During the compute phase, we have the 64x64 tile split into
  // subtiles of 16x16. We have a grid of 4x4 subtiles.
  let subtile_id = u32(local_idx / subtile_size);
  let subtile_idx = u32(subtile_id / 4);
  let subtile_idy = u32(subtile_id % 4);
  let base_A = subtile_idx * 16;
  let base_B = subtile_idy * 16;
  // For each subtile we have 16 threads assigned.
  let a_idx = u32(local_idx % subtile_size);

  var lane_output1: vec4<output_element_t>;
  var lane_output2: vec4<output_element_t>;
  var lane_output3: vec4<output_element_t>;
  var lane_output4: vec4<output_element_t>;
  // K's vectrorization is 16 items per index. See input_a/input_b.
  // tile_size_k_vec - is the k tile size in vectorized space (1/16). That is
  // k tile size is 32. In vectorized space that is 32/16 = 2.
  for (var kidx_v:u32 = 0; kidx_v < uniforms.K16; kidx_v+=tile_size_k_vec)
  {
    // Load Phase: Populate shared memory for the workgroup.
    if (load_AorB == 0)
    {
      loadSHMA(a_global_base, kidx_v, load_row, load_col);
    }
    else
    {
      loadSHMB(b_global_base, kidx_v, load_row, load_col);
    }
    workgroupBarrier();

    // Compute phase: Perform matmul for this subtile 16 x 32 x 16.
    // Step 1: Load from shared memory into registers across entire subgroup.
    var own_a0: vec4<u32> = tile_A[0][base_A + a_idx];
    var own_a1: vec4<u32> = tile_A[1][base_A + a_idx];
    var own_scale_a: output_element_t = scale_A[base_A + a_idx];
    if (sg_size == 16)
    {
      var own_b0: vec4<u32> = tile_B[0][base_B + sg_id];
      var own_b1: vec4<u32> = tile_B[1][base_B + sg_id];
      var own_scale_b: output_element_t  = scale_B[base_B + sg_id];
      // Step 2: Access registers across the subgroup using subgroupShuffle and perform the matmul.
      lane_output1[0] += SDP8AI(own_a0, subgroupShuffle(own_b0, 0), own_a1, subgroupShuffle(own_b1, 0), subgroupShuffle(own_scale_b, 0) * own_scale_a);
      lane_output1[1] += SDP8AI(own_a0, subgroupShuffle(own_b0, 1), own_a1, subgroupShuffle(own_b1, 1), subgroupShuffle(own_scale_b, 1) * own_scale_a);
      lane_output1[2] += SDP8AI(own_a0, subgroupShuffle(own_b0, 2), own_a1, subgroupShuffle(own_b1, 2), subgroupShuffle(own_scale_b, 2) * own_scale_a);
      lane_output1[3] += SDP8AI(own_a0, subgroupShuffle(own_b0, 3), own_a1, subgroupShuffle(own_b1, 3), subgroupShuffle(own_scale_b, 3) * own_scale_a);

      lane_output2[0] += SDP8AI(own_a0, subgroupShuffle(own_b0, 4), own_a1, subgroupShuffle(own_b1, 4), subgroupShuffle(own_scale_b, 4) * own_scale_a);
      lane_output2[1] += SDP8AI(own_a0, subgroupShuffle(own_b0, 5), own_a1, subgroupShuffle(own_b1, 5), subgroupShuffle(own_scale_b, 5) * own_scale_a);
      lane_output2[2] += SDP8AI(own_a0, subgroupShuffle(own_b0, 6), own_a1, subgroupShuffle(own_b1, 6), subgroupShuffle(own_scale_b, 6) * own_scale_a);
      lane_output2[3] += SDP8AI(own_a0, subgroupShuffle(own_b0, 7), own_a1, subgroupShuffle(own_b1, 7), subgroupShuffle(own_scale_b, 7) * own_scale_a);

      lane_output3[0] += SDP8AI(own_a0, subgroupShuffle(own_b0, 8), own_a1, subgroupShuffle(own_b1, 8), subgroupShuffle(own_scale_b, 8) * own_scale_a);
      lane_output3[1] += SDP8AI(own_a0, subgroupShuffle(own_b0, 9), own_a1, subgroupShuffle(own_b1, 9), subgroupShuffle(own_scale_b, 9) * own_scale_a);
      lane_output3[2] += SDP8AI(own_a0, subgroupShuffle(own_b0, 10), own_a1, subgroupShuffle(own_b1, 10), subgroupShuffle(own_scale_b, 10) * own_scale_a);
      lane_output3[3] += SDP8AI(own_a0, subgroupShuffle(own_b0, 11), own_a1, subgroupShuffle(own_b1, 11), subgroupShuffle(own_scale_b, 11) * own_scale_a);

      lane_output4[0] += SDP8AI(own_a0, subgroupShuffle(own_b0, 12), own_a1, subgroupShuffle(own_b1, 12), subgroupShuffle(own_scale_b, 12) * own_scale_a);
      lane_output4[1] += SDP8AI(own_a0, subgroupShuffle(own_b0, 13), own_a1, subgroupShuffle(own_b1, 13), subgroupShuffle(own_scale_b, 13) * own_scale_a);
      lane_output4[2] += SDP8AI(own_a0, subgroupShuffle(own_b0, 14), own_a1, subgroupShuffle(own_b1, 14), subgroupShuffle(own_scale_b, 14) * own_scale_a);
      lane_output4[3] += SDP8AI(own_a0, subgroupShuffle(own_b0, 15), own_a1, subgroupShuffle(own_b1, 15), subgroupShuffle(own_scale_b, 15) * own_scale_a);
    }
    else
    {
      // Code for other subgroup sizes, simply doesnt use subgroups at all.
      // Relies on reads from single location tile_B[][base_B + col] by all
      // being optimized by the hardware.
      lane_output1[0] += SDP8AI(own_a0, tile_B[0][base_B + 0], own_a1, tile_B[1][base_B + 0],  own_scale_a * scale_B[base_B + 0]);
      lane_output1[1] += SDP8AI(own_a0, tile_B[0][base_B + 1], own_a1, tile_B[1][base_B + 1],  own_scale_a * scale_B[base_B + 1]);
      lane_output1[2] += SDP8AI(own_a0, tile_B[0][base_B + 2], own_a1, tile_B[1][base_B + 2],  own_scale_a * scale_B[base_B + 2]);
      lane_output1[3] += SDP8AI(own_a0, tile_B[0][base_B + 3], own_a1, tile_B[1][base_B + 3],  own_scale_a * scale_B[base_B + 3]);

      lane_output2[0] += SDP8AI(own_a0, tile_B[0][base_B + 4], own_a1, tile_B[1][base_B + 4],  own_scale_a * scale_B[base_B + 4]);
      lane_output2[1] += SDP8AI(own_a0, tile_B[0][base_B + 5], own_a1, tile_B[1][base_B + 5],  own_scale_a * scale_B[base_B + 5]);
      lane_output2[2] += SDP8AI(own_a0, tile_B[0][base_B + 6], own_a1, tile_B[1][base_B + 6],  own_scale_a * scale_B[base_B + 6]);
      lane_output2[3] += SDP8AI(own_a0, tile_B[0][base_B + 7], own_a1, tile_B[1][base_B + 7],  own_scale_a * scale_B[base_B + 7]);

      lane_output3[0] += SDP8AI(own_a0, tile_B[0][base_B + 8], own_a1, tile_B[1][base_B + 8],  own_scale_a * scale_B[base_B + 8]);
      lane_output3[1] += SDP8AI(own_a0, tile_B[0][base_B + 9], own_a1, tile_B[1][base_B + 9],  own_scale_a * scale_B[base_B + 9]);
      lane_output3[2] += SDP8AI(own_a0, tile_B[0][base_B + 10], own_a1, tile_B[1][base_B + 10],  own_scale_a * scale_B[base_B + 10]);
      lane_output3[3] += SDP8AI(own_a0, tile_B[0][base_B + 11], own_a1, tile_B[1][base_B + 11],  own_scale_a * scale_B[base_B + 11]);

      lane_output4[0] += SDP8AI(own_a0, tile_B[0][base_B + 12], own_a1, tile_B[1][base_B + 12],  own_scale_a * scale_B[base_B + 12]);
      lane_output4[1] += SDP8AI(own_a0, tile_B[0][base_B + 13], own_a1, tile_B[1][base_B + 13],  own_scale_a * scale_B[base_B + 13]);
      lane_output4[2] += SDP8AI(own_a0, tile_B[0][base_B + 14], own_a1, tile_B[1][base_B + 14],  own_scale_a * scale_B[base_B + 14]);
      lane_output4[3] += SDP8AI(own_a0, tile_B[0][base_B + 15], own_a1, tile_B[1][base_B + 15],  own_scale_a * scale_B[base_B + 15]);
    }
    workgroupBarrier();
  }

  let a_global = a_global_base + base_A + a_idx;
  let b_global = b_global_base + base_B;
  let output_idx = ((a_global) * uniforms.N + b_global)/4;
  // This creates a shader requirement that uniforms.N % 16 == 0
  if (a_global < uniforms.M && b_global < uniforms.N)
  {
    output[output_idx] = lane_output1;
    output[output_idx+1] = lane_output2;
    output[output_idx+2] = lane_output3;
    output[output_idx+3] = lane_output4;
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
  const uint32_t data_size = gsl::narrow<uint32_t>(y->Shape().Size());
  if (data_size == 0) {
    return Status::OK();
  }

  const uint32_t batch_count = gsl::narrow<uint32_t>(helper.OutputOffsets().size());
  const uint32_t M = gsl::narrow<uint32_t>(helper.M());
  const uint32_t N = gsl::narrow<uint32_t>(helper.N());
  const uint32_t K = gsl::narrow<uint32_t>(helper.K());
  const uint32_t block_size = gsl::narrow<uint32_t>(block_size_);
  constexpr uint32_t nbits = 4;

  const uint32_t n_blocks_per_col = (K + block_size - 1) / block_size;
  const uint32_t blob_size = (block_size / 8) * nbits;
  const uint32_t blob_size_in_words = blob_size / 4;
  const uint32_t components_a = GetMaxComponents(K);
  const uint32_t components_b = GetMaxComponents(blob_size_in_words);
  uint32_t components = GetMaxComponents(N);

  const bool has_zero_points = zero_points != nullptr;
  // macOS - Experimental dawn support for subgroup matrix matmul on Metal.
  if (M >= kMinMForTileOptimization &&
      CanApplySubgroupMatrixMatMulNBits(context, accuracy_level_, block_size, batch_count, N, K, has_zero_points)) {
    return ApplySubgroupMatrixMatMulNBits(a, b, scales, M, N, K, context, y);
  }

  const bool has_subgroup = context.Device().HasFeature(wgpu::FeatureName::Subgroups);
  // macOS - Avoid using dp4a on Metal, as it does not appear to have native dp4a support.
  // https://github.com/gpuweb/gpuweb/issues/2677#issuecomment-1713292226
  const bool use_dp4a = has_subgroup && context.AdapterInfo().backendType != wgpu::BackendType::Metal;
  if (accuracy_level_ == 4 && block_size == 32 &&
      batch_count == 1 && components_a == 4 && K % 64 == 0 && N % 16 == 0 &&
      !has_zero_points && use_dp4a && M >= kMinMForTileOptimization) {
    constexpr uint32_t kVec4Components = 4;
    constexpr uint32_t kVec2Components = 2;
    constexpr uint32_t kU32Components = 4;

    constexpr uint32_t kBlockSizeA = 128;
    DP4AMatMulQuantizeProgram quantize_program;
    quantize_program.SetWorkgroupSize(1);
    quantize_program.SetDispatchGroupSize(M * K / kBlockSizeA, 1, 1);
    TensorShape a_quant_shape{1, M, K / kU32Components};
    Tensor a_quant = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), a_quant_shape);
    TensorShapeVector a_scales_dims({1, 1, M, K / kBlockSizeA});
    Tensor a_scale = context.CreateGPUTensor(a->DataType(), a_scales_dims);
    quantize_program.AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, gsl::narrow<int>(kVec4Components)}})
        .AddOutputs({{&a_quant, ProgramTensorMetadataDependency::Rank, a_quant.Shape(), gsl::narrow<int>(1)},
                     {&a_scale, ProgramTensorMetadataDependency::Rank, a_scale.Shape(), gsl::narrow<int>(1)}})
        .AddUniformVariable({static_cast<uint32_t>(M * K / kVec4Components)});
    ORT_RETURN_IF_ERROR(context.RunProgram(quantize_program));

    constexpr uint32_t kTileSize = 64;
    TensorShape reshaped_y_shape{1, M, N / kVec4Components};
    DP4AMatMulNBitsProgram mul_program;
    mul_program.SetWorkgroupSize(256);
    mul_program.SetDispatchGroupSize(
        (M + kTileSize - 1) / kTileSize,
        (N + kTileSize - 1) / kTileSize, 1);
    mul_program.AddInputs({{&a_quant, ProgramTensorMetadataDependency::TypeAndRank, gsl::narrow<int>(kVec4Components)},
                           {&a_scale, ProgramTensorMetadataDependency::TypeAndRank, gsl::narrow<int>(1)},
                           {b, ProgramTensorMetadataDependency::TypeAndRank, gsl::narrow<int>(kVec2Components * kU32Components)},
                           {scales, ProgramTensorMetadataDependency::TypeAndRank, gsl::narrow<int>(1)}})
        .AddUniformVariables({{static_cast<uint32_t>(M)},
                              {static_cast<uint32_t>(N)},
                              {static_cast<uint32_t>(K)},
                              {static_cast<uint32_t>(K / 8)},
                              {static_cast<uint32_t>(K / 16)}})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, reshaped_y_shape, gsl::narrow<int>(kVec4Components)});
    return context.RunProgram(mul_program);
  }

  // TODO: Support output_number > 1. Some cases are failed when output_number > 1.
  constexpr uint32_t output_number = 1;
  const uint32_t tile_m = M > kMinMForTileOptimization ? 4 : 1;
  const bool use_subgroup = has_subgroup && context.AdapterInfo().vendor == std::string_view{"intel"} && components_a == 4 && block_size == 32;
  MatMulNBitsProgram program{output_number, block_size, tile_m, gsl::narrow<int>(components_b), has_zero_points, use_subgroup};
  if (M > kMinMForTileOptimization && block_size == 32) {
    components = 1;
    constexpr uint32_t workgroup_size = 64;
    constexpr uint32_t workgroup_y = 8;
    constexpr uint32_t workgroup_x = workgroup_size / workgroup_y;
    program.SetWorkgroupSize(workgroup_x, workgroup_y, 1);
    program.SetDispatchGroupSize((N + workgroup_y - 1) / workgroup_y,
                                 (M + tile_m - 1) / tile_m,
                                 batch_count);
    program.CacheHint("T_M" + std::to_string(tile_m) + "Subgroup" + std::to_string(use_subgroup));
  } else if (block_size == 32) {
    components = 1;
    constexpr uint32_t workgroup_size = 64;
    const uint32_t workgroup_y = N % 8 == 0 ? 8 : 1;
    const uint32_t workgroup_x = workgroup_size / workgroup_y;
    program.SetWorkgroupSize(workgroup_x, workgroup_y, 1);
    program.SetDispatchGroupSize(data_size / components / workgroup_y);
    program.CacheHint("T_M" + std::to_string(tile_m));
  } else {
    program.SetDispatchGroupSize(data_size / components / output_number);
    program.CacheHint("O_N" + std::to_string(output_number));
  }

  TensorShape reshaped_a_shape{batch_count, M, K / components_a};
  TensorShape reshaped_b_shape{N, n_blocks_per_col, blob_size_in_words / components_b};
  TensorShape reshaped_y_shape{batch_count, M, N / components};

  program
      .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, reshaped_a_shape, gsl::narrow<int>(components_a)},
                  {b, ProgramTensorMetadataDependency::TypeAndRank, reshaped_b_shape, gsl::narrow<int>(components_b * 4 /** b will be accessed as uint32 which includs 4 uint8. So here we need to multiply 4.*/)},
                  {scales, ProgramTensorMetadataDependency::None}})
      .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, reshaped_y_shape, gsl::narrow<int>(components)})
      .AddUniformVariable({block_size});
  if (has_zero_points) {
    program.AddInput({zero_points, ProgramTensorMetadataDependency::None, {(zero_points->Shape().Size() + 3) / 4}, 4});
  }
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
