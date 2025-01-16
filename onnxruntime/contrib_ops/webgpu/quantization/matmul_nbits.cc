// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string_view>

#include "contrib_ops/webgpu/quantization/matmul_nbits.h"
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
  shader.AddInput("input_a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  shader.AddOutput("output", ShaderUsage::UseUniform);
  shader.AddOutput("scales", ShaderUsage::UseUniform);

  shader.AdditionalImplementation() << R"ADDNL_FN(
    var<workgroup> max_values : array<f16, 2>;
 )ADDNL_FN";

  shader.MainFunctionBody() << R"MAIN_FN(
    var local_a = input_a[global_idx];
    var max_val = subgroupMax(abs(local_a));
    var max_temp = max(max_val.xy, max_val.zw);
    var scale = max(max_temp[0], max_temp[1]);
    if (sg_size == 8)
    {
      if (local_idx % sg_size == 0) {
        max_values[local_idx / sg_size] = scale;
      }
      workgroupBarrier();
      scale = max(max_values[0], max_values[1]);
    }
    var norm_a = local_a/scale;
    output[global_idx] = pack4x8snorm(vec4<f32>(norm_a));
    if (local_idx == 0)
    {
      // 127 is the max value of signed int8 [-127,127] used by pack4x8snorm for 1.0f.
      scales[workgroup_idx] = scale/127;
    }
)MAIN_FN";
  return Status::OK();
}

Status DP4AMatMulDeQuantizeProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input", ShaderUsage::UseUniform);
  shader.AddInput("scales", ShaderUsage::UseUniform);
  shader.AddOutput("output", ShaderUsage::UseUniform);

  shader.MainFunctionBody() << R"MAIN_FN(
  var local_a = unpack4xI8(input[workgroup_id.x*16 + local_idx/4]);
  output[workgroup_id.x*64+local_idx ] = f16(local_a[local_idx%4]) * scales[workgroup_id.x];
)MAIN_FN";

  return Status::OK();
}

Status DP4AMatMulNBitsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  shader.AddInput("scales_a", ShaderUsage::UseUniform);
  shader.AddInput("input_b", ShaderUsage::UseUniform);
  shader.AddInput("scales_b", ShaderUsage::UseUniform);
  shader.AddOutput("output", ShaderUsage::UseUniform);

  // This shader implements co-operative matrix multiply. The key idea here is to
  // assume there is a primitive for medium size matrix multiply a subgroup can perform,
  // using all its lanes and pooling all its registers to keep the values in registry.
  //
  // The entire workgroup which has N subgroups first loads a tile into shared memory,
  // Then each subgroup loads a subtile from shared memory into registers and uses
  // the medium size matrix multiply primitive to perform the math.
  // The values for tile/subtile size are choosen to conform to the resource limits
  // of an alderlake/tiger lake gpu. A tile is 64x64, workgroup is 256 threads -
  // therefore there are 16 subgroups and 16 lanes in each subgroup.
  // K the hidden dimension is paged in from RAM at k tile size which is 64.
  // All this puts the shared memory requirement slightly above 16KB.
  // WebGPU limit is 16KB, output is moved to registers instead of SHM to make
  // everything fit in shared memory.
  //
  // Each subgroup performs a 16 x 64 x 16 multiply which is implemented with
  // subgroup shuffle as a placeholder for the day the medium matrix mul primitive
  // becomes available in WGSL. The registy requirements is ~2KB per subgroup, on
  // Alderlake/Tigerlake subgroup has 8KB of registry space pooling the
  // 512B of registery from each lane.
  //
  // The medium size matmul is implemented using dot4I8Packed, so the inputs for
  // this shader require A to be int8 quantized with block size 64. B is regular
  // matmulnbits input with block size 32.

  shader.AdditionalImplementation() << R"ADDNL_FN(
  const tile_size_a = 64;
  const tile_size_b = 64;
  const tile_size_k =  64;
  const vec_factor = 4;
  const u32_factor = 4;
  const tile_size_kv = 4;
  const block_size = 32;

  // Shared memory
  var<workgroup> tile_A : array<array<vec4<u32>, tile_size_kv>, tile_size_a>;                        // 64 x 64
  var<workgroup> scale_A : array<f16, tile_size_a>;                                                  // 64 x 1
  var<workgroup> tile_B : array<array<vec4<u32>, tile_size_kv>, tile_size_b>;                        // 64 x 64
  var<workgroup> scale_B : array<vec2<f16>, tile_size_b>;                                            // 64 x 2

  // Private memory
  var<private> lane_output: array<f16, 16>;

  fn loadSHMA(a_global_base:u32, kidx_v:u32, subtile_id: u32, sg_id: u32)
  {
    // Workgroup needs to fill 64 slots, there are 16 subgroups in the workgroup.
    // Each subgroup fills out 4 slots.
    let a_slot = subtile_id * 4 + sg_id/4;
    let a_global = a_global_base + a_slot;
    if (a_global >= uniforms.M)
    {
      return;
    }
    tile_A[a_slot][sg_id%4] = input_a[a_global*uniforms.K16+kidx_v+sg_id%4];
    if (sg_id%4 == 0)
    {
      // kidx_v - each kidx_v covers 16 values of k, therefore 4 index must share
      // a single scale kidx_v/4.
      scale_A[a_slot] = scales_a[a_global*(uniforms.K/64) + kidx_v/4];
    }
  }

  fn loadSHMB(b_global_base:u32, kidx_v:u32, subtile_id: u32, sg_id: u32)
  {
      // Workgroup needs to fill 64 slots, there are 16 subgroups in the workgroup.
      // Each subgroup fills out 4 slots.
      let b_slot = subtile_id * 4 + sg_id/4;
      let weight_offset:u32 = kidx_v + sg_id%4;
      let b_global = b_global_base + b_slot;
      if (b_global >= uniforms.N)
      {
        return;
      }

      let b_value = input_b[b_global*uniforms.K16+weight_offset];
      var b_value_lower = vec4<i32>(unpack4xU8(b_value[0] & 0x0F0F0F0Fu)) - vec4<i32>(8);
      var b_value_upper = vec4<i32>(unpack4xU8((b_value[0] >> 4) & 0x0F0F0F0Fu)) - vec4<i32>(8);
      tile_B[b_slot][sg_id%4][0] = pack4xI8(vec4<i32>(b_value_lower[0], b_value_upper[0], b_value_lower[1], b_value_upper[1]));
      tile_B[b_slot][sg_id%4][1] = pack4xI8(vec4<i32>(b_value_lower[2], b_value_upper[2], b_value_lower[3], b_value_upper[3]));
      b_value_lower = vec4<i32>(unpack4xU8(b_value[1] & 0x0F0F0F0Fu)) - vec4<i32>(8);
      b_value_upper = vec4<i32>(unpack4xU8((b_value[1] >> 4) & 0x0F0F0F0Fu)) - vec4<i32>(8);
      tile_B[b_slot][sg_id%4][2] = pack4xI8(vec4<i32>(b_value_lower[0], b_value_upper[0], b_value_lower[1], b_value_upper[1]));
      tile_B[b_slot][sg_id%4][3] = pack4xI8(vec4<i32>(b_value_lower[2], b_value_upper[2], b_value_lower[3], b_value_upper[3]));
      if (sg_id%4 == 0)
      {
        // kidx_v - each kidx_v covers 16 values of k, therefore 4 index must share
        // a single scale kidx_v/2. But scales_b is a vec2, making the math same as A.
        scale_B[b_slot] = scales_b[b_global*(uniforms.K/64) + kidx_v/4];
      }
  }

  fn DP4AI(a:vec4<u32>, b:vec4<u32>) -> i32
  {
      var local_sum = dot4I8Packed(a[0], b[0]);
      local_sum += dot4I8Packed(a[1], b[1]);
      local_sum += dot4I8Packed(a[2], b[2]);
      local_sum += dot4I8Packed(a[3], b[3]);
      return local_sum;
  }

  // Implement a 16x64x16 matrix multipy primitive using the 16 lanes of the subgroup.
  // Each Lane first loads a row of A and a column of B.
  // Then each lane becomes responsible for a row and loops through the columns to compute
  // dot product. The columns are fetched from other owning lanes using subgroupShuffle.
  fn perform16_64_16MatMul(base_A:u32, base_B:u32, sg_id:u32)
  {
    var own_a0: vec4<u32> = tile_A[base_A + sg_id][0];
    var own_a1: vec4<u32> = tile_A[base_A + sg_id][1];
    var own_a2: vec4<u32> = tile_A[base_A + sg_id][2];
    var own_a3: vec4<u32> = tile_A[base_A + sg_id][3];
    var own_b0: vec4<u32> = tile_B[base_B + sg_id][0];
    var own_b1: vec4<u32> = tile_B[base_B + sg_id][1];
    var own_b2: vec4<u32> = tile_B[base_B + sg_id][2];
    var own_b3: vec4<u32> = tile_B[base_B + sg_id][3];
    var own_scale_b: vec2<f16> = scale_B[base_B + sg_id];
    var own_scale_a: f16 = scale_A[base_A+sg_id];;

    for (var col:u32 = 0; col < 16; col++)
    {
      var local_scale_b = subgroupShuffle(own_scale_b, col);
      local_scale_b = local_scale_b * own_scale_a;
      var local_b = own_b0;
      local_b = subgroupShuffle(local_b, col);
      var local_sum = DP4AI(own_a0, local_b);
      local_b = own_b1;
      local_b = subgroupShuffle(local_b, col);
      local_sum += DP4AI(own_a1, local_b);
      local_b = own_b2;
      local_b = subgroupShuffle(local_b, col);
      var local_sum2 = DP4AI(own_a2, local_b);
      local_b = own_b3;
      local_b = subgroupShuffle(local_b, col);
      local_sum2 += DP4AI(own_a3, local_b);
      lane_output[col] += (f16(local_sum) * local_scale_b[0] + f16(local_sum2) * local_scale_b[1]);
    }
  }

)ADDNL_FN";

  shader.MainFunctionBody() << R"MAIN_FN(
  let subtile_id = u32(local_idx / sg_size);
  let subtile_idx = u32(subtile_id / 4);
  let subtile_idy = u32(subtile_id % 4);

  let a_global_base = workgroup_id.x * tile_size_a;
  let b_global_base = workgroup_id.y * tile_size_b;

  // Clear the accumulator
  for (var i:u32 = 0; i < 16; i++)
  {
    lane_output[i] = 0;
  }

  // K's vectrorization is 16 items per index. See input_a/input_b.
  // tile_size_kv - is the k tile size in vectorized k units/space (1/16).
  for (var kidx_v:u32 = 0; kidx_v < uniforms.K16; kidx_v+=tile_size_kv)
  {
    // Populate shared memory for the workgroup
    loadSHMA(a_global_base, kidx_v, subtile_id, sg_id);
    loadSHMB(b_global_base, kidx_v, subtile_id, sg_id);
    workgroupBarrier();

    // Perform the matmul for the subtile this subgroup is responsible for.
    perform16_64_16MatMul(subtile_idx*16, subtile_idy*16, sg_id);
    workgroupBarrier();
  }

  let a_global = a_global_base + subtile_idx * 16+ sg_id;
  let b_global = b_global_base + subtile_idy * 16;
  let output_idx = ((a_global) * uniforms.N + b_global)/4;
  // This creates a shader requirement that uniforms.N % 16 == 0
  if (a_global < uniforms.M && b_global < uniforms.N)
  {
    for (var i:u32 = 0; i < 4; i++)
    {
      let lidx = i * 4;
      output[output_idx+i] = vec4<f16>(lane_output[lidx], lane_output[lidx+1] , lane_output[lidx+2], lane_output[lidx+3]);
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

  Tensor a_clone = context.CreateGPUTensor(a->DataType(), a->Shape());
  if (block_size == 32 && batch_count == 1 &&
      components_a == 4 && K % 64 == 0 &&
      !has_zero_points && M >= kMinMForTileOptimization) {
    constexpr uint32_t kVec4Components = 4;
    constexpr uint32_t kVec2Components = 2;
    constexpr uint32_t kU32Components = 4;

    constexpr uint32_t kBlockSizeA = 64;
    DP4AMatMulQuantizeProgram quantize_program;
    quantize_program.SetWorkgroupSize(16);
    quantize_program.SetDispatchGroupSize(M*K/kBlockSizeA, 1, 1);
    TensorShape a_quant_shape{1, M, K/kU32Components};
    Tensor a_quant = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), a_quant_shape);
    TensorShapeVector a_scales_dims({1, 1, M, K/kBlockSizeA});
    Tensor a_scale = context.CreateGPUTensor(a->DataType(), a_scales_dims);
    quantize_program.AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, gsl::narrow<int>(kVec4Components)}})
          .AddOutputs({{&a_quant, ProgramTensorMetadataDependency::Rank, a_quant.Shape(), gsl::narrow<int>(1)},
                      {&a_scale, ProgramTensorMetadataDependency::Rank, a_scale.Shape(), gsl::narrow<int>(1)}});
    ORT_RETURN_IF_ERROR(context.RunProgram(quantize_program));

    // Enable this block and disable the matmul block below to test if quantization is correct.
    // DP4AMatMulDeQuantizeProgram dequantize_program;
    // dequantize_program.SetWorkgroupSize(64);
    // dequantize_program.SetDispatchGroupSize(M*K/kBlockSizeA, 1, 1);
    // dequantize_program.AddInputs({
    //   {&a_quant, ProgramTensorMetadataDependency::TypeAndRank, gsl::narrow<int>(1)},
    //   {&a_scale, ProgramTensorMetadataDependency::TypeAndRank, gsl::narrow<int>(1)}})
    //   .AddOutput({&a_clone, ProgramTensorMetadataDependency::Rank, a->Shape(), gsl::narrow<int>(1)});
    // ORT_RETURN_IF_ERROR(context.RunProgram(dequantize_program));
    // a = &a_clone;

    constexpr uint32_t kTileSize = 64;
    TensorShape reshaped_y_shape{1, M, N / kVec4Components};
    DP4AMatMulNBitsProgram mul_program;
    mul_program.SetWorkgroupSize(256);
    mul_program.SetDispatchGroupSize(
      (M + kTileSize - 1) / kTileSize,
      (N + kTileSize - 1) / kTileSize, 1);
    mul_program.AddInputs({
        {&a_quant, ProgramTensorMetadataDependency::TypeAndRank, gsl::narrow<int>(kVec4Components)},
        {&a_scale, ProgramTensorMetadataDependency::TypeAndRank, gsl::narrow<int>(1)},
        {b, ProgramTensorMetadataDependency::TypeAndRank, gsl::narrow<int>(kVec2Components * kU32Components)},
        {scales, ProgramTensorMetadataDependency::TypeAndRank, gsl::narrow<int>(kVec2Components)}})
      .AddUniformVariables({{static_cast<uint32_t>(M)},
                            {static_cast<uint32_t>(N)},
                            {static_cast<uint32_t>(K)},
                            {static_cast<uint32_t>(K/8)},
                            {static_cast<uint32_t>(K/16)}})
      .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, reshaped_y_shape, gsl::narrow<int>(kVec4Components)});
    return context.RunProgram(mul_program);
  }

  // TODO: Support output_number > 1. Some cases are failed when output_number > 1.
  constexpr uint32_t output_number = 1;
  const uint32_t tile_m = M > kMinMForTileOptimization ? 4 : 1;
  const bool use_subgroup = context.Device().HasFeature(wgpu::FeatureName::Subgroups) && context.AdapterInfo().vendor == std::string_view{"intel"} && components_a == 4 && block_size == 32;
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
