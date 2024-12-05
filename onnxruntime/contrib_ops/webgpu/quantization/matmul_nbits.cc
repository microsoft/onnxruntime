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

  if (use_block32_) {
    const uint32_t workgroup_size = WorkgroupSizeX() * WorkgroupSizeY();
    const uint32_t tile_size = WorkgroupSizeX() * components_b_ * 8;  // each uint32 has 8 data.
    const uint32_t a_length_per_tile = tile_size / a.NumComponents();
    constexpr uint32_t block_size = 32;
    const uint32_t blocks_per_tile = tile_size / block_size;
    shader.AdditionalImplementation() << "var<workgroup> sub_a: array<input_a_value_t, " << a_length_per_tile << ">;\n"
                                      << "var<workgroup> inter_results: array<array<output_value_t, " << WorkgroupSizeX() << ">, " << WorkgroupSizeY() << ">;\n";
    std::string offset = "workgroup_idx * " + std::to_string(WorkgroupSizeY());
    shader.MainFunctionBody() << "  let output_indices = " << y.OffsetToIndices(offset) << ";\n"
                              << "  let col = output_indices[2];\n"
                                 "  let row = output_indices[1];\n"
                                 "  let batch = output_indices[0];\n"
                                 "  let n_blocks_per_col = uniforms.input_b_shape[1];\n"
                              << "  let num_tiles =  (n_blocks_per_col - 1) / " << blocks_per_tile << " + 1;\n"
                              // Loop over shared dimension.
                              << "  for (var tile: u32 = 0; tile < num_tiles; tile += 1) {\n"
                              << "    let a_col_start = tile * " << a_length_per_tile << ";\n"
                              << "    // load one tile A data into shared memory.\n"
                              << "    for (var a_offset = local_idx; a_offset < " << a_length_per_tile << "; a_offset += " << workgroup_size << ") {\n"
                              << "      let a_col = a_col_start + a_offset;\n"
                                 "      if (a_col < uniforms.input_a_shape[2]) {\n"
                              << "        sub_a[a_offset] = " << a.GetByIndices("input_a_indices_t(batch, row, a_col)") << ";\n"
                              << "      } else {\n"
                                 "        sub_a[a_offset] = input_a_value_t(0);\n"
                                 "      }\n"
                                 "    }\n"
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
                              << "    var word_offset = local_id.x * " << block_size / a.NumComponents() << ";\n"
                              << "    for (var i: u32 = 0; i < " << components_b_ << "; i++) {\n";
    switch (a.NumComponents()) {
      case 1:
        shader.MainFunctionBody() << "      let a_data0 = vec4<output_element_t>(sub_a[word_offset], sub_a[word_offset + 1], sub_a[word_offset + 2], sub_a[word_offset + 3]);\n"
                                     "      let a_data1 = vec4<output_element_t>(sub_a[word_offset + 4], sub_a[word_offset + 5], sub_a[word_offset + 6], sub_a[word_offset + 7]);\n";
        break;
      case 2:
        shader.MainFunctionBody() << "      let a_data0 = vec4<output_element_t>(sub_a[word_offset], sub_a[word_offset + 1]);\n"
                                     "      let a_data1 = vec4<output_element_t>(sub_a[word_offset + 2], sub_a[word_offset + 3]);\n";
        break;
      case 4:
        shader.MainFunctionBody() << "      let a_data0 = sub_a[word_offset];\n"
                                     "      let a_data1 = sub_a[word_offset + 1];\n";
        break;
      default:
        break;
    }
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
    shader.MainFunctionBody() << ")) * scale;\n"
                                 "      inter_results[local_id.y][local_id.x] += dot(a_data0, b_dequantized_values[0]) + dot(a_data1, b_dequantized_values[1]);\n"
                              << "      word_offset += " << 8 / a.NumComponents() << ";\n"
                              << "    }\n"
                                 "    workgroupBarrier();\n"
                                 "  }\n"
                              << "  if (local_idx < " << WorkgroupSizeY() << ") {\n"
                              << "    var output_value = output_value_t(0);\n"
                              << "    for (var b = 0u; b < " << WorkgroupSizeX() << "; b++) {\n"
                              << "      output_value += inter_results[local_idx][b];\n"
                                 "    }\n"
                                 "    if (col + local_idx < uniforms.output_shape[2]) {\n"
                              << "      " << y.SetByIndices("output_indices_t(batch, row, col + local_idx)", "output_value") << ";\n"
                              << "    }\n"
                                 "  }\n";
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

  // Use block32 for Intel Gen12LP architecture.
  const bool use_block32 = context.AdapterInfo().vendor == std::string_view{"intel"} &&
                           context.AdapterInfo().architecture == std::string_view{"gen-12lp"} &&
                           block_size == 32;
  const bool has_zero_points = zero_points != nullptr;
  // TODO: Support output_number > 1. Some cases are failed when output_number > 1.
  // const uint32_t output_number = M > 1 && (N / components) % 2 == 0 ? 2 : 1;
  constexpr uint32_t output_number = 1;
  MatMulNBitsProgram program{output_number, gsl::narrow<int>(components_b), has_zero_points, use_block32};

  if (use_block32) {
    components = 1;
    constexpr uint32_t workgroup_size = 128;
    const uint32_t workgroup_y = N % 8 == 0 ? 8 : N % 4 == 0 ? 4
                                                             : 1;
    const uint32_t workgroup_x = workgroup_size / workgroup_y;
    program.SetWorkgroupSize(workgroup_x, workgroup_y, 1);
    program.SetDispatchGroupSize(data_size / components / workgroup_y);
  } else {
    program.SetDispatchGroupSize(data_size / components / output_number);
  }

  TensorShape reshaped_a_shape{batch_count, M, K / components_a};
  TensorShape reshaped_b_shape{N, n_blocks_per_col, blob_size_in_words / components_b};
  TensorShape reshaped_y_shape{batch_count, M, N / components};

  program
      .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, reshaped_a_shape, gsl::narrow<int>(components_a)},
                  {b, ProgramTensorMetadataDependency::TypeAndRank, reshaped_b_shape, gsl::narrow<int>(components_b * 4 /** b will be accessed as uint32 which includs 4 uint8. So here we need to multiply 4.*/)},
                  {scales, ProgramTensorMetadataDependency::None}})
      .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, reshaped_y_shape, gsl::narrow<int>(components)})
      .AddUniformVariable({block_size})
      .CacheHint(std::to_string(output_number));
  if (has_zero_points) {
    program.AddInput({zero_points, ProgramTensorMetadataDependency::None, {(zero_points->Shape().Size() + 3) / 4}, 4});
  }
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
