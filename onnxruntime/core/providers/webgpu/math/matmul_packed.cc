// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/matmul_packed.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include <string>
namespace onnxruntime {
namespace webgpu {

void MatMulProgram::MatMulReadWriteFnSource(ShaderHelper& shader,
                                            const ShaderVariableHelper& a,
                                            const ShaderVariableHelper& b,
                                            const ShaderVariableHelper& output,
                                            const ShaderIndicesHelper& batch_dims,
                                            std::string activation_snippet) const {
  int components = is_vec4_ ? 4 : 1;
  const std::string data_type = "a_element_t";
  const std::string type_string = MakeScalarOrVectorType(components, data_type);

  // Add the mm_readA function
  shader.AdditionalImplementation()
      << "fn mm_readA(batch: i32, row: i32, colIn: i32, batch_indices: batch_dims_indices_t) -> " << type_string << " {\n"
      << "    var value = " << type_string << "(0.0);\n"
      << "    let col = colIn * " << components << ";\n"
      << "    if(row < i32(uniforms.dim_a_outer) && col < i32(uniforms.dim_inner)) {\n"
      << "        var a_indices: a_indices_t;\n"
      << ConvertOutputBatchIndicesToInputBatchIndices("a", a, a.Rank() - 2, batch_dims.Rank(), "batch_indices")
      << a.IndicesSet("a_indices", a.Rank() - 2, "u32(row)") << "\n"
      << a.IndicesSet("a_indices", a.Rank() - 1, "u32(colIn)") << "\n"
      << "        value = " << a.GetByIndices("a_indices") << ";\n"
      << "    }\n"
      << "    return value;\n"
      << "}\n\n";

  // Add the mm_readB function
  shader.AdditionalImplementation()
      << "fn mm_readB(batch: i32, row: i32, colIn: i32, batch_indices: batch_dims_indices_t) -> " << type_string << " {\n"
      << "    var value = " << type_string << "(0.0);\n"
      << "    let col = colIn * " << components << ";\n"
      << "    if(row < i32(uniforms.dim_inner) && col < i32(uniforms.dim_b_outer)) {\n"
      << "        var b_indices: b_indices_t;\n"
      << ConvertOutputBatchIndicesToInputBatchIndices("b", b, b.Rank() - 2, batch_dims.Rank(), "batch_indices")
      << b.IndicesSet("b_indices", b.Rank() - 2, "u32(row)") << "\n"
      << b.IndicesSet("b_indices", b.Rank() - 1, "u32(colIn)") << "\n"
      << "        value = " << b.GetByIndices("b_indices") << ";\n"
      << "    }\n"
      << "    return value;\n"
      << "}\n\n";

  // Add the mm_write function
  shader.AdditionalImplementation()
      << "fn mm_write(batch: i32, row: i32, colIn: i32, valueIn: " << type_string << ") {\n"
      << "  let col = colIn * " << components << ";\n"
      << "  if (row < i32(uniforms.dim_a_outer) && col < i32(uniforms.dim_b_outer)) {\n"
      << "    var value = valueIn;\n"
      << "    let coords = vec3<i32>(batch, row, colIn);\n";

  if (has_bias_) {
    shader.AdditionalImplementation() << "    value = value + " << (is_channels_last_ ? "bias[colIn]" : type_string + "(bias[row])") << ";\n";
  }

  shader.AdditionalImplementation() << "    " << activation_snippet << "\n";

  shader.AdditionalImplementation()
      << output.SetByIndices("vec3<u32>(coords)", "value") << "\n"
      << "  }\n"
      << "}\n\n";
}

Status MatMulProgram::MakeMatMulPackedVec4Source(ShaderHelper& shader,
                                                 const InlinedVector<int64_t>& elements_per_thread,
                                                 uint32_t workgroup_size_x,
                                                 uint32_t workgroup_size_y,
                                                 const std::string& data_type,
                                                 const ShaderIndicesHelper* batch_dims,
                                                 bool transpose_a,
                                                 uint32_t tile_inner,
                                                 bool split_k,
                                                 uint32_t splitted_dim_inner) {
  ORT_UNUSED_PARAMETER(split_k);
  ORT_UNUSED_PARAMETER(splitted_dim_inner);
  std::string write_data_to_sub_a_vec4_snippet =
      transpose_a ? std::string("mm_Asub[inputRow][inputCol] = mm_readA(batch, kStart + inputRow, globalRowStart / innerElementSize + inputCol") + (batch_dims ? ", batchIndices" : "") + ");\n"
                  : std::string("mm_Asub[inputRow][inputCol] = mm_readA(batch, globalRow + innerRow, kStart / innerElementSize + inputCol") + (batch_dims ? ", batchIndices" : "") + ");\n";
  // elements per thread
  const auto elements_per_thread_x = elements_per_thread[0];
  const auto elements_per_thread_y = elements_per_thread[1];

  const auto tile_a_outer = workgroup_size_y * elements_per_thread_y;
  const auto tile_b_outer = workgroup_size_x * elements_per_thread_x;
  const auto tile_a_width = transpose_a ? tile_a_outer : tile_inner;
  const auto tile_a_height = transpose_a ? tile_inner : tile_a_outer;
  const auto inner_elements_size = tile_a_width / workgroup_size_x;
  const auto row_per_thread_b = tile_inner / workgroup_size_y;

  if (!((transpose_a && inner_elements_size == 4 && elements_per_thread[1] == 4) ||
        (!transpose_a && (inner_elements_size == 3 || inner_elements_size == 4))) &&
      tile_a_width % workgroup_size_x == 0 &&
      tile_inner % workgroup_size_y == 0 &&
      elements_per_thread_x == 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid matrix multiplication configuration inner_elements_size: ", inner_elements_size,
                           " must be 3 or 4. tile_a_width: ", tile_a_width, " must be divisible by WorkgroupSizeX: ",
                           workgroup_size_x, ". tile_inner: ", tile_inner, " must be divisible by WorkgroupSizeY: ",
                           workgroup_size_y, ". elements_per_thread_x: ", elements_per_thread_x, " must be 4.");
  }

  shader.AdditionalImplementation()
      << "var<workgroup> mm_Asub: array<array<vec" << inner_elements_size << "<" << data_type << ">, " << tile_a_width / inner_elements_size << ">, " << tile_a_height << ">;\n"
      << "var<workgroup> mm_Bsub: array<array<vec4<" << data_type << ">, " << tile_b_outer / elements_per_thread_x << ">, " << tile_inner << ">;\n"
      << "const rowPerThread = " << elements_per_thread_y << ";\n"
      << "const colPerThread = " << elements_per_thread_x << ";\n"
      << "const innerElementSize = " << inner_elements_size << ";\n"
      << "const tileInner = " << tile_inner << ";\n";

  shader.MainFunctionBody()
      << "  let localRow = i32(local_id.y);\n"
      << "  let tileRow = localRow * rowPerThread;\n"
      << "  let tileCol = i32(local_id.x);\n"
      << "  let globalRow = i32(global_id.y) * rowPerThread;\n"
      << "  let globalCol = i32(global_id.x);\n"
      << "  let batch = i32(global_id.z);\n"
      << (nullptr != batch_dims ? "  let batchIndices = " + batch_dims->OffsetToIndices("u32(batch)") + ";\n" : "")
      << "  let globalRowStart = i32(workgroup_id.y) * " << tile_a_outer << ";\n"
      << "  let num_tiles = (uniforms.dim_inner - 1) / tileInner + 1;\n"
      << "  var kStart = 0;\n"
      << "  var acc: array<vec4<" << data_type << ">, rowPerThread>;\n";

  // Loop over shared dimension.
  shader.MainFunctionBody()
      << "  let tileRowB = localRow * " << row_per_thread_b << ";\n"
      << "  for (var t = 0; t < i32(num_tiles); t = t + 1) {\n";

  // Load one tile of A into local memory.
  shader.MainFunctionBody()
      << "    for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {\n"
      << "      let inputRow = tileRow + innerRow;\n"
      << "      let inputCol = tileCol;\n"
      << "      " << write_data_to_sub_a_vec4_snippet
      << "    }\n";

  // Load one tile of B into local memory.
  shader.MainFunctionBody()
      << "    for (var innerRow = 0; innerRow < " << row_per_thread_b << "; innerRow = innerRow + 1) {\n"
      << "      let inputRow = tileRowB + innerRow;\n"
      << "      let inputCol = tileCol;\n"
      << "      mm_Bsub[inputRow][inputCol] = mm_readB(batch, kStart + inputRow, globalCol" << (nullptr != batch_dims ? ", batchIndices" : "") << ");\n"
      << "    }\n"
      << "    kStart = kStart + tileInner;\n"
      << "    workgroupBarrier();\n";

  // Compute acc values for a single thread.
  shader.MainFunctionBody()
      << "    for (var k = 0; k < tileInner / innerElementSize; k = k + 1) {\n"
      << "      let BCached0 = mm_Bsub[k * innerElementSize][tileCol];\n"
      << "      let BCached1 = mm_Bsub[k * innerElementSize + 1][tileCol];\n"
      << "      let BCached2 = mm_Bsub[k * innerElementSize + 2][tileCol];\n";

  if (inner_elements_size != 3) {
    shader.MainFunctionBody() << "      let BCached3 = mm_Bsub[k * innerElementSize + 3][tileCol];\n";
  }

  if (transpose_a) {
    shader.MainFunctionBody()
        << "      let Acached0 = mm_Asub[k * innerElementSize][localRow];\n"
        << "      let Acached1 = mm_Asub[k * innerElementSize + 1][localRow];\n"
        << "      let Acached2 = mm_Asub[k * innerElementSize + 2][localRow];\n"
        << (inner_elements_size == 3 ? "" : "      let Acached3 = mm_Asub[k * innerElementSize + 3][localRow];\n")
        << "      for (var i = 0; i < rowPerThread; i = i + 1) {\n"
        << "        let ACached = mm_Asub[tileCol][i];\n"
        << "        acc[i] = BCached0 * ACached0[i] + acc[i];\n"
        << "        acc[i] = BCached1 * ACached1[i] + acc[i];\n"
        << "        acc[i] = BCached2 * ACached2[i] + acc[i];\n"
        << "        " << (inner_elements_size == 3 ? "" : "acc[i] = BCached3 * ACached3[i] + acc[i];") << "\n"
        << "      }\n";
  } else {
    shader.MainFunctionBody()
        << "      for (var i = 0; i < rowPerThread; i = i + 1) {\n"
        << "        let ACached = mm_Asub[tileRow + i][k];\n"
        << "        acc[i] = BCached0 * ACached.x + acc[i];\n"
        << "        acc[i] = BCached1 * ACached.y + acc[i];\n"
        << "        acc[i] = BCached2 * ACached.z + acc[i];\n"
        << "        " << (inner_elements_size == 3 ? "" : "acc[i] = BCached3 * ACached.w + acc[i];") << "\n"
        << "      }\n";
  }
  shader.MainFunctionBody()
      << "    }\n"
      << "    workgroupBarrier();\n"
      << "  }\n";  // main for loop

  // Write the results to the output buffer
  shader.MainFunctionBody()
      << "  for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {\n"
      << "    mm_write(batch, globalRow + innerRow, globalCol, acc[innerRow]);\n"
      << "  }\n";

  return Status::OK();
}

Status MatMulProgram::MakeMatMulPackedSource(ShaderHelper& shader,
                                             const InlinedVector<int64_t>& elements_per_thread,
                                             uint32_t workgroup_size_x,
                                             uint32_t workgroup_size_y,
                                             const std::string& data_type,
                                             const ShaderIndicesHelper* batch_dims,
                                             bool transpose_a,
                                             uint32_t tile_inner,
                                             bool split_k,
                                             uint32_t splitted_dim_inner,
                                             bool sequentially_access_by_threads) {
  ORT_UNUSED_PARAMETER(split_k);
  ORT_UNUSED_PARAMETER(splitted_dim_inner);

  const auto elements_per_thread_x = elements_per_thread[0];
  const auto elements_per_thread_y = elements_per_thread[1];

  const auto tile_a_outer = workgroup_size_y * elements_per_thread_y;
  const auto tile_b_outer = workgroup_size_x * elements_per_thread_x;
  const auto tile_a_width = transpose_a ? tile_a_outer : tile_inner;
  const auto tile_a_height = transpose_a ? tile_inner : tile_a_outer;

  if (!(tile_a_height % workgroup_size_y == 0 && tile_a_width % workgroup_size_x == 0 && tile_inner % workgroup_size_y == 0)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "tile_a_height: ", tile_a_height, " must be divisible by WorkgroupSizeY: ", workgroup_size_y,
                           ", tile_a_width: ", tile_a_width, " must be divisible by WorkgroupSizeX: ", workgroup_size_x,
                           ", tile_inner: ", tile_inner, " must be divisible by WorkgroupSizeY: ", workgroup_size_y);
  }

  const auto row_per_thread_a = tile_a_height / workgroup_size_y;
  const auto col_per_thread_a = tile_a_width / workgroup_size_x;
  const auto row_per_thread_b = tile_inner / workgroup_size_y;
  std::string write_data_to_sub_a_snippet = transpose_a ? std::string("mm_Asub[inputRow][inputCol] = mm_readA(batch, kStart + inputRow, globalRowStart + inputCol") + (batch_dims ? ", batchIndices" : "") + ");\n"
                                                        : std::string("mm_Asub[inputRow][inputCol] = mm_readA(batch, globalRowStart + inputRow, kStart + inputCol") + (batch_dims ? ", batchIndices" : "") + ");\n";
  shader.AdditionalImplementation()
      << "var<workgroup> mm_Asub: array<array<" << data_type << ", " << tile_a_width << ">, " << tile_a_height << ">;\n"
      << "var<workgroup> mm_Bsub: array<array<" << data_type << ", " << tile_b_outer << ">, " << tile_inner << ">;\n"
      << "const rowPerThread = " << elements_per_thread_y << ";\n"
      << "const colPerThread = " << elements_per_thread_x << ";\n"
      << "const tileInner = " << tile_inner << ";\n";

  shader.MainFunctionBody() << " let batch = i32(global_id.z);\n"
                            << (nullptr != batch_dims ? "  let batchIndices = " + batch_dims->OffsetToIndices("u32(batch)") + ";\n" : "")
                            << " let num_tiles = (uniforms.dim_inner - 1) / tileInner + 1;\n"
                            << " var kStart = 0;\n"
                            << " var acc: array<array<" << data_type << ", colPerThread>, rowPerThread>;\n";

  if (sequentially_access_by_threads) {
    shader.MainFunctionBody() << "let localRow = i32(local_id.y);\n"
                              << "let localCol = i32(local_id.x);\n"
                              << "let globalRowStart = i32(workgroup_id.y) * " << tile_a_outer << ";\n"
                              << "let globalColStart = i32(workgroup_id.x) * " << tile_b_outer << ";\n"
                              << "\n"
                              << "// Loop over shared dimension.\n"
                              << "for (var t = 0; t < i32(num_tiles); t = t + 1) {\n"
                              << "  // Load one tile of A into local memory.\n"
                              << "  for (var inputRow = localRow; inputRow < " << tile_a_height << "; inputRow = inputRow + " << workgroup_size_y << ") {\n"
                              << "    for (var inputCol = localCol; inputCol < " << tile_a_width << "; inputCol = inputCol + " << workgroup_size_x << ") {\n"
                              << "      " << write_data_to_sub_a_snippet << "\n"
                              << "    }\n"
                              << "  }\n"
                              << "  // Load one tile of B into local memory.\n"
                              << "  for (var inputRow = localRow; inputRow < " << tile_inner << "; inputRow = inputRow + " << workgroup_size_y << ") {\n"
                              << "        for (var inputCol = localCol; inputCol < " << tile_b_outer << "; inputCol = inputCol + " << workgroup_size_x << ") {\n"
                              << "      mm_Bsub[inputRow][inputCol] = mm_readB(batch,\n"
                              << "        kStart + inputRow,\n"
                              << "        globalColStart + inputCol" << (batch_dims ? ", batchIndices" : "") << ");\n "
                              << "    }\n"
                              << "  }\n"
                              << "  kStart = kStart + tileInner;\n"
                              << "  workgroupBarrier();\n"
                              << "\n"
                              << "  // Compute acc values for a single thread.\n"
                              << "  var BCached : array<" << data_type << ", colPerThread>;\n"
                              << "  for (var k = 0; k < tileInner; k = k + 1) {\n"
                              << "    for (var inner = 0; inner < colPerThread; inner = inner + 1) {\n"
                              << "      BCached[inner] = mm_Bsub[k][localCol + inner * " << workgroup_size_x << "];\n"
                              << "    }\n"
                              << "    for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {\n"
                              << "      let ACached = " << (transpose_a ? "mm_Asub[k][localRow + innerRow * " + std::to_string(workgroup_size_y) + "];" : "mm_Asub[localRow + innerRow * " + std::to_string(workgroup_size_y) + "][k];") << "\n"
                              << "      for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {\n"
                              << "        acc[innerRow][innerCol] = acc[innerRow][innerCol] +\n"
                              << "            ACached * BCached[innerCol];\n"
                              << "      }\n"
                              << "    }\n"
                              << "  }\n"
                              << "  workgroupBarrier();\n"
                              << "}\n"
                              << "for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {\n"
                              << "  let gRow = globalRowStart + localRow + innerRow * " << workgroup_size_y << ";\n"
                              << "  for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {\n"
                              << "    let gCol = globalColStart + localCol + innerCol * " << workgroup_size_x << ";\n"
                              << "    mm_write(batch, gRow, gCol, acc[innerRow][innerCol]);\n"
                              << "  }\n"
                              << "}\n";
  } else {
    shader.MainFunctionBody()
        << "let tileRow = i32(local_id.y) * rowPerThread;\n"
        << "let tileCol = i32(local_id.x) * colPerThread;\n"
        << "let globalRow = i32(global_id.y) * rowPerThread;\n"
        << "let globalCol = i32(global_id.x) * colPerThread;\n"
        << "let globalRowStart = i32(workgroup_id.y) * " << tile_a_outer << ";\n"
        << "let tileRowA = i32(local_id.y) * " << row_per_thread_a << ";\n"
        << "let tileColA = i32(local_id.x) * " << col_per_thread_a << ";\n"
        << "let tileRowB = i32(local_id.y) * " << row_per_thread_b << ";\n";

    // Loop over shared dimension.
    shader.MainFunctionBody()
        << "for (var t = 0; t < i32(num_tiles); t = t + 1) {\n";

    // Load one tile of A into local memory.
    shader.MainFunctionBody()
        << "  for (var innerRow = 0; innerRow < i32(" << row_per_thread_a << "); innerRow = innerRow + 1) {\n"
        << "    for (var innerCol = 0; innerCol < i32(" << col_per_thread_a << "); innerCol = innerCol + 1) {\n"
        << "      let inputRow = tileRowA + innerRow;\n"
        << "      let inputCol = tileColA + innerCol;\n"
        << "      " << write_data_to_sub_a_snippet << "\n"
        << "    }\n"
        << "  }\n";

    // Load one tile of B into local memory.
    shader.MainFunctionBody()
        << "  for (var innerRow = 0; innerRow < i32(" << row_per_thread_b << "); innerRow = innerRow + 1) {\n"
        << "    for (var innerCol = 0; innerCol < i32(colPerThread); innerCol = innerCol + 1) {\n"
        << "      let inputRow = tileRowB + innerRow;\n"
        << "      let inputCol = tileCol + innerCol;\n"
        << "      mm_Bsub[inputRow][inputCol] = mm_readB(batch, kStart + inputRow, globalCol + innerCol" << (nullptr != batch_dims ? ", batchIndices" : "") << ");\n"
        << "    }\n"
        << "  }\n"
        << "  kStart = kStart + tileInner;\n"
        << "  workgroupBarrier();\n";

    // Compute acc values for a single thread.
    shader.MainFunctionBody()
        << "var BCached: array<" << data_type << ", colPerThread>;\n"
        << "  for (var k = 0; k < tileInner; k = k + 1) {\n"
        << "    for (var inner = 0; inner < i32(colPerThread); inner = inner + 1) {\n"
        << "      BCached[inner] = mm_Bsub[k][tileCol + inner];\n"
        << "    }\n"
        << "    for (var innerRow = 0; innerRow < i32(rowPerThread); innerRow = innerRow + 1) {\n"
        << "      let ACached = mm_Asub[tileRow + innerRow][k];\n"
        << "      for (var innerCol = 0; innerCol < i32(colPerThread); innerCol = innerCol + 1) {\n"
        << "        acc[innerRow][innerCol] = acc[innerRow][innerCol] + ACached * BCached[innerCol];\n"
        << "      }\n"
        << "    }\n"
        << "  }\n"
        << "  workgroupBarrier();\n"
        << "}\n";

    // Write the results to the output buffer
    shader.MainFunctionBody()
        << "for (var innerRow = 0; innerRow < i32(rowPerThread); innerRow = innerRow + 1) {\n"
        << "  for (var innerCol = 0; innerCol < i32(colPerThread); innerCol = innerCol + 1) {\n"
        << "    mm_write(batch, globalRow + innerRow, globalCol + innerCol, acc[innerRow][innerCol]);\n"
        << "  }\n"
        << "}\n";
  }
  return Status::OK();
}

Status MatMulProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& b = shader.AddInput("b", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& batch_dims = shader.AddIndices("batch_dims", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);

  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  std::string apply_activation = GetActivationSnippet(activation_, "output_value_t", "output_element_t");
  // declare the read and write functions
  MatMulReadWriteFnSource(shader, a, b, output, batch_dims, apply_activation);
  std::string data_type = "a_element_t";
  // generate the main function
  if (is_vec4_) {
    ORT_RETURN_IF_ERROR(MakeMatMulPackedVec4Source(shader, elements_per_thread_, WorkgroupSizeX(), WorkgroupSizeY(), data_type, &batch_dims));
  } else {
    ORT_RETURN_IF_ERROR(MakeMatMulPackedSource(shader, elements_per_thread_, WorkgroupSizeX(), WorkgroupSizeY(), data_type, &batch_dims));
  }
  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
