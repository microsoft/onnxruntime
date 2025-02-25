// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/matmul_packed.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {


// std::string MakeTypeString(int components, std::string_view data_type) {
//   switch (components) {
//     case 1:
//       return std::string{data_type};
//     case 2:
//       return MakeStringWithClassicLocale("vec2<", data_type, ">");
//     case 3:
//       return MakeStringWithClassicLocale("vec3<", data_type, ">");
//     case 4:
//       return MakeStringWithClassicLocale("vec4<", data_type, ">");
//     default:
//       ORT_THROW("Unsupported number of components: ", components);
//   }
// }

// // Helper that convert output batch indices to input batch indices using only the rank and
// // the shape information in uniform
// std::string ConvertOutputBatchIndicesToInputBatchIndices(const std::string& name, const ShaderVariableHelper& input, int input_batch_rank, int output_batch_rank, const std::string& batch_indices) {
//     std::ostringstream oss;
//     const std::string input_shape = "uniforms." + name + "_shape";
//     const std::string input_indices = name + "_indices";
//     int extending_input_rank = output_batch_rank - input_batch_rank;
//     for (int i = 0; i < input_batch_rank; ++i) {
//         oss << "if (" << GetElementAt(input_shape, i, input.Rank()) << " != 1) {\n"
//             << input.IndicesSet(input_indices, i, GetElementAt(batch_indices, i + extending_input_rank, output_batch_rank)) << "\n"
//             << "} else {\n"
//             << input.IndicesSet(input_indices, i, 0) << "\n"
//             << "}\n";
//     }
//     return oss.str();
// }

void MatMulProgram::MatMulReadWriteFnSource(ShaderHelper& shader, const ShaderVariableHelper& a, const ShaderVariableHelper& b, const ShaderVariableHelper& output, const ShaderIndicesHelper& batch_dims) const {
    int components = is_vec4_ ? 4 : 1;
    const std::string data_type = "a_value_t";
    const std::string type_string =  MakeTypeString(components, data_type);

    // Add the mm_readA function
    shader.AdditionalImplementation()
        << "fn mm_readA(batch: i32, row: i32, colIn: i32, batch_indices: batch_dims_indices_t) -> " << type_string << " {\n"
        << "    var value = " << type_string << "(0.0);\n"
        << "    let col = colIn * " << components << ";\n"
        << "    if(row < uniforms.dim_a_outer && col < uniforms.dim_inner) {\n"
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
        << "    if(row < uniforms.dim_inner && col < uniforms.dim_b_outer) {\n"
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
        << "  if (row < uniforms.dim_a_outer && col < uniforms.dim_b_outer) {\n"
        << "    var value = valueIn;\n"
        << "    let coords = vec3<i32>(batch, row, colIn);\n";

    if (has_bias_) {
        shader.AdditionalImplementation() << "    value = value + " << type_string << "(bias[row]);\n";
    }

    shader.AdditionalImplementation()
        << output.SetByIndices("vec3<u32>(coords)", "value") << "\n"
        << "  }\n"
        << "}\n\n";
}


Status MatMulProgram::MakeMatMUlPackedVec4Source(ShaderHelper& shader, const ShaderIndicesHelper& batch_dims) const {
        // elements per thread
    const int elements_per_thread_x = elements_per_thread_[0];
    const int elements_per_thread_y = elements_per_thread_[1];
    //const int elements_per_thread_z = 1;
    const int tile_inner = 32;
   // const int splitted_dim_inner = 32;

    const int tile_a_outer =  WorkgroupSizeY() * elements_per_thread_y;
    const int tile_b_outer = WorkgroupSizeX() * elements_per_thread_x;
    const int tile_a_width = tile_inner;
    const int tile_a_height = tile_a_outer;
    const int inner_elements_size = tile_a_width / WorkgroupSizeX();
    const int row_per_thread_b = tile_inner / WorkgroupSizeY();

    const std::string data_type = "a_value_t";

    if (!((inner_elements_size == 3 || inner_elements_size == 4) &&
      tile_a_width % WorkgroupSizeX() == 0 &&
      tile_inner % WorkgroupSizeY() == 0 &&
      elements_per_thread_x == 4)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
            "Invalid matrix multiplication configuration inner_elements_size: ", inner_elements_size,
            " must be 3 or 4. tile_a_width: ", tile_a_width, " must be divisible by WorkgroupSizeX: ", WorkgroupSizeX(),
            ". tile_inner: ", tile_inner, " must be divisible by WorkgroupSizeY: ", WorkgroupSizeY(),
            ". elements_per_thread_x: ", elements_per_thread_x, " must be 4.");
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
        << "  let batchIndices = " << batch_dims.OffsetToIndices("u32(batch)") << ";\n"
        << "  let globalRowStart = i32(workgroup_id.y) * " << tile_a_outer << ";\n"
        << "  let num_tiles = (uniforms.dim_inner - 1) / tileInner + 1;\n"
        << "  var kStart = 0;\n"
        << "  var acc: array<vec4<" << data_type << ">, rowPerThread>;\n";

    // Loop over shared dimension.
    shader.MainFunctionBody()
        << "  let tileRowB = localRow * " << row_per_thread_b << ";\n"
        << "  for (var t = 0; t < num_tiles; t = t + 1) {\n";

    // Load one tile of A into local memory.
    shader.MainFunctionBody()
        << "    for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {\n"
        << "      let inputRow = tileRow + innerRow;\n"
        << "      let inputCol = tileCol;\n"
        << "      mm_Asub[inputRow][inputCol] = mm_readA(batch, globalRow + innerRow, kStart / innerElementSize + inputCol, batchIndices);\n"
        << "    }\n";

    // Load one tile of B into local memory.
    shader.MainFunctionBody()
        << "    for (var innerRow = 0; innerRow < " << row_per_thread_b << "; innerRow = innerRow + 1) {\n"
        << "      let inputRow = tileRowB + innerRow;\n"
        << "      let inputCol = tileCol;\n"
        << "      mm_Bsub[inputRow][inputCol] = mm_readB(batch, kStart + inputRow, globalCol, batchIndices);\n"
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

    shader.MainFunctionBody()
        << "      for (var i = 0; i < rowPerThread; i = i + 1) {\n"
        << "        let ACached = mm_Asub[tileRow + i][k];\n"
        << "        acc[i] = BCached0 * ACached.x + acc[i];\n"
        << "        acc[i] = BCached1 * ACached.y + acc[i];\n"
        << "        acc[i] = BCached2 * ACached.z + acc[i];\n"
        << "        " << (inner_elements_size == 3 ? "" : "acc[i] = BCached3 * ACached.w + acc[i];") << "\n"
        << "      }\n";

    shader.MainFunctionBody() << "    workgroupBarrier();\n"
                              << "  }\n"; // main for loop

    // Write the results to the output buffer
    shader.MainFunctionBody()
        << "  for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {\n"
        << "    mm_write(batch, globalRow + innerRow, globalCol, acc[innerRow]);\n"
        << "  }\n";

    return Status::OK();
}

Status MatMulProgram::MakeMatMulPackedSource(ShaderHelper& shader, const ShaderIndicesHelper& batch_dims) const {
    const int elements_per_thread_x = elements_per_thread_[0];
    const int elements_per_thread_y = elements_per_thread_[1];
    const int tile_inner = 32;
   // const int splitted_dim_inner = 32;

    const int tile_a_outer =  WorkgroupSizeY() * elements_per_thread_y;
    const int tile_b_outer = WorkgroupSizeX() * elements_per_thread_x;
    const int tile_a_width = tile_inner;
    const int tile_a_height = tile_a_outer;


    if(!(tile_a_height % WorkgroupSizeY() == 0 && tile_a_width % WorkgroupSizeX() == 0 && tile_inner % WorkgroupSizeY() == 0)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
            "tile_a_height: ", tile_a_height, " must be divisible by WorkgroupSizeY: ", WorkgroupSizeY(),
            ", tile_a_width: ", tile_a_width, " must be divisible by WorkgroupSizeX: ", WorkgroupSizeX(),
            ", tile_inner: ", tile_inner, " must be divisible by WorkgroupSizeY: ", WorkgroupSizeY());
    }

    const std::string data_type = "a_value_t";

    const int row_per_thread_a = tile_a_height / WorkgroupSizeY();
    const int col_per_thread_a = tile_a_width / WorkgroupSizeX();
    const int row_per_thread_b = tile_inner / WorkgroupSizeY();

    shader.AdditionalImplementation()
        << "var<workgroup> mm_Asub: array<array<" << data_type << ", " << tile_a_width << ">, " << tile_a_height << ">;\n"
        << "var<workgroup> mm_Bsub: array<array<" << data_type << ", " << tile_b_outer << ">, " << tile_inner << ">;\n"
        << "const rowPerThread = " << elements_per_thread_y << ";\n"
        << "const colPerThread = " << elements_per_thread_x << ";\n"
        << "const tileInner = " << tile_inner << ";\n";

    shader.MainFunctionBody() << " let batch = i32(global_id.z);\n"
                              << " let batchIndices = " << batch_dims.OffsetToIndices("u32(batch)") << ";\n"
                              << " let num_tiles = (uniforms.dim_inner - 1) / tileInner + 1;\n"
                              << " var kStart = 0;\n"
                              << " var acc: array<vec4<" << data_type << ">, rowPerThread>;\n";

    shader.MainFunctionBody()
        <<  "let tileRow = i32(local_id.y) * rowPerThread;\n"
        <<  "let tileCol = i32(local_id.x) * colPerThread;\n"
        <<  "let globalRow = i32(global_id.y) * rowPerThread;\n"
        <<  "let globalCol = i32(global_id.x) * colPerThread;\n"
        <<  "let globalRowStart = i32(workgroup_id.y) * " << tile_a_outer << ";\n"
        <<  "let tileRowA = i32(local_id.y) * " << row_per_thread_a << ";\n"
        <<  "let tileColA = i32(local_id.x) * " << col_per_thread_a << ";\n"
        <<  "let tileRowB = i32(local_id.y) * " << row_per_thread_b << ";\n";

    // Loop over shared dimension.
    shader.MainFunctionBody()
        << "for (var t = 0; t < num_tiles; t = t + 1) {\n";

    // Load one tile of A into local memory.
    shader.MainFunctionBody()
        << "  for (var innerRow = 0; innerRow < " << row_per_thread_a << "; innerRow = innerRow + 1) {\n"
        << "    for (var innerCol = 0; innerCol < " << col_per_thread_a << "; innerCol = innerCol + 1) {\n"
        << "      let inputRow = tileRowA + innerRow;\n"
        << "      let inputCol = tileColA + innerCol;\n"
        << "      mm_Asub[inputRow][inputCol] = mm_readA(batch, globalRowStart + innerRow, kStart + inputCol, batchIndices);\n"
        << "    }\n"
        << "  }\n";

    // Load one tile of B into local memory.
    shader.MainFunctionBody()
        << "  for (var innerRow = 0; innerRow < " << row_per_thread_b << "; innerRow = innerRow + 1) {\n"
        << "    for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {\n"
        << "      let inputRow = tileRowB + innerRow;\n"
        << "      let inputCol = tileCol + innerCol;\n"
        << "      mm_Bsub[inputRow][inputCol] = mm_readB(batch, kStart + inputRow, globalCol + innerCol, batchIndices);\n"
        << "    }\n"
        << "  }\n"
        << "  kStart = kStart + tileInner;\n"
        << "  workgroupBarrier();\n";

    // Compute acc values for a single thread.
    shader.MainFunctionBody()
        << "var BCached: array<" << data_type << ", colPerThread>;\n"
        << "  for (var k = 0; k < tileInner; k = k + 1) {\n"
        << "    for (var inner = 0; inner < colPerThread; inner = inner + 1) {\n"
        << "      BCached[inner] = mm_Bsub[k][tileCol + inner];\n"
        << "    }\n"
        << "    for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {\n"
        << "      let ACached = mm_Asub[tileRow + innerRow][k];\n"
        << "      for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {\n"
        << "        acc[innerRow][innerCol] = acc[innerRow][innerCol] + ACached * BCached[innerCol];\n"
        << "      }\n"
        << "    }\n"
        << "  }\n"
        << "  workgroupBarrier();\n"
        << "}\n";

    // Write the results to the output buffer
    shader.MainFunctionBody()
        << "for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {\n"
        << "  for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {\n"
        << "    mm_write(batch, globalRow + innerRow, globalCol + innerCol, acc[innerRow][innerCol]);\n"
        << "  }\n"
        << "}\n";

    return Status::OK();

}



Status MatMulProgram::GenerateShaderCode(ShaderHelper& shader) const {
    const auto& a = shader.AddInput("a",ShaderUsage::UseIndicesTypeAlias| ShaderUsage::UseValueTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
    const auto& output = shader.AddOutput("output", ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
    const auto& batch_dims = shader.AddIndices("batch_dims");

    if (has_bias_) {
        shader.AddInput("bias", ShaderUsage::UseUniform);
    }


    // declare the read and write functions
    MatMulReadWriteFnSource(shader, a, b, output, batch_dims);

    // generate the main function
    if (is_vec4_) {
        ORT_RETURN_IF_ERROR(MakeMatMUlPackedVec4Source(shader, batch_dims));
    } else {
        ORT_RETURN_IF_ERROR(MakeMatMulPackedSource(shader, batch_dims));
    }
    return Status::OK();
}

}  // namespace webgpu
} // namespace onnxruntime
