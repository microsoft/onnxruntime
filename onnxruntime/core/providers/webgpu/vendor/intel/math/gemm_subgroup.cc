// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/vendor/intel/math/gemm_utils_intel.h"
#include "core/providers/webgpu/vendor/intel/math/gemm_subgroup.h"

namespace onnxruntime {
namespace webgpu {

Status MakeMatMulSubgroupVec4Source(ShaderHelper& shader,
                                    const InlinedVector<int64_t>& elements_per_thread,
                                    const std::string& data_type,
                                    const ShaderIndicesHelper* batch_dims,
                                    bool transpose_a,
                                    bool transpose_b,
                                    float alpha,
                                    bool need_handle_matmul,
                                    uint32_t tile_inner) {
  ORT_UNUSED_PARAMETER(transpose_a);
  ORT_UNUSED_PARAMETER(transpose_b);

  const std::string type_string = MakeScalarOrVectorType(4 /*components*/, data_type);

  std::string write_data_to_sub_b_vec4_snippet =
      std::string("mm_Bsub[inputRow][inputCol] = mm_readB(batch, kStart + inputRow, globalColStart + localCol") + (batch_dims ? ", batchIndices" : "") + ");\n";

  // elements per thread
  const auto elements_per_thread_x = elements_per_thread[0];
  const auto elements_per_thread_y = elements_per_thread[1];

  const auto tile_a_outer = kSubgroupLogicalWorkGroupSizeY * elements_per_thread_y;
  const auto tile_b_outer = kSubgroupLogicalWorkGroupSizeX * elements_per_thread_x;

  shader.AdditionalImplementation()
      << "var<workgroup> mm_Bsub: array<array<b_value_t, " << tile_b_outer / elements_per_thread_x << ">, " << tile_inner << ">;\n"
      << "const rowPerThread = " << elements_per_thread_y << ";\n"
      << "const colPerThread = " << elements_per_thread_x << ";\n"
      << "const tileInner = " << tile_inner << ";\n";

  shader.MainFunctionBody()
      << "  let localRow = i32(local_id.x / " << kSubgroupLogicalWorkGroupSizeX << ");\n"
      << "  let localCol = i32(local_id.x % " << kSubgroupLogicalWorkGroupSizeX << ");\n"
      << "  let batch = i32(global_id.z);\n"
      << (nullptr != batch_dims ? "  let batchIndices = " + batch_dims->OffsetToIndices("u32(batch)") + ";\n" : "")
      << "  let globalRowStart = i32(workgroup_id.y) * " << tile_a_outer << ";\n"
      << "  let globalColStart = i32(workgroup_id.x) * " << tile_b_outer / elements_per_thread_x << ";\n"
      << "  let num_tiles = (uniforms.dim_inner - 1) / tileInner + 1;\n"
      << "  var kStart = 0;\n"
      << "  var BCache: b_value_t;\n";

  for (uint32_t i = 0; i < elements_per_thread_y; i++) {
    shader.MainFunctionBody() << "  var acc_" << i << ": output_value_t;\n";
  }

  if (need_handle_matmul) {
    shader.MainFunctionBody() << "  for (var t = 0; t < i32(num_tiles); t = t + 1) {\n";
    // Load one tile of B into local memory.
    const uint32_t loadRowPerThread = kSubgroupLogicalWorkGroupSizeX / kSubgroupLogicalWorkGroupSizeY;
    shader.MainFunctionBody()
        << "    for (var innerRow = 0; innerRow < " << loadRowPerThread << "; innerRow++) {\n"
        << "      let inputRow = " << loadRowPerThread << " * localRow + innerRow;\n"
        << "      let inputCol = localCol;\n"
        << "     " << write_data_to_sub_b_vec4_snippet
        << "    }\n"
        << "    workgroupBarrier();\n";

    for (uint32_t i = 0; i < elements_per_thread_y; i++) {
      shader.MainFunctionBody()
          << "    let a_val_" << i << " = " << std::string("mm_readA(batch, globalRowStart + rowPerThread * localRow + ")
          << i << std::string(", kStart + localCol") + (batch_dims ? ", batchIndices" : "") + ");\n";
    }

    for (uint32_t i = 0; i < kSubgroupLogicalWorkGroupSizeX; i++) {
      shader.MainFunctionBody() << "    BCache = mm_Bsub[" << i << "][localCol];\n";
      for (uint32_t j = 0; j < elements_per_thread_y; j++) {
        shader.MainFunctionBody() << "    acc_" << j << " += subgroupBroadcast(a_val_" << j << ", " << i << ") * BCache;\n";
      }
    }
    shader.MainFunctionBody()
        << "    kStart = kStart + tileInner;\n"
        << "    workgroupBarrier();\n"
        << "  }\n";  // main for loop

    // Calculate alpha * acc
    if (alpha != 1.0f) {
      for (uint32_t i = 0; i < elements_per_thread_y; i++) {
        shader.MainFunctionBody() << "  acc_" << i << " *= output_element_t(uniforms.alpha);\n";
      }
    }
  }

  // Write the results to the output buffer
  for (uint32_t i = 0; i < elements_per_thread_y; i++) {
    shader.MainFunctionBody() << "  mm_write(batch, globalRowStart + rowPerThread * localRow + " << i
                              << ", globalColStart + localCol, acc_" << i << ");\n";
  }
  return Status::OK();
}

Status MakeMatMulSubgroupSource(ShaderHelper& shader,
                                const InlinedVector<int64_t>& elements_per_thread,
                                const std::string& data_type,
                                const ShaderIndicesHelper* batch_dims,
                                bool transpose_a,
                                bool transpose_b,
                                float alpha,
                                bool need_handle_matmul,
                                uint32_t tile_inner) {
  ORT_UNUSED_PARAMETER(transpose_a);
  ORT_UNUSED_PARAMETER(transpose_b);
  ORT_UNUSED_PARAMETER(alpha);

  // elements per thread
  const auto elements_per_thread_x = elements_per_thread[0];
  const auto elements_per_thread_y = elements_per_thread[1];

  const auto tile_a_outer = kSubgroupLogicalWorkGroupSizeY * elements_per_thread_y;
  const auto tile_b_outer = kSubgroupLogicalWorkGroupSizeX * elements_per_thread_x;

  std::string write_data_to_sub_b_snippet = std::string("mm_Bsub[inputRow][inputCol] = mm_readB(batch, kStart + inputRow, globalColStart + inputCol") +
                                            (batch_dims ? ", batchIndices" : "") + ");\n";
  shader.AdditionalImplementation()
      << "var<workgroup> mm_Bsub: array<array<" << data_type << ", " << tile_b_outer << ">, " << tile_inner << ">;\n"
      << "const rowPerThread = " << elements_per_thread_y << ";\n"
      << "const colPerThread = " << elements_per_thread_x << ";\n"
      << "const tileInner = " << tile_inner << ";\n";

  shader.MainFunctionBody() << "  let batch = i32(global_id.z);\n"
                            << (nullptr != batch_dims ? "  let batchIndices = " + batch_dims->OffsetToIndices("u32(batch)") + ";\n" : "")
                            << "  let globalRowStart = i32(workgroup_id.y) * " << tile_a_outer << ";\n"
                            << "  let globalColStart = i32(workgroup_id.x) * " << tile_b_outer << ";\n"
                            << "  let tileRow = i32(local_id.x / " << kSubgroupLogicalWorkGroupSizeX << ") * " << elements_per_thread_y << ";\n"
                            << "  let tileCol = i32(local_id.x % " << kSubgroupLogicalWorkGroupSizeX << ");\n"
                            << "  let num_tiles = (uniforms.dim_inner - 1) / tileInner + 1;\n"
                            << "  var kStart = 0;\n"
                            << "  var BCache: vec4<b_element_t>;\n";

  for (uint32_t i = 0; i < elements_per_thread_y; i++) {
    shader.MainFunctionBody() << "  var acc_" << i << ": vec4<output_element_t>;\n";
  }

  if (need_handle_matmul) {
    // Loop over shared dimension.
    shader.MainFunctionBody()
        << "  for (var t = 0; t < i32(num_tiles); t = t + 1) {\n";

    // Load one tile of B into local memory.
    for (int i = 0; i < elements_per_thread_y; i++) {
      for (int j = 0; j < tile_b_outer; j += kSubgroupLogicalWorkGroupSizeX) {
        shader.MainFunctionBody() << "    " << "mm_Bsub[tileRow + " << i << "][tileCol + " << j
                                  << "] = mm_readB(batch, kStart + tileRow + " << i << ", globalColStart + tileCol + "
                                  << j << (batch_dims ? ", batchIndices" : "") << ");\n";
      }
    }
    shader.MainFunctionBody() << "    workgroupBarrier();\n";

    // Compute acc values for a single thread.
    for (uint32_t i = 0; i < elements_per_thread_y; i++) {
      shader.MainFunctionBody()
          << "    let a_val_" << i << " = " << "mm_readA(batch, globalRowStart + tileRow + " << i
          << ", kStart + tileCol" << (batch_dims ? ", batchIndices" : "") << ");\n";
    }

    for (uint32_t i = 0; i < tile_inner; i++) {
      shader.MainFunctionBody()
          << "    BCache = vec4<b_element_t>(mm_Bsub[" << i << "][tileCol], mm_Bsub["
          << i << "][tileCol + " << kSubgroupLogicalWorkGroupSizeX << "], mm_Bsub["
          << i << "][tileCol + 2 * " << kSubgroupLogicalWorkGroupSizeX << "], mm_Bsub["
          << i << "][tileCol + 3 * " << kSubgroupLogicalWorkGroupSizeX << "]);\n";
      for (uint32_t j = 0; j < elements_per_thread_y; j++) {
        shader.MainFunctionBody() << "    acc_" << j << " += subgroupBroadcast(a_val_" << j << ", " << i << ") * BCache;\n";
      }
    }

    shader.MainFunctionBody() << "    kStart = kStart + tileInner;\n"
                              << "    workgroupBarrier();\n"
                              << "  }\n";
  }
  // Calculate alpha * acc
  if (alpha != 1.0f) {
    for (uint32_t i = 0; i < elements_per_thread_y; i++) {
      shader.MainFunctionBody() << "  acc_" << i << " *= output_element_t(uniforms.alpha);\n";
    }
  }

  // Write the results to the output buffer
  for (uint32_t i = 0; i < elements_per_thread_y; i++) {
    for (uint32_t j = 0; j < elements_per_thread_x; j++) {
      shader.MainFunctionBody() << "  "
                                << "mm_write(batch, globalRowStart + tileRow + " << i << ", globalColStart + tileCol + "
                                << j * kSubgroupLogicalWorkGroupSizeX << ", acc_" << i << "[" << j << "]);\n";
    }
  }

  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
