// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/string_macros.h"
#include "core/providers/webgpu/vendor/intel/math/gemm_subgroup.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

namespace {

std::string LoadAStr(const ShaderIndicesHelper* batch_dims, int64_t elements_per_thread_y) {
  SS(load_a_ss, 128);
  for (int64_t i = 0; i < elements_per_thread_y; i++) {
    load_a_ss << "      a_val_" << i << " = " << std::string("mm_readA(batch, globalRowStart + ")
              << i << std::string(", aCol") + (batch_dims ? ", batchIndices" : "") + ");\n";
  }
  return SS_GET(load_a_ss);
}

// Load one tile of B into local memory.
std::string LoadBStr(const ShaderIndicesHelper* batch_dims, int64_t tile_b_outer, bool is_vec4) {
  SS(load_b_ss, 256);
  load_b_ss << "    let loadRowsPerThread = " << kSubgroupLogicalWorkGroupSizeX / kSubgroupLogicalWorkGroupSizeY << ";\n"
            << "    for (var innerRow = 0; innerRow < loadRowsPerThread; innerRow++) {\n"
            << "      let inputRow = loadRowsPerThread * localRow + innerRow;\n"
            << "      let inputCol = tileCol;\n";
  if (is_vec4) {
    load_b_ss << "      mm_Bsub[inputRow][inputCol] = mm_readB(batch, kStart + inputRow, globalColStart"
              << (batch_dims ? ", batchIndices" : "") << ");\n";
  } else {
    for (int j = 0; j < tile_b_outer; j += kSubgroupLogicalWorkGroupSizeX) {
      load_b_ss << "      mm_Bsub[inputRow][inputCol + " << j << "] = mm_readB(batch, kStart + inputRow, globalColStart + "
                << j << (batch_dims ? ", batchIndices" : "") << ");\n";
    }
  }
  load_b_ss << "    }\n"
            << "    workgroupBarrier();\n";

  return SS_GET(load_b_ss);
}

std::string LoadBCacheStr(bool is_vec4, uint32_t offset) {
  SS(b_cache_ss, 256);
  if (is_vec4) {
    b_cache_ss << "BCache = mm_Bsub[" << offset << "][tileCol];\n";
  } else {
    b_cache_ss << "BCache = vec4<b_element_t>(mm_Bsub[" << offset << "][tileCol], "
               << "mm_Bsub[" << offset << "][tileCol + " << kSubgroupLogicalWorkGroupSizeX << "], "
               << "mm_Bsub[" << offset << "][tileCol + " << 2 * kSubgroupLogicalWorkGroupSizeX << "], "
               << "mm_Bsub[" << offset << "][tileCol + " << 3 * kSubgroupLogicalWorkGroupSizeX << "]);\n";
  }
  return SS_GET(b_cache_ss);
}

std::string CalculateAccStr(const ShaderIndicesHelper* batch_dims, int64_t elements_per_thread_y, bool is_vec4) {
  SS(cal_acc_ss, 1024);

  // key: simd size; value: the offset row of mm_Bsub.
  std::map<uint32_t, std::vector<uint32_t>> simd_map = {
      {32, {0}},
      {16, {0, 16}},
      {8, {0, 8, 16, 24}}};
  for (const auto& [simd, offsets] : simd_map) {
    cal_acc_ss << "    if (sg_size == " << simd << ") {\n";
    for (uint32_t offset : offsets) {
      cal_acc_ss << LoadAStr(batch_dims, elements_per_thread_y)
                 << "      aCol += " << simd << ";\n";
      for (uint32_t sg_idx = 0; sg_idx < simd; sg_idx++) {
        cal_acc_ss << "      " << LoadBCacheStr(is_vec4, sg_idx + offset);
        for (uint32_t i = 0; i < elements_per_thread_y; i++) {
          cal_acc_ss << "      acc_" << i << " += subgroupBroadcast(a_val_" << i << ", " << sg_idx << ") * BCache;\n";
        }
      }
    }
    cal_acc_ss << "    }\n";
  }

  return SS_GET(cal_acc_ss);
}

}  // namespace

bool CanApplySubgroup(const ComputeContext& context, int64_t M, int64_t N, int64_t K, bool transA, bool transB) {
  if (context.AdapterInfo().vendor == std::string_view{"intel"}) {
    bool use_subgroup = context.HasFeature(wgpu::FeatureName::Subgroups) &&
                        M >= 64 && N >= 512 && K >= 32 && !transA && !transB;
    return use_subgroup;
  }

  return false;
}

int64_t ElementsPerThreadY(bool is_vec4, uint32_t M) {
  return is_vec4 ? (M <= 8 ? 1 : (M <= 16 ? 2 : (M <= 32 ? 4 : 8))) : 4;
}

Status MakeMatMulSubgroupSource(ShaderHelper& shader,
                                const InlinedVector<int64_t>& elements_per_thread,
                                const ShaderIndicesHelper* batch_dims,
                                bool is_vec4,
                                bool transpose_a,
                                bool transpose_b,
                                float alpha,
                                bool need_handle_matmul) {
  ORT_UNUSED_PARAMETER(transpose_a);
  ORT_UNUSED_PARAMETER(transpose_b);

  // elements per thread
  const auto elements_per_thread_x = elements_per_thread[0];
  const auto elements_per_thread_y = elements_per_thread[1];

  const auto tile_a_outer = kSubgroupLogicalWorkGroupSizeY * elements_per_thread_y;
  const auto tile_b_outer = kSubgroupLogicalWorkGroupSizeX * elements_per_thread_x;

  shader.AdditionalImplementation()
      << "var<workgroup> mm_Bsub: array<array<b_value_t, " << (is_vec4 ? tile_b_outer / elements_per_thread_x : tile_b_outer) << ">, 32>;\n";

  shader.MainFunctionBody()
      << "  let workgroupIdXStride = (uniforms.dim_b_outer - 1) / " << tile_b_outer << " + 1;\n"
      << "  let workgroupIdYStride = (uniforms.dim_a_outer - 1) / " << tile_a_outer << " + 1;\n"
      << "  let batch = i32(workgroup_idx / (workgroupIdXStride * workgroupIdYStride));\n"
      << "  let workgroupIdXY = workgroup_idx % (workgroupIdXStride * workgroupIdYStride);\n"
      << "  let workgroupIdX = workgroupIdXY % workgroupIdXStride;\n"
      << "  let workgroupIdY = workgroupIdXY / workgroupIdXStride;\n"
      << "  let tileRow = i32(local_id.x / " << kSubgroupLogicalWorkGroupSizeX << ") * " << elements_per_thread_y << ";\n"
      << "  let tileCol = i32(local_id.x % " << kSubgroupLogicalWorkGroupSizeX << ");\n"
      << "  let localRow = i32(local_id.x / " << kSubgroupLogicalWorkGroupSizeX << ");\n"
      << (nullptr != batch_dims ? "  let batchIndices = " + batch_dims->OffsetToIndices("u32(batch)") + ";\n" : "")
      << "  let globalRowStart = i32(workgroupIdY) * " << tile_a_outer << " + tileRow;\n"
      << "  let globalColStart = i32(workgroupIdX) * " << (is_vec4 ? tile_b_outer / elements_per_thread_x : tile_b_outer) << " + tileCol;\n"
      << "  let numTiles = (uniforms.dim_inner - 1) / 32 + 1;\n"
      << "  var kStart = 0;\n"
      << "  var aCol = 0;\n"
      << "  var BCache: vec4<b_element_t>;\n";

  for (uint32_t i = 0; i < elements_per_thread_y; i++) {
    shader.MainFunctionBody() << "  var acc_" << i << " = vec4<output_element_t>(0);\n"
                              << "  var a_val_" << i << " = a_value_t(0);\n";
  }

  if (need_handle_matmul) {
    shader.MainFunctionBody() << "  for (var t = 0; t < i32(numTiles); t++) {\n"
                              << LoadBStr(batch_dims, tile_b_outer, is_vec4)
                              << "    aCol = kStart + tileCol % i32(sg_size);\n"
                              << CalculateAccStr(batch_dims, elements_per_thread_y, is_vec4)
                              << "    kStart = kStart + 32;\n"
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
  if (is_vec4) {
    for (uint32_t i = 0; i < elements_per_thread_y; i++) {
      shader.MainFunctionBody() << "  mm_write(batch, globalRowStart + " << i
                                << ", globalColStart, acc_" << i << ");\n";
    }
  } else {
    for (uint32_t i = 0; i < elements_per_thread_y; i++) {
      for (uint32_t j = 0; j < elements_per_thread_x; j++) {
        shader.MainFunctionBody() << "  "
                                  << "mm_write(batch, globalRowStart + " << i << ", globalColStart + "
                                  << j * kSubgroupLogicalWorkGroupSizeX << ", acc_" << i << "[" << j << "]);\n";
      }
    }
  }

  return Status::OK();
}

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
