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

// Load one tile of B into local-memory buffer `buf`, reading from K offset `k_start`.
std::string LoadBStr(const ShaderIndicesHelper* batch_dims, int64_t tile_b_outer, bool is_vec4,
                     const std::string& buf, const std::string& k_start) {
  SS(load_b_ss, 256);
  load_b_ss << "      for (var innerRow = 0; innerRow < loadRowsPerThread; innerRow++) {\n"
            << "        let inputRow = loadRowsPerThread * localRow + innerRow;\n"
            << "        let inputCol = tileCol;\n";
  if (is_vec4) {
    load_b_ss << "        mm_Bsub[" << buf << "][inputRow][inputCol] = mm_readB(batch, " << k_start << " + inputRow, globalColStart"
              << (batch_dims ? ", batchIndices" : "") << ");\n";
  } else {
    for (int j = 0; j < tile_b_outer; j += kSubgroupLogicalWorkGroupSizeX) {
      load_b_ss << "        mm_Bsub[" << buf << "][inputRow][inputCol + " << j << "] = mm_readB(batch, " << k_start << " + inputRow, globalColStart + "
                << j << (batch_dims ? ", batchIndices" : "") << ");\n";
    }
  }
  load_b_ss << "      }\n";

  return SS_GET(load_b_ss);
}

std::string LoadBCacheStr(bool is_vec4, uint32_t offset, const std::string& buf) {
  SS(b_cache_ss, 256);
  if (is_vec4) {
    b_cache_ss << "BCache = mm_Bsub[" << buf << "][" << offset << "][tileCol];\n";
  } else {
    b_cache_ss << "BCache = vec4<b_element_t>(mm_Bsub[" << buf << "][" << offset << "][tileCol], "
               << "mm_Bsub[" << buf << "][" << offset << "][tileCol + " << kSubgroupLogicalWorkGroupSizeX << "], "
               << "mm_Bsub[" << buf << "][" << offset << "][tileCol + " << 2 * kSubgroupLogicalWorkGroupSizeX << "], "
               << "mm_Bsub[" << buf << "][" << offset << "][tileCol + " << 3 * kSubgroupLogicalWorkGroupSizeX << "]);\n";
  }
  return SS_GET(b_cache_ss);
}

std::string CalculateAccStr(const ShaderIndicesHelper* batch_dims, int64_t elements_per_thread_y, bool is_vec4, bool a_vec4, const std::string& buf) {
  SS(cal_acc_ss, 1024);

  if (a_vec4) {
    // Load A from global memory as vec4 (one vec4 = 4 consecutive K elements) using cooperative
    // loading across the subgroup. The 32-wide K tile is 8 vec4 columns. Lanes are split into
    // groups of 8: each group covers the 8 vec4 columns (lane i in a group loads vec4 column i),
    // and the eptY rows of the tile are distributed across the S/8 groups, so each lane only
    // loads (eptY / (S/8)) rows from global memory with no redundancy.
    //
    // Example (S=32, eptY=8): 4 groups of 8 lanes, each group loads 2 rows -> 4*2 = 8 rows,
    // i.e. the full 8x32 (8 vec4 per row) tile, each lane reads 2 vec4. The per-lane vec4 are
    // then exchanged with subgroupBroadcast; for vec4 column 0 the broadcast source lanes are
    // 0, 8, 16, 24 (the leading lane of each group).
    const std::map<uint32_t, uint32_t> sg_groups = {{32, 4}, {16, 2}, {8, 1}};
    for (const auto& [simd, num_groups] : sg_groups) {
      // The vec4 cooperative-load path distributes eptY rows evenly across `num_groups`
      // lane groups (up to 4), so it requires eptY to be a positive multiple of num_groups.
      // This holds today because CanApplySubgroup gates the kernel to M >= 64, which makes
      // ElementsPerThreadY return 4. Enforce it explicitly so that relaxing the M >= 64 guard
      // or retuning ElementsPerThreadY fails loudly here instead of silently dividing by zero
      // in the `g = r / rows_per_group` / `j = r % rows_per_group` computations below.
      ORT_ENFORCE(elements_per_thread_y > 0 && elements_per_thread_y % num_groups == 0,
                  "a_vec4 cooperative load requires elements_per_thread_y (", elements_per_thread_y,
                  ") to be a positive multiple of num_groups (", num_groups, ").");
      const uint32_t rows_per_group = static_cast<uint32_t>(elements_per_thread_y) / num_groups;
      cal_acc_ss << "    if (sg_size == " << simd << ") {\n"
                 << "      let aSgLane = tileCol % " << simd << ";\n"
                 << "      let aGroup = aSgLane / 8;\n"
                 << "      let aKvec = aSgLane % 8;\n"
                 << "      let aColV = kStart / 4 + aKvec;\n";
      // Cooperative load: this lane loads `rows_per_group` rows at vec4 column `aColV`.
      for (uint32_t j = 0; j < rows_per_group; j++) {
        cal_acc_ss << "      a_val_" << j << " = mm_readA(batch, globalRowStart + aGroup * "
                   << rows_per_group << " + " << j << ", aColV"
                   << (batch_dims ? ", batchIndices" : "") << ");\n";
      }
      // Accumulate over the 8 vec4 columns (32 K) of the tile. Each kvec block is wrapped in
      // braces so the broadcast temporaries get a fresh scope (avoids redeclaration).
      for (uint32_t kvec = 0; kvec < 8; kvec++) {
        // Fresh scope per kvec: the `aB_*` broadcast temporaries below are redeclared each
        // iteration, which WGSL only allows in a new block scope.
        cal_acc_ss << "      {\n";
        for (uint32_t r = 0; r < elements_per_thread_y; r++) {
          const uint32_t g = r / rows_per_group;
          const uint32_t j = r % rows_per_group;
          const uint32_t src_lane = g * 8 + kvec;
          cal_acc_ss << "        let aB_" << r << " = subgroupBroadcast(a_val_" << j << ", " << src_lane << ");\n";
        }
        for (uint32_t c = 0; c < 4; c++) {
          const uint32_t k = kvec * 4 + c;
          cal_acc_ss << "        " << LoadBCacheStr(is_vec4, k, buf);
          for (uint32_t r = 0; r < elements_per_thread_y; r++) {
            cal_acc_ss << "        acc_" << r << " += aB_" << r << "[" << c << "] * BCache;\n";
          }
        }
        cal_acc_ss << "      }\n";
      }
      cal_acc_ss << "    }\n";
    }
    return SS_GET(cal_acc_ss);
  }

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
        cal_acc_ss << "      " << LoadBCacheStr(is_vec4, sg_idx + offset, buf);
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

int64_t ElementsPerThreadY(ComputeContext& context, uint32_t M) {
  // For Xe-LPG and Xe-3LPG, we have observed that 4 elements per thread is optimal when M is large.
  const auto& arch = context.AdapterInfo().architecture;
  const bool is_xe_lpg_or_xe_3lpg = arch == gpu_arch::kXeLpg ||
                                    arch == gpu_arch::kXe3Lpg;
  return M <= 8 ? 1 : (M <= 16 ? 2 : (M <= 32 ? 4 : (is_xe_lpg_or_xe_3lpg ? 4 : 8)));
}

Status MakeMatMulSubgroupSource(ShaderHelper& shader,
                                const InlinedVector<int64_t>& elements_per_thread,
                                const ShaderIndicesHelper* batch_dims,
                                bool is_vec4,
                                bool a_vec4,
                                bool b_is_fp16,
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

  // Double-buffering of the B tile in workgroup memory is only enabled for float16 B inputs.
  // The workgroup buffer holds the B tile, so its footprint scales with B's element size. For
  // float32 B, a second buffer would double the (already larger) workgroup memory footprint and
  // risk exceeding device limits, so a single buffer is used instead.
  const uint32_t num_b_buffers = b_is_fp16 ? 2 : 1;

  shader.AdditionalImplementation()
      << "var<workgroup> mm_Bsub: array<array<array<b_value_t, " << (is_vec4 ? tile_b_outer / elements_per_thread_x : tile_b_outer) << ">, 32>, " << num_b_buffers << ">;\n";

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
      << (a_vec4 ? "" : "  var aCol = 0;\n")
      << "  var BCache: vec4<b_element_t>;\n";

  for (uint32_t i = 0; i < elements_per_thread_y; i++) {
    shader.MainFunctionBody() << "  var acc_" << i << " = vec4<output_element_t>(0);\n"
                              << "  var a_val_" << i << " = a_value_t(0);\n";
  }

  if (need_handle_matmul) {
    shader.MainFunctionBody()
        << "  let loadRowsPerThread = " << kSubgroupLogicalWorkGroupSizeX / kSubgroupLogicalWorkGroupSizeY << ";\n";
    if (b_is_fp16) {
      // Double-buffered K loop: while computing on the current B tile, prefetch the next tile
      // into the other workgroup buffer. This overlaps global-memory load latency with compute
      // and needs only a single workgroupBarrier per iteration (vs. two for single buffering).
      shader.MainFunctionBody()
          << "  {\n"  // prologue: prefetch tile 0 into buffer 0
          << LoadBStr(batch_dims, tile_b_outer, is_vec4, "0", "0")
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  for (var t = 0; t < i32(numTiles); t++) {\n"
          << "    let curr = t % 2;\n"
          << (a_vec4 ? "" : "    aCol = kStart + tileCol % i32(sg_size);\n")
          << CalculateAccStr(batch_dims, elements_per_thread_y, is_vec4, a_vec4, "curr")
          << "    if (t + 1 < i32(numTiles)) {\n"
          << LoadBStr(batch_dims, tile_b_outer, is_vec4, "(t + 1) % 2", "kStart + 32")
          << "    }\n"
          << "    workgroupBarrier();\n"
          << "    kStart = kStart + 32;\n"
          << "  }\n";  // main for loop
    } else {
      // Single-buffered K loop: load the current B tile, then compute on it. Two barriers per
      // iteration are required (after the load and after the compute) to avoid overwriting the
      // shared buffer while it is still being read.
      shader.MainFunctionBody()
          << "  for (var t = 0; t < i32(numTiles); t++) {\n"
          << "    {\n"
          << LoadBStr(batch_dims, tile_b_outer, is_vec4, "0", "kStart")
          << "    }\n"
          << "    workgroupBarrier();\n"
          << (a_vec4 ? "" : "    aCol = kStart + tileCol % i32(sg_size);\n")
          << CalculateAccStr(batch_dims, elements_per_thread_y, is_vec4, a_vec4, "0")
          << "    kStart = kStart + 32;\n"
          << "    workgroupBarrier();\n"
          << "  }\n";  // main for loop
    }

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
