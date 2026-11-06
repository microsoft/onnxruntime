// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/math/gemm_utils.h"
#include "core/providers/webgpu/vendor/intel/math/gemm_subgroup16.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

namespace {

// Each K tile is as deep as one subgroup, so every lane loads exactly one K element (or vec4) of A per row.
constexpr uint32_t kTileK = kMatMulSubgroup16Size;
constexpr uint32_t kBRowsPerInvocation = kTileK / kSubgroup16WorkGroupSizeY;

// Smallest workgroup count that keeps the GPU busy, measured on an Arc A770 (32 Xe-cores). Below it, fewer rows
// per invocation (and so more workgroups) win over the extra B reuse of taller tiles.
constexpr uint64_t kMinWorkgroups = 320;

std::string BSub(const std::string& buffer, const std::string& row, const std::string& col) {
  return "mm_Bsub[" + buffer + "][" + row + "][" + col + "]";
}

std::string BRow(uint32_t r) {
  return "localRow * " + std::to_string(kBRowsPerInvocation) + " + " + std::to_string(r);
}

// The four output columns of an invocation, as a vec4. Scalar B stores them kSubgroup16WorkGroupSizeX apart.
std::string LoadBCache(bool is_vec4, const std::string& row) {
  if (is_vec4) {
    return "      BCache = " + BSub("curr", row, "tileCol") + ";\n";
  }
  std::string cols;
  for (uint32_t j = 0; j < 4; j++) {
    cols += (j ? ", " : "") + BSub("curr", row, "tileCol + " + std::to_string(j * kSubgroup16WorkGroupSizeX));
  }
  return "      BCache = vec4<b_element_t>(" + cols + ");\n";
}

std::string Accumulate(int64_t row, const std::string& a_value, bool use_f32_accumulation) {
  return "      acc_" + std::to_string(row) + " += " +
         (use_f32_accumulation ? "f32(" + a_value + ") * vec4<f32>(BCache)" : a_value + " * BCache") + ";\n";
}

}  // namespace

bool UseMatMulSubgroup16Kernel(std::string_view architecture, uint32_t subgroup_size) {
  // Alchemist runs the default kernel at subgroup size 32 with heavy register spills (GitHub issue #33856). Add an
  // architecture here only after measuring that the subgroup-16 kernel beats both the default and packed kernels.
  return subgroup_size == kMatMulSubgroup16Size && architecture == gpu_arch::kGen12Hp;
}

int64_t MatMulSubgroup16ElementsPerThreadY(uint32_t m, uint32_t n, uint64_t batch_size,
                                           bool f32_accumulator, bool a_vec4) {
  int64_t rows = m <= 8 ? 1 : (m <= 16 ? 2 : (m <= 32 ? 4 : 8));
  // Eight rows of f32 accumulators plus scalar A values exceed the register budget and spill.
  if (f32_accumulator && !a_vec4) {
    rows = std::min<int64_t>(rows, 4);
  }
  // Taller tiles reuse each B element more, but launch fewer workgroups. Keep the most rows that still launch
  // kMinWorkgroups; f32 accumulators never drop below 4 rows and f16 below 2.
  const int64_t min_rows = std::min<int64_t>(rows, f32_accumulator ? 4 : 2);
  const uint64_t workgroups_x = CeilDiv(uint64_t{n}, uint64_t{kSubgroup16WorkGroupSizeX} * 4);
  for (; rows > min_rows; rows /= 2) {
    const uint64_t workgroups_xy =
        workgroups_x * CeilDiv(uint64_t{m}, uint64_t{kSubgroup16WorkGroupSizeY} * static_cast<uint64_t>(rows));
    // Written as a division so that a large batch cannot overflow the product.
    if (workgroups_xy >= kMinWorkgroups ||
        (workgroups_xy > 0 && batch_size >= CeilDiv(kMinWorkgroups, workgroups_xy))) {
      break;
    }
  }
  return rows;
}

Status MakeMatMulSubgroup16Source(ShaderHelper& shader,
                                  const InlinedVector<int64_t>& elements_per_thread,
                                  const ShaderIndicesHelper* batch_dims,
                                  bool is_vec4,
                                  bool a_vec4,
                                  bool use_f32_accumulation,
                                  float alpha,
                                  bool need_handle_matmul) {
  const int64_t elements_per_thread_x = elements_per_thread[0];
  const int64_t elements_per_thread_y = elements_per_thread[1];
  // vec4 A loads give each lane ceil(rows / 4) vec4 values, which needs rows <= 4 or a multiple of 4.
  ORT_RETURN_IF_NOT(elements_per_thread_x == 4 && elements_per_thread_y >= 1 &&
                        (elements_per_thread_y <= 4 || elements_per_thread_y % 4 == 0),
                    "Unsupported elements_per_thread for the subgroup MatMul: ", elements_per_thread_x, "x",
                    elements_per_thread_y);

  const int64_t tile_a_outer = kSubgroup16WorkGroupSizeY * elements_per_thread_y;
  // A B tile row holds kSubgroup16WorkGroupSizeX vec4 values, or four times as many scalars.
  const int64_t b_columns = is_vec4 ? kSubgroup16WorkGroupSizeX
                                    : kSubgroup16WorkGroupSizeX * elements_per_thread_x;
  const std::string accumulator_type = use_f32_accumulation ? "f32" : "output_element_t";
  const std::string batch_arg = batch_dims ? ", batchIndices" : "";

  InitializeLogicalWorkgroupIDAndGlobalID(shader);

  auto& body = shader.MainFunctionBody();
  body << "  let batch = i32(logical_workgroup_id.z);\n"
       << "  let tileCol = i32(local_id.x % " << kSubgroup16WorkGroupSizeX << ");\n"
       << "  let localRow = i32(local_id.x / " << kSubgroup16WorkGroupSizeX << ");\n"
       << (batch_dims ? "  let batchIndices = " + batch_dims->OffsetToIndices("u32(batch)") + ";\n" : "")
       << "  let globalRowStart = i32(logical_workgroup_id.y) * " << tile_a_outer << " + localRow * "
       << elements_per_thread_y << ";\n"
       << "  let globalColStart = i32(logical_workgroup_id.x) * " << b_columns << " + tileCol;\n";
  for (int64_t i = 0; i < elements_per_thread_y; i++) {
    body << "  var acc_" << i << " = vec4<" << accumulator_type << ">(0);\n";
  }

  // With an empty A or B (Gemm), the inputs and their type aliases are absent and the product is zero.
  if (need_handle_matmul) {
    // B tiles are double-buffered in workgroup memory: while the current tile is used, the next one is loaded into
    // registers, so its global-memory latency overlaps the math and each tile needs a single barrier.
    shader.AdditionalImplementation() << "var<workgroup> mm_Bsub: array<array<array<b_value_t, " << b_columns
                                      << ">, " << kTileK << ">, 2>;\n";
    // (workgroup-memory column, global column) of every B element an invocation loads per tile row.
    std::vector<std::pair<std::string, std::string>> b_columns_to_load;
    for (int64_t j = 0; j < (is_vec4 ? 1 : elements_per_thread_x); j++) {
      const std::string offset = j ? " + " + std::to_string(j * kSubgroup16WorkGroupSizeX) : "";
      b_columns_to_load.emplace_back("tileCol" + offset, "globalColStart" + offset);
    }

    // A is read by subgroup lane, not by tileCol: subgroupBroadcast below addresses lanes.
    body << "  let lane = i32(sg_id);\n"
         << "  let numTiles = (uniforms.dim_inner - 1) / " << kTileK << " + 1;\n"
         << "  var BCache: vec4<b_element_t>;\n";
    // vec4 A: lane l holds row (l / 4 + 4 * j), vec4 column (l % 4) in a_j. With fewer than 4 rows the extra
    // lanes load duplicates. Scalar A: every lane holds K element `lane` of each row.
    const int64_t a_vec4_slots = std::max<int64_t>(1, elements_per_thread_y / 4);
    for (int64_t i = 0; i < (a_vec4 ? a_vec4_slots : elements_per_thread_y); i++) {
      body << "  var a_" << i << " = a_value_t(0);\n";
    }

    for (uint32_t r = 0; r < kBRowsPerInvocation; r++) {
      for (const auto& [smem_col, global_col] : b_columns_to_load) {
        body << "  " << BSub("0", BRow(r), smem_col) << " = mm_readB(batch, " << BRow(r) << ", " << global_col
             << batch_arg << ");\n";
      }
    }
    body << "  workgroupBarrier();\n"
         << "  for (var t = 0; t < i32(numTiles); t++) {\n"
         << "    let kStart = t * " << kTileK << ";\n"
         << "    let curr = t % 2;\n";
    // mm_readB returns zero past dim_inner, so the prefetch in the last iteration is harmless.
    for (uint32_t r = 0; r < kBRowsPerInvocation; r++) {
      for (size_t c = 0; c < b_columns_to_load.size(); c++) {
        body << "    let nextB_" << r << "_" << c << " = mm_readB(batch, kStart + " << kTileK << " + " << BRow(r)
             << ", " << b_columns_to_load[c].second << batch_arg << ");\n";
      }
    }

    if (a_vec4) {
      for (int64_t j = 0; j < a_vec4_slots; j++) {
        const std::string row = elements_per_thread_y >= 4 ? "lane / 4 + " + std::to_string(4 * j)
                                                           : "(lane / 4) % " + std::to_string(elements_per_thread_y);
        body << "    a_" << j << " = mm_readA(batch, globalRowStart + " << row << ", kStart / 4 + lane % 4"
             << batch_arg << ");\n";
      }
      // One vec4 broadcast per row serves four K steps.
      for (uint32_t q = 0; q < kTileK / 4; q++) {
        body << "    {\n";
        for (int64_t i = 0; i < elements_per_thread_y; i++) {
          body << "      let aRow_" << i << " = subgroupBroadcast(a_" << i / 4 << ", " << (i % 4) * 4 + q << ");\n";
        }
        for (uint32_t c = 0; c < 4; c++) {
          body << LoadBCache(is_vec4, std::to_string(q * 4 + c));
          for (int64_t i = 0; i < elements_per_thread_y; i++) {
            body << Accumulate(i, "aRow_" + std::to_string(i) + "[" + std::to_string(c) + "]",
                               use_f32_accumulation);
          }
        }
        body << "    }\n";
      }
    } else {
      for (int64_t i = 0; i < elements_per_thread_y; i++) {
        body << "    a_" << i << " = mm_readA(batch, globalRowStart + " << i << ", kStart + lane" << batch_arg
             << ");\n";
      }
      for (uint32_t k = 0; k < kTileK; k++) {
        body << LoadBCache(is_vec4, std::to_string(k));
        for (int64_t i = 0; i < elements_per_thread_y; i++) {
          body << Accumulate(i, "subgroupBroadcast(a_" + std::to_string(i) + ", " + std::to_string(k) + ")",
                             use_f32_accumulation);
        }
      }
    }

    // The other buffer was last read in iteration t - 1, which every invocation finished before its barrier.
    for (uint32_t r = 0; r < kBRowsPerInvocation; r++) {
      for (size_t c = 0; c < b_columns_to_load.size(); c++) {
        body << "    " << BSub("1 - curr", BRow(r), b_columns_to_load[c].first) << " = nextB_" << r << "_" << c
             << ";\n";
      }
    }
    body << "    workgroupBarrier();\n"
         << "  }\n";

    if (alpha != 1.0f) {
      for (int64_t i = 0; i < elements_per_thread_y; i++) {
        body << "  acc_" << i << " *= " << accumulator_type << "(uniforms.alpha);\n";
      }
    }
  }

  for (int64_t i = 0; i < elements_per_thread_y; i++) {
    if (is_vec4) {
      body << "  mm_write(batch, globalRowStart + " << i << ", globalColStart, output_value_t(acc_" << i << "));\n";
    } else {
      for (int64_t j = 0; j < elements_per_thread_x; j++) {
        body << "  mm_write(batch, globalRowStart + " << i << ", globalColStart + " << j * kSubgroup16WorkGroupSizeX
             << ", output_value_t(acc_" << i << "[" << j << "]));\n";
      }
    }
  }

  return Status::OK();
}

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
