// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(__wasm__)

#include "core/providers/webgpu/vendor/intel/math/subgroup_matrix_tiling_selector.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include "core/common/narrow.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/math/subgroup_matrix_matmul.h"
#include "core/providers/webgpu/vendor/intel/intel_device_info.h"

// Pretuned tile + split-K table baked into the build; consulted before the
// heuristic in SelectTiling.
#include "core/providers/webgpu/vendor/intel/math/subgroup_matrix_matmul_tuned.inc"

namespace onnxruntime {
namespace webgpu {
namespace intel {

namespace {

// Subgroup matrix configuration used by this implementation (Intel Xe2/Xe3, F16).
constexpr uint32_t kSubgroupMatrixM = 8;
constexpr uint32_t kSubgroupMatrixN = 16;
constexpr uint32_t kSubgroupMatrixK = 16;

constexpr uint32_t kTileMCandidates[] = {8, 16, 32, 64};
constexpr uint32_t kTileNCandidates[] = {16, 32, 64};
constexpr uint32_t kSplitKCandidates[] = {1, 2, 4, 8};
// Scratch holds split_k partial f16 tiles; cap keeps it within the SLM budget.
constexpr uint32_t kMaxScratchElems = 16384;  // 32 KB

// Hard constraints a tiling must satisfy to run correctly for this problem.
// Used to reject a pretuned entry whose bucket does not fit the actual K.
bool IsTilingValid(const SubgroupMatrixTiling& t, uint32_t K) {
  if (t.tile_m == 0 || t.tile_n == 0 ||
      t.tile_m % kSubgroupMatrixM != 0 || t.tile_n % kSubgroupMatrixN != 0) {
    return false;
  }
  const uint32_t k_blocks = K / kSubgroupMatrixK;
  if (t.split_k == 0 || t.split_k > k_blocks) {
    return false;
  }
  return t.tile_m * t.tile_n * t.split_k <= kMaxScratchElems;
}

// HwSubgroups returns 0 for an unrecognized arch; fall back to a conservative
// default so the occupancy target is still reasonable.
uint32_t EffectiveHwSubgroups(std::string_view arch) {
  const uint32_t hw = HwSubgroups(arch);
  return hw == 0 ? 256u : hw;
}

// Fallback tiling selection when no pretuned entry applies. The goal is to keep
// the GPU busy without over-subscribing it: pick the tile whose independent
// output-tile grid just fills the resident subgroups, preferring larger tiles
// (more data reuse) among those that qualify, and only using smaller tiles when
// nothing else would fill the machine. Split-K then adds cooperative subgroups
// to reach up to 2x occupancy, subject to the per-subgroup K-work minimum and
// the scratch (SLM) budget. batch scales the independent-tile count: each z
// slice contributes its own output-tile grid, so a larger batch fills the
// machine with bigger tiles and needs less (or no) split-K.
SubgroupMatrixTiling HeuristicTiling(std::string_view arch, uint32_t M, uint32_t N, uint32_t K, uint32_t batch) {
  const uint32_t hw = EffectiveHwSubgroups(arch);
  const uint32_t k_blocks = K / kSubgroupMatrixK;
  auto tile_count = [](uint32_t dim, uint32_t tile) { return (dim + tile - 1) / tile; };

  // Choose tile_m x tile_n. Skip tiles larger than the dimension (they only waste
  // lanes). Prefer the largest-area tile whose grid already fills the hardware
  // (num_tiles >= hw); if none does, take the tile that yields the most
  // independent tiles (largest area breaks ties).
  uint32_t tile_m = kTileMCandidates[0];
  uint32_t tile_n = kTileNCandidates[0];
  uint32_t best_tiles = 0;  // most output tiles seen so far (used only while under-filling)
  uint32_t best_area = 0;   // tile_m * tile_n of the current pick (tie-breaker: larger reuse wins)
  bool filled = false;      // true once some tile reaches num_tiles >= hw (machine is filled)
  for (uint32_t tm : kTileMCandidates) {
    if (tm > M && tm != kTileMCandidates[0]) {
      continue;
    }
    for (uint32_t tn : kTileNCandidates) {
      if (tn > N && tn != kTileNCandidates[0]) {
        continue;
      }
      const uint32_t tiles = batch * tile_count(M, tm) * tile_count(N, tn);
      const uint32_t area = tm * tn;
      if (tiles >= hw) {
        // This tile fills the machine on its own. Among all such tiles, keep the
        // largest-area one (best operand reuse); this also supersedes any
        // under-filling pick made earlier.
        if (!filled || area > best_area) {
          filled = true;
          best_area = area;
          tile_m = tm;
          tile_n = tn;
        }
      } else if (!filled && (tiles > best_tiles || (tiles == best_tiles && area > best_area))) {
        // No tile has filled the machine yet: chase occupancy by maximizing the
        // number of independent tiles, breaking ties toward larger area.
        best_tiles = tiles;
        best_area = area;
        tile_m = tm;
        tile_n = tn;
      }
    }
  }

  // Add split-K only while the independent tiles under-fill the machine; cap the
  // total at 2x hardware and keep enough K work per subgroup and scratch budget.
  const uint32_t num_tiles = batch * tile_count(M, tile_m) * tile_count(N, tile_n);
  constexpr uint32_t kMinBlocksPerSplit = 2;
  uint32_t split_k = 1;
  for (uint32_t cand : kSplitKCandidates) {
    if (k_blocks >= cand * kMinBlocksPerSplit &&
        num_tiles * cand <= 2 * hw &&
        tile_m * tile_n * cand <= kMaxScratchElems) {
      split_k = cand;
    }
  }
  return {tile_m, tile_n, split_k};
}

// Looks up a pretuned tiling for this problem in the baked-in table
// (subgroup_matrix_matmul_tuned.inc). The table holds one sub-table per GPU
// architecture; we match `arch`, then map each problem dimension to the smallest
// grid value >= it (clamped to the largest bucket). Returns nullopt when the
// arch is not covered or no entry matches.
std::optional<SubgroupMatrixTiling> LookupPretunedTiling(std::string_view arch, uint32_t M, uint32_t N, uint32_t K) {
  for (const auto& table : sgmm_tuned::kArchTables) {
    if (table.arch != arch) {
      continue;
    }
    auto bucket = [&table](uint32_t d) -> int {
      int last = 0;
      for (size_t i = 0; i < table.grid_count; ++i) {
        if (table.grid[i] >= d) {
          return static_cast<int>(i);
        }
        last = static_cast<int>(i);
      }
      return last;
    };
    const int mi = bucket(M);
    const int ni = bucket(N);
    const int ki = bucket(K);
    for (size_t i = 0; i < table.entry_count; ++i) {
      const auto& e = table.entries[i];
      if (e.mi == mi && e.ni == ni && e.ki == ki) {
        return SubgroupMatrixTiling{e.tile_m, e.tile_n, e.split_k};
      }
    }
    break;  // arch matched but no entry for this problem; fall through to heuristic
  }
  return std::nullopt;
}

// Batch slices are dispatched on z as independent output-tile grids, so batch is
// a pure occupancy multiplier. The pretuned table and heuristic pick tile shape
// for a single (M, N, K) slice; once batch fills the machine, split-K's
// cooperative subgroups are redundant. Retire split-K factors whose batch-scaled
// occupancy would exceed ~2x hardware.
void ClampSplitKForBatch(SubgroupMatrixTiling& t, uint32_t M, uint32_t N, uint32_t batch, uint32_t hw) {
  auto tile_count = [](uint32_t dim, uint32_t tile) { return (dim + tile - 1) / tile; };
  const uint32_t eff_tiles = batch * tile_count(M, t.tile_m) * tile_count(N, t.tile_n);
  while (t.split_k > 1 && eff_tiles * t.split_k > 2 * hw) {
    t.split_k /= 2;
  }
}

// Chooses the tile + split-K tiling for the given problem: use the pretuned
// table entry when one exists and fits, otherwise fall back to the heuristic.
// batch is the number of z-dispatched slices; it scales occupancy (see
// ClampSplitKForBatch) but not the per-slice tile shape.
SubgroupMatrixTiling SelectTiling(std::string_view arch, uint32_t M, uint32_t N, uint32_t K, uint32_t batch) {
  if (const auto tuned = LookupPretunedTiling(arch, M, N, K); tuned && IsTilingValid(*tuned, K)) {
    SubgroupMatrixTiling tiling = *tuned;
    // The table is tuned per (M, N, K) at batch 1, so only revisit split-K when
    // batch adds occupancy.
    if (batch > 1) {
      ClampSplitKForBatch(tiling, M, N, batch, EffectiveHwSubgroups(arch));
    }
    return tiling;
  }
  return HeuristicTiling(arch, M, N, K, batch);
}

}  // namespace

SubgroupMatrixTilingSelector CreateSubgroupMatrixTilingSelector(
    const ComputeContextBase& context) {
  if (context.AdapterInfo().vendor != std::string_view{"intel"}) {
    return nullptr;
  }
  // Intel tiling policy consumed by the common 8x16x16 subgroup-matrix kernel:
  // the tile shape and split-K factor from the pretuned table or the heuristic.
  // The subgroup-matrix shape itself is fixed by the kernel and not selected here.
  return [](const ComputeContext& context, uint32_t M, uint32_t N,
            uint32_t K, uint32_t batch) -> std::optional<SubgroupMatrixTiling> {
    // Only K needs to align to the subgroup tiling; M and N partial tiles are
    // handled by bounds-checked stores in the kernel.
    if (K % kSubgroupMatrixK != 0) {
      return std::nullopt;
    }
    const std::string_view arch = std::string_view{context.AdapterInfo().architecture};
    return SelectTiling(arch, M, N, K, batch);
  };
}

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
