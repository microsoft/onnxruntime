// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/math/subgroup_matrix_tiling.h"
#include "core/providers/webgpu/vendor/intel/intel_device_info.h"
#include "core/providers/webgpu/vendor/intel/math/subgroup_matrix_tiling_selector.h"

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
// Declared in subgroup_matrix_tiling_selector.h (as intel::kMaxScratchElems) so the
// Conv kernel's own split-K safety clamp can reuse the same value instead of
// hardcoding a second copy that has to be kept in sync by hand.

// Extra per-tile N a kernel stages in workgroup memory beyond the split-K output
// tiles. The MatMul kernel stages nothing else (0); the Conv kernel also holds the
// im2col A tile (tile_m x sg_mat_k), so its scratch bound is
// split_k * tile_m * (tile_n + sg_mat_k) -- up to 2x stricter than MatMul's. This is
// the only difference in the scratch-budget math shared by ScratchElems,
// IsTilingValid and ClampSplitKToProblem below, so those helpers take it as a
// parameter instead of being duplicated per kernel.
// (The two kernels' tiling constraints differ elsewhere too -- e.g. ClampTileNToProblem,
// and MatMul alone has a pretuned table to consult before its heuristic -- those are
// handled separately.)
constexpr uint32_t kMatMulScratchNPad = 0;
constexpr uint32_t kConvScratchNPad = kSubgroupMatrixK;

// Workgroup-memory footprint of a tiling, in f16 elements.
uint64_t ScratchElems(const SubgroupMatrixTiling& t, uint32_t scratch_n_pad) {
  return static_cast<uint64_t>(t.split_k) * t.tile_m * (t.tile_n + scratch_n_pad);
}

// Hard constraints a tiling must satisfy to run correctly for this problem.
// Used to reject a pretuned entry whose bucket does not fit the actual K.
bool IsTilingValid(const SubgroupMatrixTiling& t, uint32_t K, uint32_t scratch_n_pad) {
  if (t.tile_m == 0 || t.tile_n == 0 ||
      t.tile_m % kSubgroupMatrixM != 0 || t.tile_n % kSubgroupMatrixN != 0) {
    return false;
  }
  const uint32_t k_blocks = K / kSubgroupMatrixK;
  if (t.split_k == 0 || t.split_k > k_blocks) {
    return false;
  }
  return ScratchElems(t, scratch_n_pad) <= kMaxScratchElems;
}

// HwSubgroups returns 0 for an unrecognized arch; fall back to a conservative
// default so the occupancy target is still reasonable.
uint32_t EffectiveHwSubgroups(std::string_view arch) {
  const uint32_t hw = HwSubgroups(arch);
  return hw == 0 ? 256u : hw;
}

// Fallback MatMul tiling selection when no pretuned entry applies. The goal is to
// keep the GPU busy without over-subscribing it: pick the tile whose independent
// output-tile grid just fills the resident subgroups, preferring larger tiles
// (more data reuse) among those that qualify, and only using smaller tiles when
// nothing else would fill the machine. Split-K then adds cooperative subgroups to
// reach up to 2x occupancy, subject to the per-subgroup K-work minimum and the
// scratch (SLM) budget. batch scales the independent-tile count: each z slice
// contributes its own output-tile grid, so a larger batch fills the machine with
// bigger tiles and needs less (or no) split-K.
//
// Deliberately not shared with ConvHeuristicTiling below, even where the two look
// similar: shrinking tile_n to manufacture more occupancy tiles is free here (for
// MatMul, tile_n only trades off against tile_m for a fixed amount of
// global-memory traffic) but not for Conv -- see that function's comment -- so the
// two kernels' tile and split-K policies are expected to keep diverging and tend
// to get tuned/debugged independently. A shared helper would just mean every
// future change has to re-check it does not perturb the other kernel.
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
        ScratchElems({tile_m, tile_n, cand}, kMatMulScratchNPad) <= kMaxScratchElems) {
      split_k = cand;
    }
  }
  return {tile_m, tile_n, split_k};
}

// Once N needs at least two full-width (kTileNCandidates' max) N tiles, gather
// volume is already at its floor (see ConvHeuristicTiling's comment), and real
// hardware profiling (an xe-3lpg shape sweep) showed split-K keeps paying off past
// the point where occupancy alone would call the machine full: the win there is
// shortening each workgroup's serial K-reduction chain (gather-latency hiding), not
// filling idle subgroups, so a large M does not make split-K less useful the way it
// would under an occupancy model. kConvWideN marks where the search below drops its
// occupancy cap accordingly.
constexpr uint32_t kConvWideN = 2 * kTileNCandidates[2];  // 128: needs 2+ tile_n=64 tiles

// Conv tiling heuristic: the sole tiling strategy SelectConvTiling uses, on every
// architecture. (A separate decision tree fit to a shape sweep on one GPU used to
// cover that one architecture; a sweep like that does not generalize to untested
// hardware, so it was dropped in favor of this heuristic alone, pending a more
// broadly-informed retuning in a follow-up.)
//
// Does *not* pick tile_m/tile_n the way HeuristicTiling does. Shrinking tile_n to
// manufacture more occupancy tiles is free for MatMul (tile_n only trades off
// against tile_m for a fixed amount of global-memory traffic), but not for Conv:
// the im2col A tile is staged per-workgroup in workgroup memory, not shared, so it
// is re-gathered once per N tile that covers it -- gather volume is
// M * N * K / tile_n, independent of tile_m or split_k. A real hardware sweep
// showed HeuristicTiling's occupancy-driven choice shrinking tile_n to 16-32 on
// shapes that profiled best at 64, multiplying gather volume 2x-4x regardless of
// how split-K was tuned (why an earlier split-K-only fix did not help).
//
// So tile_n is fixed at the largest candidate that fits N, and tile_m at the
// smallest (splitting M adds tiles without touching gather volume). split-K then
// takes the largest candidate with enough K-blocks per split and scratch headroom,
// subject to an occupancy cap below kConvWideN (see its comment for why that cap is
// dropped above it).
SubgroupMatrixTiling ConvHeuristicTiling(std::string_view arch, uint32_t M, uint32_t N, uint32_t K, uint32_t batch) {
  const uint32_t k_blocks = K / kSubgroupMatrixK;

  uint32_t tile_n = kTileNCandidates[0];
  for (uint32_t tn : kTileNCandidates) {
    if (tn <= N) {
      tile_n = tn;
    }
  }
  const uint32_t tile_m = kTileMCandidates[0];

  const bool wide_n = N >= kConvWideN;
  const uint32_t hw = EffectiveHwSubgroups(arch);
  auto tile_count = [](uint32_t dim, uint32_t tile) { return (dim + tile - 1) / tile; };
  const uint32_t num_tiles = batch * tile_count(M, tile_m) * tile_count(N, tile_n);

  constexpr uint32_t kMinBlocksPerSplit = 2;
  uint32_t split_k = 1;
  for (uint32_t cand : kSplitKCandidates) {
    if (k_blocks >= cand * kMinBlocksPerSplit &&
        (wide_n || num_tiles * cand <= 2 * hw) &&
        ScratchElems({tile_m, tile_n, cand}, kConvScratchNPad) <= kMaxScratchElems) {
      split_k = cand;
    }
  }
  return {tile_m, tile_n, split_k};
}

// Repairs a predicted tiling to this problem by halving split_k until it fits both
// K's block count and the scratch budget. ConvHeuristicTiling's own split-K search
// already checks both of these for the tiling it returns, so in practice this is a
// no-op safety net (mirroring the Conv kernel's own split-K clamp in
// subgroup_matrix_conv.cc) rather than something relied on to fire -- kept in case a
// future change to that search (or a future non-heuristic tiling source) ever
// predicts a split_k outside what K or the scratch budget can actually support.
// Clamping keeps the predicted tile shape -- the part that carries most of the win
// -- instead of discarding the prediction.
void ClampSplitKToProblem(SubgroupMatrixTiling& t, uint32_t K, uint32_t scratch_n_pad) {
  const uint32_t k_blocks = K / kSubgroupMatrixK;
  while (t.split_k > 1 && (t.split_k > k_blocks || ScratchElems(t, scratch_n_pad) > kMaxScratchElems)) {
    t.split_k /= 2;
  }
}

// Caps tile_n at the largest whole number of subgroup-matrix N blocks that fits in N.
//
// Correctness, not performance: the Conv kernel loads its B right operand straight
// from the weight buffer, and subgroupMatrixLoad is not bounds-checked, so a trailing
// partial N tile is shifted back to N - tile_n. That shift only exists when
// tile_n <= N. ConvHeuristicTiling already picks the largest candidate that fits N,
// but N itself need not be a multiple of kSubgroupMatrixN, so this cap is still
// needed. Shrinking tile_n keeps it a multiple of
// kSubgroupMatrixN and only lowers ScratchElems, so it cannot invalidate a tiling
// that was already valid.
//
// Callers must have rejected N < kSubgroupMatrixN, which cannot be served at all
// (there is no non-zero multiple of kSubgroupMatrixN that fits).
void ClampTileNToProblem(SubgroupMatrixTiling& t, uint32_t N) {
  const uint32_t max_tile_n = (N / kSubgroupMatrixN) * kSubgroupMatrixN;
  if (t.tile_n > max_tile_n) {
    t.tile_n = max_tile_n;
  }
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
//
// Only for the batch-blind MatMul pretuned table: SelectConvTiling's
// ConvHeuristicTiling already takes batch into account when picking split-K (see its
// narrow-N branch), so applying this on top of it would double-count occupancy.
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
  if (const auto tuned = LookupPretunedTiling(arch, M, N, K); tuned && IsTilingValid(*tuned, K, kMatMulScratchNPad)) {
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

// Conv counterpart of SelectTiling: always goes through ConvHeuristicTiling, under
// the Conv scratch budget. There is no MatMul-table lookup here: the Conv kernel's
// cost profile (the im2col gather) differs, and its scratch budget is stricter, so
// it must not reuse the MatMul-tuned entries; and unlike SelectTiling, there is no
// pretuned Conv entry point at all to try first (see ConvHeuristicTiling's comment
// for why).
SubgroupMatrixTiling SelectConvTiling(std::string_view arch, uint32_t M, uint32_t N, uint32_t K, uint32_t batch) {
  SubgroupMatrixTiling tiling = ConvHeuristicTiling(arch, M, N, K, batch);
  ClampSplitKToProblem(tiling, K, kConvScratchNPad);
  ClampTileNToProblem(tiling, N);
  return tiling;
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

SubgroupMatrixTilingSelector CreateSubgroupMatrixConvTilingSelector(
    const ComputeContextBase& context) {
  if (context.AdapterInfo().vendor != std::string_view{"intel"}) {
    return nullptr;
  }
  // Conv tiling policy for the 8x16x16 subgroup-matrix Conv kernel: same tile
  // candidates as MatMul, but selection uses the Conv scratch budget (which also
  // stages the im2col A tile) and SelectConvTiling's ConvHeuristicTiling.
  return [](const ComputeContext& context, uint32_t M, uint32_t N,
            uint32_t K, uint32_t batch) -> std::optional<SubgroupMatrixTiling> {
    // K must align to the subgroup tiling, and N must hold at least one whole
    // subgroup-matrix N block: the Conv kernel loads B unchecked from the weight
    // buffer and shifts a trailing partial N tile back to N - tile_n, which is
    // impossible below one block. A partial N tile above that is served by the shift
    // (see ClampTileNToProblem); M partial tiles need no guarantee at all, since the
    // A operand is staged in workgroup memory sized by the tile.
    //
    // Declining here rather than clamping keeps this in agreement with
    // CanApplySubgroupMatrixConv, which rejects the same N so Conv::PrePackInternal
    // does not skip prepacking for a problem this path will not serve.
    if (K % kSubgroupMatrixK != 0 || N < kSubgroupMatrixN) {
      return std::nullopt;
    }
    const std::string_view arch = std::string_view{context.AdapterInfo().architecture};
    return SelectConvTiling(arch, M, N, K, batch);
  };
}

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
