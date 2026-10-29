#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace onnxruntime::contrib::cuda::qmoe {

struct Nvfp4BankDims {
  bool valid = false;
  int64_t hidden = 0;
  int64_t inter = 0;
};

// Row-major FP4 expert banks use the same bytes per expert in two accepted weight shapes:
//   standard: FC1 [E, 2*I, H/2], FC2 [E, H, I/2]
//   legacy:   FC1 [E, H, I],     FC2 [E, I, H/2]
// The forms cannot be confused: FC1.dim1 / FC2.dim2 is 4 for standard and 2 for legacy. The block-scale banks must
// hold exactly [E, 2*I, H/scale_group] and [E, H, I/scale_group] bytes, which is what per-expert slicing assumes.
// Shape is any type with NumDimensions() and operator[] (for example TensorShape).
template <typename Shape>
inline Nvfp4BankDims ResolveNvfp4BankDims(const Shape& fc1_weights, const Shape& fc2_weights, int64_t scale_group,
                                          uint64_t fc1_scale_bytes, uint64_t fc2_scale_bytes,
                                          uint64_t scale_elem_bytes = 1) {
  Nvfp4BankDims dims;
  if (fc1_weights.NumDimensions() != 3 || fc2_weights.NumDimensions() != 3 || scale_group <= 0 ||
      scale_elem_bytes == 0) {
    return dims;
  }
  const int64_t experts = fc1_weights[0];
  const int64_t fc1_dim1 = fc1_weights[1];
  const int64_t fc1_dim2 = fc1_weights[2];
  const int64_t fc2_dim1 = fc2_weights[1];
  const int64_t fc2_dim2 = fc2_weights[2];
  if (experts <= 0 || fc2_weights[0] != experts || fc1_dim1 <= 0 || fc1_dim2 <= 0 || fc2_dim1 <= 0 ||
      fc2_dim2 <= 0) {
    return dims;
  }

  constexpr int64_t kMaxDim = std::numeric_limits<int32_t>::max();
  // The raw dimensions of a valid layout are H/2, H, I/2 or 2*I with H and I at most kMaxDim, so anything above
  // 2 * kMaxDim is rejected before the products below, which then stay far below INT64_MAX.
  constexpr int64_t kMaxRawDim = 2 * kMaxDim;
  if (fc1_dim1 > kMaxRawDim || fc1_dim2 > kMaxRawDim || fc2_dim1 > kMaxRawDim || fc2_dim2 > kMaxRawDim) {
    return dims;
  }

  int64_t hidden = 0;
  int64_t inter = 0;
  if (fc1_dim1 == fc2_dim2 * 2 && fc1_dim2 == fc2_dim1) {
    hidden = fc1_dim1;
    inter = fc2_dim1;
  } else if (fc1_dim1 == fc2_dim2 * 4 && fc1_dim2 * 2 == fc2_dim1) {
    hidden = fc2_dim1;
    inter = fc1_dim1 / 2;
  } else {
    return dims;
  }

  if (hidden > kMaxDim || inter > kMaxDim || experts > kMaxDim || hidden % scale_group != 0 ||
      inter % scale_group != 0) {
    return dims;
  }
  const uint64_t fc1_scale_per_expert =
      static_cast<uint64_t>(2 * inter) * static_cast<uint64_t>(hidden / scale_group) * scale_elem_bytes;
  const uint64_t fc2_scale_per_expert =
      static_cast<uint64_t>(hidden) * static_cast<uint64_t>(inter / scale_group) * scale_elem_bytes;
  if (fc1_scale_bytes % fc1_scale_per_expert != 0 || fc1_scale_bytes / fc1_scale_per_expert != static_cast<uint64_t>(experts) ||
      fc2_scale_bytes % fc2_scale_per_expert != 0 || fc2_scale_bytes / fc2_scale_per_expert != static_cast<uint64_t>(experts)) {
    return dims;
  }
  dims.valid = true;
  dims.hidden = hidden;
  dims.inter = inter;
  return dims;
}

template <typename Shape>
inline Nvfp4BankDims ResolveInt3BankDims(const Shape& fc1_weights, const Shape& fc2_weights, int64_t scale_group,
                                         uint64_t fc1_scale_bytes, uint64_t fc2_scale_bytes,
                                         uint64_t scale_elem_bytes = sizeof(uint16_t)) {
  Nvfp4BankDims dims;
  if (fc1_weights.NumDimensions() != 3 || fc2_weights.NumDimensions() != 3 ||
      (scale_group != 32 && scale_group != 64 && scale_group != 128) || scale_elem_bytes == 0) {
    return dims;
  }
  const int64_t experts = fc1_weights[0];
  const int64_t fc1_dim1 = fc1_weights[1];
  const int64_t fc1_dim2 = fc1_weights[2];
  const int64_t fc2_dim1 = fc2_weights[1];
  const int64_t fc2_dim2 = fc2_weights[2];
  if (experts <= 0 || fc2_weights[0] != experts || fc1_dim1 <= 0 || fc1_dim2 <= 0 || fc2_dim1 <= 0 ||
      fc2_dim2 <= 0) {
    return dims;
  }

  constexpr int64_t kMaxDim = std::numeric_limits<int32_t>::max();
  constexpr int64_t kMaxRawDim = 2 * kMaxDim;
  if (fc1_dim1 > kMaxRawDim || fc1_dim2 > kMaxRawDim || fc2_dim1 > kMaxRawDim || fc2_dim2 > kMaxRawDim ||
      (fc1_dim1 % 2) != 0) {
    return dims;
  }

  const int64_t hidden = fc2_dim1;
  const int64_t inter = fc1_dim1 / 2;
  if (hidden <= 0 || inter <= 0 || hidden > kMaxDim || inter > kMaxDim || experts > kMaxDim ||
      hidden % 8 != 0 || inter % 8 != 0 || hidden % scale_group != 0 || inter % scale_group != 0) {
    return dims;
  }
  if (fc1_dim2 != (hidden * 3) / 8 || fc2_dim2 != (inter * 3) / 8) {
    return dims;
  }

  const uint64_t fc1_scale_per_expert =
      static_cast<uint64_t>(2 * inter) * static_cast<uint64_t>(hidden / scale_group) * scale_elem_bytes;
  const uint64_t fc2_scale_per_expert =
      static_cast<uint64_t>(hidden) * static_cast<uint64_t>(inter / scale_group) * scale_elem_bytes;
  if (fc1_scale_bytes % fc1_scale_per_expert != 0 || fc1_scale_bytes / fc1_scale_per_expert != static_cast<uint64_t>(experts) ||
      fc2_scale_bytes % fc2_scale_per_expert != 0 || fc2_scale_bytes / fc2_scale_per_expert != static_cast<uint64_t>(experts)) {
    return dims;
  }
  dims.valid = true;
  dims.hidden = hidden;
  dims.inter = inter;
  return dims;
}

template <typename T>
inline void TransposeOffloadScales(const T* source, T* destination, int columns, int blocks) {
  constexpr int column_tile = 32;
  constexpr int block_tile = 8;
  for (int column_start = 0; column_start < columns; column_start += column_tile) {
    const int column_end = std::min(columns, column_start + column_tile);
    for (int block_start = 0; block_start < blocks; block_start += block_tile) {
      const int block_end = std::min(blocks, block_start + block_tile);
      for (int column = column_start; column < column_end; ++column) {
        const auto* source_row = source + static_cast<size_t>(column) * blocks;
        for (int block = block_start; block < block_end; ++block) {
          destination[static_cast<size_t>(block) * columns + column] = source_row[block];
        }
      }
    }
  }
}

struct OffloadWave {
  int next_expert = 0;
  int expert_count = 0;
  int route_count = 0;
};

struct StreamExpertLocation {
  int buffer = -1;
  int slot = -1;
};

inline StreamExpertLocation FindLiveStreamExpert(int expert, const int* streamed_experts,
                                                 int streamed_count, int wave_size) {
  if (!streamed_experts || streamed_count <= 0 || wave_size <= 0) return {};
  const int waves = (streamed_count - 1) / wave_size + 1;
  // Only the final wave in each half of the two-buffer landing ring survives.
  const int first_live = std::max(0, waves - 2) * wave_size;
  const auto* end = streamed_experts + streamed_count;
  const auto* found = std::lower_bound(streamed_experts + first_live, end, expert);
  if (found == end || *found != expert) return {};
  const int index = static_cast<int>(found - streamed_experts);
  return {(index / wave_size) & 1, index % wave_size};
}

inline bool InitializeOffloadRoutes(const int* selected_experts, int route_count,
                                    int num_experts, int k, int* expert_to_slot) {
  if (route_count <= 0 || num_experts <= 0 || k <= 0 || k > num_experts || route_count % k != 0) {
    return false;
  }
  std::fill_n(expert_to_slot, num_experts, -1);
  for (int route = 0; route < route_count; ++route) {
    const int expert = selected_experts[route];
    if (expert < 0 || expert >= num_experts) {
      return false;
    }
    const int token_start = route - route % k;
    for (int previous = token_start; previous < route; ++previous) {
      if (selected_experts[previous] == expert) {
        return false;
      }
    }
    expert_to_slot[expert] = 0;
  }
  return true;
}

inline OffloadWave MakeOffloadWave(const int* selected_experts, int route_count,
                                   int num_experts, int first_expert, int wave_capacity,
                                   int* expert_to_slot, int* wave_experts,
                                   int* route_ids, int* compact_experts) {
  OffloadWave wave;
  wave.next_expert = first_expert;
  while (wave.next_expert < num_experts && wave.expert_count < wave_capacity) {
    const int expert = wave.next_expert++;
    if (expert_to_slot[expert] >= 0) {
      expert_to_slot[expert] = wave.expert_count;
      wave_experts[wave.expert_count++] = expert;
    }
  }
  for (int route = 0; route < route_count; ++route) {
    const int expert = selected_experts[route];
    if (expert >= first_expert && expert < wave.next_expert && expert_to_slot[expert] >= 0) {
      route_ids[wave.route_count] = route;
      compact_experts[wave.route_count++] = expert_to_slot[expert];
    }
  }
  return wave;
}

// Where a QMoE layer sits for cross-layer promotion. The doorbell worker is shared by every session in the process,
// so a layer is identified by its owning session as well as by its layer index.
struct CrossLayerSite {
  const void* session = nullptr;  // identity of the owning session; null when unknown
  int64_t layer_index = 0;
  bool doorbell_pending = false;  // routes already classified for this step, CPU service not finished yet
};

// A cross-layer promotion rewrites the target layer's slot table and host maps from the doorbell worker. That is
// only sound for a layer that has not classified its routes for the current step. One session runs on one stream
// and rejects overlapped runs, so a LATER layer of the SAME session is still queued behind the servicing layer's
// merge kernel. A layer of another session has no such ordering: its GPU may already have classified with the old
// placement, and the new placement would then be mixed with those decisions.
inline bool IsCrossLayerLookaheadTarget(const CrossLayerSite& servicing, const CrossLayerSite& target) noexcept {
  return servicing.session != nullptr && servicing.session == target.session &&
         target.layer_index > servicing.layer_index;
}

// Publication boundary: besides being a look-ahead target, the target must not have classified yet. A raised
// doorbell request means it has, and its current execution still depends on the old placement.
inline bool CanPublishCrossLayerPromotion(const CrossLayerSite& servicing, const CrossLayerSite& target) noexcept {
  return IsCrossLayerLookaheadTarget(servicing, target) && !target.doorbell_pending;
}

}  // namespace onnxruntime::contrib::cuda::qmoe
