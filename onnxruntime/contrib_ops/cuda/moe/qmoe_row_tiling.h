// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

#include "core/common/common.h"
#include "core/common/safeint.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace qmoe {

constexpr int64_t kDisabledRowTileSize = 0;

constexpr bool IsValidRowTileSize(int64_t row_tile_size) {
  return row_tile_size > 0 && row_tile_size <= std::numeric_limits<int>::max();
}

struct RowTilePlan {
  int64_t num_rows;
  int64_t rows_per_tile;

  int64_t TileCount() const {
    return num_rows / rows_per_tile + (num_rows % rows_per_tile != 0 ? 1 : 0);
  }

  int64_t RowOffset(int64_t tile_index) const {
    ORT_ENFORCE(tile_index >= 0 && tile_index < TileCount(),
                "QMoE row tile index ", tile_index, " is outside [0, ", TileCount(), ").");
    return SafeInt<int64_t>(tile_index) * rows_per_tile;
  }

  int64_t RowsInTile(int64_t tile_index) const {
    return std::min(rows_per_tile, num_rows - RowOffset(tile_index));
  }

  bool IsTiled() const {
    return rows_per_tile < num_rows;
  }
};

inline RowTilePlan MakeRowTilePlan(int64_t num_rows, int64_t row_tile_size, bool enable_tiling) {
  ORT_ENFORCE(num_rows > 0, "QMoE row tiling requires a positive row count, got ", num_rows, ".");
  if (!enable_tiling) {
    return RowTilePlan{num_rows, num_rows};
  }
  ORT_ENFORCE(IsValidRowTileSize(row_tile_size),
              "QMoE row tile size must be in [1, ", std::numeric_limits<int>::max(),
              "], got ", row_tile_size, ".");
  return RowTilePlan{num_rows, std::min(num_rows, row_tile_size)};
}

struct ScratchLayout {
  size_t scales_bytes;
  size_t indices_bytes;
  size_t permutation_bytes;
  size_t total_bytes;
};

inline ScratchLayout MakeScratchLayout(size_t runner_workspace_bytes,
                                       const RowTilePlan& row_tile_plan,
                                       int64_t experts_per_token) {
  ORT_ENFORCE(experts_per_token > 0,
              "QMoE scratch layout requires a positive experts-per-token count, got ",
              experts_per_token, ".");

  const SafeInt<size_t> expanded_rows =
      SafeInt<size_t>(row_tile_plan.rows_per_tile) * SafeInt<size_t>(experts_per_token);
  const size_t scales_bytes = expanded_rows * sizeof(float);
  const size_t indices_bytes = expanded_rows * sizeof(int);
  const size_t permutation_bytes = expanded_rows * sizeof(int);
  const size_t total_bytes = SafeInt<size_t>(runner_workspace_bytes) +
                             scales_bytes + indices_bytes + permutation_bytes;
  return ScratchLayout{scales_bytes, indices_bytes, permutation_bytes, total_bytes};
}

}  // namespace qmoe
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
