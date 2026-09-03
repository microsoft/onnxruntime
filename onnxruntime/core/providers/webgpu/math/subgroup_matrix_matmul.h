// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <functional>
#include <optional>

namespace onnxruntime {
namespace webgpu {

class ComputeContext;

// Per-workgroup output tiling for one MatMul problem: the tile shape and split-K
// factor chosen by a vendor-specific policy. The subgroup-matrix shape itself is
// separate from this selection.
struct SubgroupMatrixTiling {
  uint32_t tile_m;   // output rows per workgroup
  uint32_t tile_n;   // output cols per workgroup
  uint32_t split_k;  // subgroups cooperating along K (1 = no split)
};

// Vendor-supplied callback that selects the output tiling for a given problem.
// batch is the number of z-dispatched slices (1 for a shared 2D weight), used by
// the policy to scale occupancy. Returning nullopt declines the problem, so
// MatMul falls back to another compute path. An empty selector likewise yields
// no implementation.
using SubgroupMatrixTilingSelector =
    std::function<std::optional<SubgroupMatrixTiling>(const ComputeContext& context,
                                                      uint32_t M, uint32_t N, uint32_t K, uint32_t batch)>;

}  // namespace webgpu
}  // namespace onnxruntime
