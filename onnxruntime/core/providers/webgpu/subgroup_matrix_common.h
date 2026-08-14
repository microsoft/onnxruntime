// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(__wasm__)

#include <cstdint>
#include <functional>
#include <optional>

#include "core/providers/webgpu/compute_context.h"

namespace onnxruntime {
namespace webgpu {

// Lanes per subgroup assumed by the subgroup-matrix kernels. The workgroup runs
// split_k subgroups, so its size is kSubgroupMatrixSubgroupSize * split_k.
// TODO: use subgroup-size-control to enforce the subgroup size is 32.
constexpr uint32_t kSubgroupMatrixSubgroupSize = 32;

// Per-workgroup output tiling for one subgroup-matrix problem: the tile shape and
// split-K factor chosen by a vendor-specific policy. The subgroup-matrix shape
// itself is separate from this selection.
struct SubgroupMatrixTiling {
  uint32_t tile_m;   // output rows per workgroup
  uint32_t tile_n;   // output cols per workgroup
  uint32_t split_k;  // subgroups cooperating along K (1 = no split)
};

// Vendor-supplied callback that selects the output tiling for a given problem.
// batch is the number of z-dispatched slices (1 for a shared 2D weight), used by
// the policy to scale occupancy. Returning nullopt declines the problem, so the
// caller falls back to another compute path. An empty selector likewise yields no
// implementation.
using SubgroupMatrixTilingSelector =
    std::function<std::optional<SubgroupMatrixTiling>(const ComputeContext& context,
                                                      uint32_t M, uint32_t N, uint32_t K, uint32_t batch)>;

// Default tiling used on any vendor without a specialized policy: a fixed 32x32
// output tile with no split-K. The fallback selector when no vendor policy applies.
SubgroupMatrixTilingSelector MakeDefaultSubgroupMatrixTilingSelector();

// Selects the device subgroup-matrix config and the vendor tiling selector shared by
// the MatMul, Gemm and Conv 1x1 factories. On a supported device sets config_index /
// tiling_selector and returns true; returns false (leaving the outputs untouched) when
// the device or vendor is unsupported, so the caller yields no implementation.
bool TrySelectSubgroupMatrixConfig(const ComputeContextBase& context,
                                   /*out*/ int32_t& config_index,
                                   /*out*/ SubgroupMatrixTilingSelector& tiling_selector);

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
