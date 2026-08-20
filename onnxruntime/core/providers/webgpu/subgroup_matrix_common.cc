// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(__wasm__)

#include "core/providers/webgpu/subgroup_matrix_common.h"

#include <string_view>
#include <utility>

#include "core/providers/webgpu/math/subgroup_matrix_config.h"
#include "core/providers/webgpu/vendor/intel/math/subgroup_matrix_tiling_selector.h"

namespace onnxruntime {
namespace webgpu {

SubgroupMatrixTilingSelector MakeDefaultSubgroupMatrixTilingSelector() {
  return [](const ComputeContext&, uint32_t /*M*/, uint32_t /*N*/,
            uint32_t K, uint32_t /*batch*/) -> std::optional<SubgroupMatrixTiling> {
    // Only K needs to align to the subgroup-matrix shape; M and N partial tiles are
    // handled by bounds-checked stores in the kernel. Decline a misaligned K, which
    // the kernel would otherwise silently truncate to whole blocks.
    if (K % kSubgroupMatrixK != 0) {
      return std::nullopt;
    }
    // 32x32 is a multiple of the subgroup-matrix M/N and fits the scratch budget.
    return SubgroupMatrixTiling{32, 32, 1};
  };
}

bool TrySelectSubgroupMatrixConfig(const ComputeContextBase& context,
                                   int32_t& config_index,
                                   SubgroupMatrixTilingSelector& tiling_selector) {
  // Only devices reporting the fixed 8x16x16 F16 subgroup-matrix config these kernels
  // are implemented for are supported.
  config_index = 0;
  if (!IsSubgroupMatrixConfigSupported(context, /*is_fp16=*/true, config_index) ||
      !supported_subgroup_matrix_configs[config_index].Is(kSubgroupMatrixM, kSubgroupMatrixN, kSubgroupMatrixK)) {
    return false;
  }
  // Intel GPUs use a tuned/heuristic tiling policy; every other vendor falls back to a
  // fixed default tiling.
  const bool is_intel = context.AdapterInfo().vendor == std::string_view{"intel"};
  SubgroupMatrixTilingSelector selector =
      is_intel ? intel::CreateSubgroupMatrixTilingSelector(context) : MakeDefaultSubgroupMatrixTilingSelector();
  if (!selector) {
    return false;
  }
  tiling_selector = std::move(selector);
  return true;
}

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
