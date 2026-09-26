// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/vendor/intel/math/split_k_config.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

SplitKConfig CreateSplitKConfig(std::string_view architecture) {
  // Disable Split-K on old Intel GPUs.
  if (architecture == "gen-7" || architecture == "gen-8" ||
      architecture == "gen-9" || architecture == "gen-11") {
    return {};
  }

  constexpr uint32_t max_batch_size = 8;
  constexpr uint32_t split_dim_inner = 256;
  constexpr uint32_t min_dim_inner_with_split_k = split_dim_inner * 2;

  if (architecture == "xe-2lpg" || architecture == "xe-2hpg" ||
      architecture == "gen-12hp") {
    // These thresholds are verified on Intel discrete GPUs and Lunar Lake iGPUs.
    return SplitKConfig{
        max_batch_size, split_dim_inner, min_dim_inner_with_split_k, {{768, 52.0}, {2304, 35.0}, {3072, 21.5}, {4096, 16.0}}};
  }

  if (architecture == "xe-3lpg") {
    // These thresholds are verified on Intel Panther Lake iGPUs (12Xe).
    return SplitKConfig{
        max_batch_size, split_dim_inner, min_dim_inner_with_split_k, {{768, 40.0}, {1792, 22.0}, {3072, 18.0}, {4096, 10.0}}};
  }

  // Default thresholds for newer Intel GPUs, chosen on a gen-12lp GPU with 32 EUs.
  return SplitKConfig{
      max_batch_size, split_dim_inner, min_dim_inner_with_split_k, {{768, 20.0}, {1792, 13.0}, {3072, 8.0}, {4096, 6.0}}};
}

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
