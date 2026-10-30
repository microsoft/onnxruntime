// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/framework/execution_provider.h"
#include "core/optimizer/graph_optimizer_registry.h"

namespace onnxruntime::cuda_runtime_fusion {

inline constexpr char kCausalConvConfig[] = "ep.cuda.runtime_fusion.causal_conv_with_state";
inline constexpr char kMobiusBf16Rule[] = "causal_conv_with_state_mobius_bf16_v1";
inline constexpr char kUnitSwishConfig[] = "ep.cuda.runtime_fusion.unit_swish";
inline constexpr char kUnitSwishRule[] = "unit_swish_scaled_silu_v1";

// Read-only discovery. Requires empty output; publishes capabilities only on success.
Status Select(const GraphViewer& graph, const IExecutionProvider::IKernelLookup& kernel_lookup,
              const GraphOptimizerRegistry& registry, const std::string& rule_id,
              std::vector<std::unique_ptr<ComputeCapability>>& capabilities);

}  // namespace onnxruntime::cuda_runtime_fusion
