// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include <array>
#include <cstddef>
#include <string_view>

#include <gsl/gsl>
#include "core/common/inlined_containers_fwd.h"
#include "core/framework/execution_provider.h"  // for IExecutionProvider::IKernelLookup
#include "core/graph/graph_viewer.h"

namespace onnxruntime {
namespace logging {
class Logger;
}

namespace fallback_cpu_capability_internal {

inline constexpr std::array<std::string_view, 7> kUnsupportedCpuFallbackTypes{
    "bfloat16",
    "float16",
    "float4e2m1",
    "float8e4m3fn",
    "float8e4m3fnuz",
    "float8e5m2",
    "float8e5m2fnuz",
};

consteval bool IsUnsupportedCpuFallbackTypeTableValid() {
  for (size_t index = 0; index < kUnsupportedCpuFallbackTypes.size(); ++index) {
    if (kUnsupportedCpuFallbackTypes[index].empty() ||
        (index > 0 && kUnsupportedCpuFallbackTypes[index - 1] >= kUnsupportedCpuFallbackTypes[index])) {
      return false;
    }
  }

  return true;
}

static_assert(IsUnsupportedCpuFallbackTypeTableValid(),
              "Unsupported CPU fallback types must be non-empty, unique, and sorted.");

constexpr bool IsUnsupportedCpuFallbackType(std::string_view type) noexcept {
  for (const auto unsupported_type : kUnsupportedCpuFallbackTypes) {
    if (type == unsupported_type) {
      return true;
    }
  }

  return false;
}

}  // namespace fallback_cpu_capability_internal

/**
  Returns a list of nodes that are preferred on CPU.
  They are commonly shape-related computation subgraphs.
  @param graph Graph viewer
  @param kernel_lookup The kernel lookup for the target execution provider
  @param tentative_nodes Nodes that are tentative to be placed on on target EP
  */
std::unordered_set<NodeIndex> GetCpuPreferredNodes(const GraphViewer& graph,
                                                   const IExecutionProvider::IKernelLookup& kernel_lookup,
                                                   gsl::span<const NodeIndex> tentative_nodes,
                                                   const logging::Logger& logger);

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
