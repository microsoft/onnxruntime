// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include <gsl/span>

#include "core/common/inlined_containers.h"
#include "core/common/status.h"
#include "core/graph/basic_types.h"

namespace onnxruntime {

class GraphViewer;

struct GraphRegion {
  std::string id;
  InlinedVector<NodeIndex> nodes;
};

struct GraphRegionDiscoveryResult {
  // Regions have a unique dependency order; members follow the graph's topological order.
  InlinedVector<GraphRegion> regions;
  InlinedVector<NodeIndex> ungrouped_nodes;
};

// Host-only discovery on one resolved, unfiltered graph, without changing annotations or placement.
// IDs are supplied by the caller and scoped to this graph. Rediscover membership after graph rewrites.
// Rejects invalid membership, grouping-induced cycles, and ambiguous region order. Failure preserves result.
Status DiscoverGraphRegions(const GraphViewer& graph, gsl::span<const GraphRegion> regions,
                            GraphRegionDiscoveryResult& result);

// Uses exact, nonempty layer_ann values as region IDs. Unannotated nodes remain explicitly ungrouped.
Status DiscoverGraphRegionsFromAnnotations(const GraphViewer& graph, GraphRegionDiscoveryResult& result);

}  // namespace onnxruntime
