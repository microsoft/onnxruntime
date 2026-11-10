// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/region_discovery.h"

#include <optional>
#include <utility>

#include "core/graph/graph_viewer.h"

namespace onnxruntime {

Status DiscoverGraphRegions(const GraphViewer& graph, gsl::span<const GraphRegion> regions,
                            GraphRegionDiscoveryResult& result) {
  if (graph.GetFilterInfo() != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Region discovery requires an unfiltered graph view.");
  }

  InlinedHashMap<NodeIndex, size_t> node_to_vertex;
  InlinedHashSet<std::string> region_ids;
  InlinedVector<GraphRegion> normalized_regions;
  normalized_regions.reserve(regions.size());
  for (size_t region_index = 0; region_index < regions.size(); ++region_index) {
    const auto& region = regions[region_index];
    if (region.id.empty() || !region_ids.insert(region.id).second) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Region IDs must be nonempty and unique: '", region.id, "'.");
    }
    if (region.nodes.empty()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Region '", region.id, "' has no members.");
    }
    for (NodeIndex node_index : region.nodes) {
      if (node_index >= static_cast<NodeIndex>(graph.MaxNodeIndex()) || graph.GetNode(node_index) == nullptr) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Region '", region.id, "' references missing node ", node_index, ".");
      }
      const auto [existing, inserted] = node_to_vertex.emplace(node_index, region_index);
      if (!inserted) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Node ", node_index,
                               " has duplicate or overlapping membership in regions '",
                               regions[existing->second].id, "' and '", region.id, "'.");
      }
    }
    normalized_regions.push_back({region.id, {}});
  }

  GraphRegionDiscoveryResult discovered;
  size_t vertex_count = regions.size();
  for (NodeIndex node_index : graph.GetNodesInTopologicalOrder()) {
    const auto owner = node_to_vertex.find(node_index);
    if (owner != node_to_vertex.end()) {
      normalized_regions[owner->second].nodes.push_back(node_index);
    } else {
      node_to_vertex.emplace(node_index, vertex_count++);
      discovered.ungrouped_nodes.push_back(node_index);
    }
  }

  // Retain every ungrouped node as a vertex: A -> ungrouped -> A must still expose a contraction cycle.
  InlinedVector<InlinedHashSet<size_t>> successors(vertex_count);
  InlinedVector<size_t> in_degree(vertex_count, 0);
  for (const auto& node : graph.Nodes()) {
    const size_t source = node_to_vertex.at(node.Index());
    for (auto output = node.OutputNodesBegin(); output != node.OutputNodesEnd(); ++output) {
      const size_t target = node_to_vertex.at(output->Index());
      if (source != target && successors[source].insert(target).second) {
        ++in_degree[target];
      }
    }
  }

  InlinedVector<size_t> ready_regions;
  InlinedVector<size_t> ready_ungrouped;
  const auto enqueue = [&](size_t vertex) {
    if (vertex < regions.size()) {
      ready_regions.push_back(vertex);
    } else {
      ready_ungrouped.push_back(vertex);
    }
  };
  for (size_t vertex = 0; vertex < vertex_count; ++vertex) {
    if (in_degree[vertex] == 0) {
      enqueue(vertex);
    }
  }

  size_t visited = 0;
  const auto consume = [&](size_t vertex) {
    ++visited;
    for (size_t successor : successors[vertex]) {
      if (--in_degree[successor] == 0) {
        enqueue(successor);
      }
    }
  };
  std::optional<std::pair<size_t, size_t>> ambiguous_regions;
  while (!ready_regions.empty() || !ready_ungrouped.empty()) {
    // Independent ungrouped work must not hide a second ready region or impose an arbitrary block order.
    while (!ready_ungrouped.empty()) {
      const size_t vertex = ready_ungrouped.back();
      ready_ungrouped.pop_back();
      consume(vertex);
    }
    if (ready_regions.empty()) {
      break;
    }
    if (ready_regions.size() > 1 && !ambiguous_regions.has_value()) {
      ambiguous_regions.emplace(ready_regions[0], ready_regions[1]);
    }
    const size_t vertex = ready_regions.back();
    ready_regions.pop_back();
    discovered.regions.push_back(std::move(normalized_regions[vertex]));
    consume(vertex);
  }

  if (visited != vertex_count) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Contracting the proposed graph regions creates a dependency cycle.");
  }
  if (ambiguous_regions.has_value()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Ambiguous region dependency order between '",
                           regions[ambiguous_regions->first].id, "' and '", regions[ambiguous_regions->second].id, "'.");
  }

  result = std::move(discovered);
  return Status::OK();
}

Status DiscoverGraphRegionsFromAnnotations(const GraphViewer& graph, GraphRegionDiscoveryResult& result) {
  InlinedHashMap<std::string, size_t> annotation_to_region;
  InlinedVector<GraphRegion> regions;
  for (const auto& node : graph.Nodes()) {
    const auto& annotation = node.GetLayeringAnnotation();
    if (annotation.empty()) {
      continue;
    }
    const auto [entry, inserted] = annotation_to_region.emplace(annotation, regions.size());
    if (inserted) {
      regions.push_back({annotation, {}});
    }
    regions[entry->second].nodes.push_back(node.Index());
  }
  return DiscoverGraphRegions(graph, regions, result);
}

}  // namespace onnxruntime
