// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/node_index_info.h"

#include "core/framework/ort_value_name_idx_map.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/node_arg.h"

namespace onnxruntime {

// if we have a full GraphViewer, assume the min node index is 0
NodeIndexInfo::NodeIndexInfo(const GraphViewer& graph_viewer, const OrtValueNameIdxMap& ort_value_idx_map)
    : min_node_index_(0), max_mlvalue_idx_(ort_value_idx_map.MaxIdx()) {
  Init(graph_viewer.Nodes(), graph_viewer.MaxNodeIndex(), ort_value_idx_map);
}

NodeIndexInfo::NodeIndexInfo(const GraphNodes& nodes, const OrtValueNameIdxMap& ort_value_idx_map)
    : max_mlvalue_idx_{ort_value_idx_map.MaxIdx()} {
  Init(nodes, 0, ort_value_idx_map);
}

NodeIndexInfo::NodeIndexInfo(const std::vector<const Node*>& nodes, const OrtValueNameIdxMap& ort_value_idx_map)
    : max_mlvalue_idx_{ort_value_idx_map.MaxIdx()} {
  Init(ValidNodes<const std::vector<const Node*>>(nodes), 0, ort_value_idx_map);
}

template <typename TValidNodes>
static void FindMinAndMaxNodeIndex(const TValidNodes& nodes, NodeIndex& min, NodeIndex& max) {
  min = std::numeric_limits<NodeIndex>::max();
  max = 0;
  std::for_each(nodes.cbegin(), nodes.cend(), [&min, &max](const Node& node) {
    auto idx = node.Index();
    if (idx > max) max = idx;
    //NodeIndex is size_t type
    if (idx < min) min = idx;
  });

  // match GraphViewer::MaxNodeIndex() which returns nodes_.size(), so is actually the max used value + 1.
  // if we didn't do this, we'd have to add 1 when calling node_offsets_.resize, which would give max used value + 2
  // if Init was called with max_node_index from GraphViewer::MaxNodeIndex(), which would create an extra invalid entry
  // in node_offsets_ at the end.
  max += 1;
}

template <typename TValidNodes>
void NodeIndexInfo::Init(const TValidNodes& nodes, NodeIndex max_node_index,
                         const OrtValueNameIdxMap& ort_value_idx_map) {
  if (nodes.empty()) {
    // fairly stupid edge case to handle unit test for Constant. the Constant node becomes an initializer, leaving
    // the graph with no nodes.
    return;
  }

  std::size_t total_def_count{};
  const bool include_missing_optional_defs = true;

  if (max_node_index == 0) {
    FindMinAndMaxNodeIndex(nodes, min_node_index_, max_node_index);
  }

  for (const auto& node : nodes) {
    node.ForEachDef(
        [&](const onnxruntime::NodeArg& /*arg*/, bool /*is_input*/) {
          ++total_def_count;
        },
        include_missing_optional_defs);
  }

  // init all to kInvalidEntry
  node_offsets_.resize(GetNodeOffsetsIndex(max_node_index), kInvalidEntry);
  node_values_.resize(total_def_count, kInvalidEntry);

  node_offsets_size_ = node_offsets_.size();
  node_values_size_ = node_values_.size();

  int cur_idx = 0;

  for (auto& node : nodes) {
    node_offsets_[GetNodeOffsetsIndex(node.Index())] = cur_idx;

    node.ForEachDef(
        [&](const onnxruntime::NodeArg& node_arg, bool /*is_input*/) {
          auto& name = node_arg.Name();
          if (node_arg.Exists()) {
            int index;
            Status status = ort_value_idx_map.GetIdx(name, index);
            ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
            node_values_[cur_idx] = index;
          }
          // else it's a missing optional input or output so leave the -1

          ++cur_idx;
        },
        include_missing_optional_defs);
  }
}

}  // namespace onnxruntime
