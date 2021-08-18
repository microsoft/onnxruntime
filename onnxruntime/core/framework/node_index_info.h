// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/common/common.h"
#include "core/framework/ort_value.h"
#include "core/graph/basic_types.h"

namespace onnxruntime {
class GraphNodes;
class GraphViewer;
class OrtValueNameIdxMap;
class Node;

class NodeIndexInfo final {
 public:
  // construct from a GraphViewer.
  NodeIndexInfo(const GraphViewer& graph_viewer, const OrtValueNameIdxMap& ort_value_idx_map);

  // construct from a subset of nodes. The min and max NodeIndex values will be calculated by iterating 'nodes'.
  NodeIndexInfo(const GraphNodes& nodes, const OrtValueNameIdxMap& ort_value_idx_map);
  NodeIndexInfo(const std::vector<const Node*>& nodes, const OrtValueNameIdxMap& ort_value_idx_map);

  enum { kInvalidEntry = -1 };

  // Index to the first argument of the given Node.
  // The Node will have (num inputs + num implicit inputs + num outputs) entries, in that order, starting at the
  // offset that is returned. Use the offset in calls to GetMLValueIndex.
  // Returns kInvalidEntry if the Node with the given node_index did not exist when the NodeIndexInfo was created.
  int GetNodeOffset(NodeIndex node_index) const {
    auto node_offsets_index = GetNodeOffsetsIndex(node_index);
    ORT_ENFORCE(node_offsets_index < node_offsets_size_);
    return node_offsets_[node_offsets_index];
  }

  // Get the ort_value index value.
  // Returns kInvalidEntry for optional inputs/outputs that do not exist in this graph.
  int GetMLValueIndex(int offset) const {
    ORT_ENFORCE(offset >= 0 && static_cast<size_t>(offset) < node_values_size_);
    return node_values_[offset];
  }

  int GetMaxMLValueIdx() const { return max_mlvalue_idx_; }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NodeIndexInfo);

  template <typename TValidNodes>
  void Init(const TValidNodes& nodes, NodeIndex max_node_index, const OrtValueNameIdxMap& ort_value_idx_map);

  // This vector contains the indices from the OrtValueNameIdxMap in the SessionState for each Node's input/outputs.
  // Order is node inputs, implicit inputs, outputs.
  std::vector<int> node_values_;

  // the minimum NodeIndex. we use this to minimize the size of node_offsets_.
  NodeIndex min_node_index_ = 0;

  // The entry at node_offsets_[GetNodeOffsetsIndex(Node::Index())] contains the index in node_values_
  // where the information for the Node begins.
  size_t GetNodeOffsetsIndex(NodeIndex node_index) const { return node_index - min_node_index_; }
  std::vector<int> node_offsets_;

  const int max_mlvalue_idx_;

  // perf optimization to avoid calls to size() on node_values_ and node_offsets_ as they don't change
  size_t node_values_size_;
  size_t node_offsets_size_;
};
}  // namespace onnxruntime
