// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/common/common.h"
#include "core/framework/ml_value.h"

namespace onnxruntime {
class GraphViewer;
class MLValueNameIdxMap;

class NodeIndexInfo final {
 public:
  NodeIndexInfo(const GraphViewer& graph_viewer, const MLValueNameIdxMap& mlvalue_idx_map);

  enum { kInvalidEntry = -1 };

  // Index to the first argument of the given Node.
  // The Node will have (num inputs + num implicit inputs + num outputs) entries, in that order, starting at the
  // offset that is returned. Use the offset in calls to GetMLValueIndex.
  // Returns kInvalidEntry if the Node with the given node_index did not exist when the NodeIndexInfo was created.
  int GetNodeOffset(onnxruntime::NodeIndex node_index) const {
    ORT_ENFORCE(node_index < node_offsets_.size());
    return node_offsets_[node_index];
  }

  // Get the mlvalue index value.
  // Returns kInvalidEntry for optional inputs/outputs that do not exist in this graph.
  int GetMLValueIndex(int offset) const {
    ORT_ENFORCE(offset >= 0 && static_cast<size_t>(offset) < node_values_.size());
    return node_values_[offset];
  }

  int GetMaxMLValueIdx() const { return max_mlvalue_idx_; }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NodeIndexInfo);

  // This vector contains the indices from the MLValueNameIdxMap in the SessionState for each Node's input/outputs.
  // Order is node inputs, implicit inputs, outputs.
  std::vector<int> node_values_;

  // The entry at node_offset_[Node::Index()] contains the index in node_values_ where the information for the Node
  // begins.
  std::vector<int> node_offsets_;

  const int max_mlvalue_idx_;
};
}  // namespace onnxruntime
