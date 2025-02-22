// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/optimizer/graph_transformer.h"
#include "core/framework/node_unit.h"

namespace onnxruntime {

/**
  *
  */
class QDQStripping : public GraphTransformer {
 public:
  QDQStripping() noexcept
      : GraphTransformer("QDQStripping", {}) {
  }

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
  Status FindCandidateQDQToRemove(Graph& graph, const NodeUnit& node_unit, std::unordered_map<Node*, Node*>& candidate_dq_to_q_map) const;
  bool AllowStripping(const Node& node) const;
  InlinedHashSet<NodeIndex> node_index_set_; // The nodes EP/caller provides to remove
};

}  // namespace onnxruntime
