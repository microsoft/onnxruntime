// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/optimizer/graph_transformer.h"
#include "core/framework/node_unit.h"

namespace onnxruntime {

/**
 * Remove Q and DQ node pairs in the graph.
 * e.g.
 * node_1 -> Q -> DQ -> node_2         => node_1 -> node_2
 * graph's input -> Q -> DQ -> node_1  => graph's input -> node_1
 * node_1 -> Q -> DQ -> graph's output => node_1 -> graph's output
 * (Note: The Q and DQ are not in the same node unit)
 *
 * This optimizer selects all the Q and DQ node pairs regardless of their data type in the graph to be removed.
 * EP/caller can provide the Q and DQ nodes pairs with specific data type in node_index_set which will be taken by this optimizer ctor.
 */
class QDQStripping : public GraphTransformer {
 public:
  QDQStripping(InlinedHashSet<NodeIndex> node_index_set) noexcept
      : GraphTransformer("QDQStripping", {}),
        node_index_set_(node_index_set) {
  }

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
  Status FindCandidateQDQToRemove(Graph& graph, const NodeUnit& node_unit, std::unordered_map<Node*, Node*>& candidate_dq_to_q_map) const;
  bool AllowStripping(const Node& node) const;
  InlinedHashSet<NodeIndex> node_index_set_;  // The nodes EP/caller provides to remove
};

}  // namespace onnxruntime
