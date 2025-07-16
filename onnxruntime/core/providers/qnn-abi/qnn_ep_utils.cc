// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "qnn_ep_utils.h"
// #include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include <iostream>
#include <string>

namespace onnxruntime {
class QnnEp;


// Implementation of GetQDQNodeUnits for OrtGraph
std::pair<std::vector<const OrtNode*>, std::unordered_map<const OrtNode*, const OrtNode*>>
GetAllNodeUnits(const OrtEp* this_ptr, const OrtGraph* graph, const logging::Logger& logger) {
  logger;
  const auto* ep = static_cast<const QnnEp*>(this_ptr);
  std::vector<const OrtNode*> node_holder;
  std::unordered_map<const OrtNode*, const OrtNode*> node_map;

  // Get all nodes from the graph
  OrtArrayOfConstObjects* nodes = nullptr;
  ep->ort_api.Graph_GetNodes(graph, &nodes);

  size_t num_nodes = 0;
  ep->ort_api.ArrayOfConstObjects_GetSize(nodes, &num_nodes);

  const void* const* node_data = nullptr;
  ep->ort_api.ArrayOfConstObjects_GetData(nodes, &node_data);

  const auto add_node_unit_to_map = [&](const std::vector<int>& node_indices) {
    for (auto node_idx : node_indices) {
    const OrtNode* node = static_cast<const OrtNode*>(node_data[node_idx]);
    node_map.insert({node, node});
    }
  };

  // Get QDQ NodeUnits first
  // QDQ::SelectorManager selector_mgr;

  // const auto qdq_selections = GetQDQSelections(graph, logger);
  // for (const auto& qdq_selection : qdq_selections) {
  //   // auto qdq_unit = std::make_unique<OrtNodeUnit>(graph, qdq_selection);

  //   // Fill the node to node_unit map for all nodes in the QDQ Group
  //   add_node_unit_to_map(qdq_selection.dq_nodes);
  //   add_node_unit_to_map(qdq_selection.q_nodes);
  //   add_node_unit_to_map({qdq_selection.target_node});
  //   // if (qdq_selection.redundant_clip_node.has_value()) {
  //   //   add_node_unit_to_map({qdq_selection.redundant_clip_node.value()}, qdq_unit.get());
  //   // }

  //   // node_unit_holder.push_back(std::move(qdq_unit));
  // }

  for (size_t node_idx = 0; node_idx < num_nodes; ++node_idx) {
    const OrtNode* node = static_cast<const OrtNode*>(node_data[node_idx]);
    node_map.insert({node, node});
    node_holder.push_back(node);
  }

  return std::make_pair(std::move(node_holder), std::move(node_map));
}


}  // namespace onnxruntime
