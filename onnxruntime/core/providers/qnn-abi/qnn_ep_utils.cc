// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "qnn_ep_utils.h"
// #include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include <iostream>
#include <string>

namespace onnxruntime {
class QnnEp;

// Implementation of GetQDQNodeUnits for OrtGraph
std::pair<std::vector<std::unique_ptr<OrtNodeUnit>>, std::unordered_map<const OrtNode*, const OrtNodeUnit*>>
GetAllOrtNodeUnits(OrtApi ort_api, const OrtGraph* graph, const logging::Logger& logger) {
  logger;
  std::vector<std::unique_ptr<OrtNodeUnit>> node_unit_holder;
  std::unordered_map<const OrtNode*, const OrtNodeUnit*> node_unit_map;

  // Get all nodes from the graph
  OrtArrayOfConstObjects* nodes = nullptr;
  ort_api.Graph_GetNodes(graph, &nodes);

  size_t num_nodes = 0;
  ort_api.ArrayOfConstObjects_GetSize(nodes, &num_nodes);

  const void* const* node_data = nullptr;
  ort_api.ArrayOfConstObjects_GetData(nodes, &node_data);

  // const auto add_node_unit_to_map = [&](const std::vector<int>& node_indices) {
  //   for (auto node_idx : node_indices) {
  //   const OrtNode* node = static_cast<const OrtNode*>(node_data[node_idx]);
  //   node_unit_map.insert({node, node});
  //   }
  // };

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

  // Get the left over single-node OrtNodeUnit.
  for (size_t node_idx = 0; node_idx < num_nodes; ++node_idx) {
    const OrtNode* node = static_cast<const OrtNode*>(node_data[node_idx]);

    // This is already part of a QDQ OrtNodeUnit.
    if (node_unit_map.find(node) != node_unit_map.cend())
      continue;

    auto node_unit = std::make_unique<OrtNodeUnit>(*node);
    node_unit_map[node] = node_unit.get();
    node_unit_holder.push_back(std::move(node_unit));
  }

  return std::make_pair(std::move(node_unit_holder), std::move(node_unit_map));
}

}  // namespace onnxruntime
