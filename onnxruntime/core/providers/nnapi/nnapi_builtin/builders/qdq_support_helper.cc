// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <string>
#include <vector>
#include <optional>

#include <core/graph/graph_viewer.h>
#include <core/providers/common.h>

#include "qdq_support_helper.h"

namespace onnxruntime {
namespace nnapi {

using std::string;
using std::vector;

using OpVersionsMap = std::unordered_map<std::string, std::vector<ONNX_NAMESPACE::OperatorSetVersion>>;

QDQSupportHelper::QDQSupportHelper(Selectors&& selectors)
    : selectors_{std::move(selectors)} {
  for (const auto& entry : selectors_.SelectorsSet()) {
    for (const auto& op_info : entry->op_versions_map) {
      bool inserted = op_type_to_selectors_map_.insert({op_info.first, &*entry}).second;

      ORT_ENFORCE(inserted, "Multiple entries for operator is not supported. OpType=", op_info.first);
    }
  }
}

void Selectors::RegisterSelector(const Selector::OpVersionsMap& ops_and_versions_in,
                                 std::unique_ptr<QDQ::BaseSelector> selector_in) {
  auto entry = std::make_unique<Selector>(
      ops_and_versions_in,
      std::move(selector_in));

  ORT_IGNORE_RETURN_VALUE(selectors_set_.insert(std::move(entry)));
}

std::optional<QDQ::NodeGroup> QDQSupportHelper::Match(const GraphViewer& graph_viewer, const Node& node) const {
  std::optional<QDQ::NodeGroup> qdq_node_group;

  if (node.Domain() != kOnnxDomain) {
    return qdq_node_group;
  }

  auto op_rule = op_type_to_selectors_map_.find(node.OpType());
  if (op_rule == op_type_to_selectors_map_.cend()) {
    return qdq_node_group;
  }

  const auto& selector = *op_rule->second;

  // check the supported versions if specified
  const auto& versions = selector.op_versions_map.find(node.OpType())->second;
  if (!versions.empty()) {
    if (std::find(versions.cbegin(), versions.cend(), node.SinceVersion()) == versions.cend()) {
      LOGS_DEFAULT(VERBOSE) << "Specified op type version is not supported";
      return qdq_node_group;
    }
  }

  qdq_node_group = selector.selector->GetQDQSelection(graph_viewer, node);
  if (!qdq_node_group.has_value()) {
    LOGS_DEFAULT(VERBOSE) << "No matched qdq selection returned";
    return qdq_node_group;
  }

  LOGS_DEFAULT(VERBOSE) << "QDQ Node Group found: " << node.OpType()
                        << " with matched target node's name: " << node.Name() << "\n";

  return qdq_node_group;
}

bool QDQSupportHelper::IsNodeInQDQGroup(const Node& node) const {
  return nodes_in_qdq_group.find(&node) != nodes_in_qdq_group.end();
}

void QDQSupportHelper::GetQDQNodeGroup(const GraphViewer& graph_viewer, const Node& node) {
  auto qdq_node_group = Match(graph_viewer, node);
  QDQ::NodeGroupNonIndex qdq_node_group_nonindex;

  // Obtain the qdq node group from the qdq node index group
  if (qdq_node_group.has_value()) {
    qdq_node_group_nonindex.target_node = &node;
    nodes_in_qdq_group.insert(&node);

    for (auto idx : qdq_node_group->dq_nodes) {
      const auto* dq_node = graph_viewer.GetNode(idx);
      qdq_node_group_nonindex.dq_nodes.push_back(dq_node);
      nodes_in_qdq_group.insert(dq_node);
    }

    for (auto idx : qdq_node_group->q_nodes) {
      const auto* q_node = graph_viewer.GetNode(idx);
      qdq_node_group_nonindex.q_nodes.push_back(q_node);
      nodes_in_qdq_group.insert(q_node);
    }
  }

  auto it = target_node_to_qdq_group_.find(&node);
  if (it != target_node_to_qdq_group_.end()) {
    it->second = qdq_node_group_nonindex;
  } else {
    target_node_to_qdq_group_.emplace(&node, qdq_node_group_nonindex);
  }
}

void QDQSupportHelper::GetQDQNodeGroups(const GraphViewer& graph_viewer) {
  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    const auto* node = graph_viewer.GetNode(index);
    GetQDQNodeGroup(graph_viewer, *node);
  }
}

/* Selector Rules Related */
void ConvQDQRules(Selectors& qdq_selectors) {
  // 4 or 5 Nodes. 0=DQ X, 1=DQ W, 2=DQ B (optional), 3=Conv, 4=Q
  // Handle the DQ input for the Bias being optional.

  std::unique_ptr<QDQ::BaseSelector> selector(new QDQ::ConvSelector());

  qdq_selectors.RegisterSelector(OpVersionsMap{{"Conv", {}}},
                                 std::move(selector));
}

Selectors CreateSelectors() {
  Selectors qdq_selectors;

  ConvQDQRules(qdq_selectors);

  return qdq_selectors;
}

}  // namespace nnapi
}  // namespace onnxruntime
