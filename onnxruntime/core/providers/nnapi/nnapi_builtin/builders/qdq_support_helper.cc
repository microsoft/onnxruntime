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

QDQSupportHelper::QDQSupportHelper(Selectors&& selectors, const GraphViewer& graph_viewer)
    : selectors_{std::move(selectors)},
      graph_viewer_{graph_viewer} {
  for (const auto& entry : selectors_.SelectorsSet()) {
    for (const auto& op_info : entry->op_versions_map) {
      bool inserted = op_type_to_selectors_map_.insert({op_info.first, &*entry}).second;
      ORT_ENFORCE(inserted, "Multiple entries for operator is not supported. OpType=", op_info.first);
    }
  }

  for (auto index : graph_viewer_.GetNodesInTopologicalOrder()) {
    const auto* node = graph_viewer_.GetNode(index);
    SetQDQNodeGroup(*node);
  }

  target_nodes_.reserve(target_node_to_qdq_group_.size());
  for (const auto kv : target_node_to_qdq_group_) {
    target_nodes_.push_back(kv.first);
  }
}

void Selectors::RegisterSelector(const Selector::OpVersionsMap& ops_and_versions_in,
                                 std::unique_ptr<QDQ::BaseSelector> selector_in) {
  auto entry = std::make_unique<Selector>(
      ops_and_versions_in,
      std::move(selector_in));

  ORT_IGNORE_RETURN_VALUE(selectors_set_.insert(std::move(entry)));
}

bool QDQSupportHelper::IsNodeInQDQGroup(const Node& node) const {
  return nodes_in_qdq_group_.find(&node) != nodes_in_qdq_group_.end();
}

bool QDQSupportHelper::IsNodeTargetNode(const Node& node) const {
  return std::find(target_nodes_.begin(), target_nodes_.end(), &node) != target_nodes_.end();
}

const QDQ::NodeGroup QDQSupportHelper::GetQDQNodeGroupWithTargetNode(const Node& target_node) const {
  QDQ::NodeGroup qdq_node_group;
  if (target_node_to_qdq_group_.find(&target_node) != target_node_to_qdq_group_.end()) {
    qdq_node_group = target_node_to_qdq_group_.find(&target_node)->second;
  }
  return qdq_node_group;
}

std::optional<QDQ::NodeGroupIndices> QDQSupportHelper::Match(const Node& node) const {
  std::optional<QDQ::NodeGroupIndices> qdq_node_group_indices;

  if (node.Domain() != kOnnxDomain) {
    return qdq_node_group_indices;
  }

  auto op_rule = op_type_to_selectors_map_.find(node.OpType());
  if (op_rule == op_type_to_selectors_map_.cend()) {
    return qdq_node_group_indices;
  }

  const auto& selector = *op_rule->second;

  // check the supported versions if specified
  const auto& versions = selector.op_versions_map.find(node.OpType())->second;
  if (!versions.empty()) {
    if (std::find(versions.cbegin(), versions.cend(), node.SinceVersion()) == versions.cend()) {
      LOGS_DEFAULT(VERBOSE) << "Specified op type version is not supported";
      return qdq_node_group_indices;
    }
  }

  qdq_node_group_indices = selector.selector->GetQDQSelection(graph_viewer_, node);
  if (!qdq_node_group_indices.has_value()) {
    LOGS_DEFAULT(VERBOSE) << "No matched qdq selection returned";
    return qdq_node_group_indices;
  }

  LOGS_DEFAULT(VERBOSE) << "QDQ Node Group found: " << node.OpType()
                        << " with matched target node's name: " << node.Name() << "\n";

  return qdq_node_group_indices;
}

void QDQSupportHelper::SetQDQNodeGroup(const Node& node) {
  auto qdq_node_group_indices = Match(node);
  QDQ::NodeGroup qdq_node_group;

  // Obtain the qdq node group from the qdq node index group
  if (qdq_node_group_indices.has_value()) {
    qdq_node_group.target_node = &node;
    nodes_in_qdq_group_.insert(&node);

    for (auto idx : qdq_node_group_indices->dq_nodes) {
      const auto* dq_node = graph_viewer_.GetNode(idx);
      qdq_node_group.dq_nodes.push_back(dq_node);
      nodes_in_qdq_group_.insert(dq_node);
    }

    for (auto idx : qdq_node_group_indices->q_nodes) {
      const auto* q_node = graph_viewer_.GetNode(idx);
      qdq_node_group.q_nodes.push_back(q_node);
      nodes_in_qdq_group_.insert(q_node);
    }

    target_node_to_qdq_group_[&node] = std::move(qdq_node_group);
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
