// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/graph/basic_types.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksTypes.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/selectors_actions/helpers.h"

namespace onnxruntime {

class GraphViewer;
class Node;

namespace nnapi {

struct Selector {
  using OpVersionsMap = std::unordered_map<std::string, std::vector<ONNX_NAMESPACE::OperatorSetVersion>>;

  Selector(const OpVersionsMap& ops_and_versions_in,
           std::unique_ptr<QDQ::BaseSelector> selector_in)
      : op_versions_map{ops_and_versions_in},
        selector{std::move(selector_in)} {}

  OpVersionsMap op_versions_map;
  std::unique_ptr<QDQ::BaseSelector> selector;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(Selector);
};

class Selectors {
 public:
  Selectors() = default;

  Selectors(Selectors&& rhs) noexcept
      : selectors_set_{std::move(rhs.selectors_set_)} {}

  void RegisterSelector(const Selector::OpVersionsMap& ops_and_versions_in,
                        std::unique_ptr<QDQ::BaseSelector> selector_in);

  const std::unordered_set<std::unique_ptr<Selector>>& SelectorsSet() const {
    return selectors_set_;
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(Selectors);

 private:
  std::unordered_set<std::unique_ptr<Selector>> selectors_set_;
};

class QDQSupportHelper {
 public:
  QDQSupportHelper(Selectors&& selectors);

  std::optional<QDQ::NodeGroup> Match(const GraphViewer& graph_viewer, const Node& node) const;

  bool IsNodeInQDQGroup(const Node& node);

  std::optional<QDQ::NodeGroup> GetQDQNodeGroup(const onnxruntime::GraphViewer& graph_viewer, const Node& node);

  void GetQDQNodeGroups(const onnxruntime::GraphViewer& graph_viewer);

  Selectors selectors_;

  std::vector<std::optional<QDQ::NodeGroup>> qdq_node_groups_;

  std::unordered_map<const Node*, std::optional<QDQ::NodeGroup>> target_node_to_qdq_group_;

  std::unordered_map<std::string, const Selector*> op_type_to_selectors_map_;

  std::vector<const Node*> dq_nodes_in_qdq_selection;
};

/* Selector Rules Related */
void ConvQDQRules(Selectors& qdq_selectors);

Selectors CreateSelectors();

}  // namespace nnapi
}  // namespace onnxruntime
