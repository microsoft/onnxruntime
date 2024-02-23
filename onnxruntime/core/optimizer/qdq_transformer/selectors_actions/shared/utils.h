// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/common/common.h"
#include "core/common/gsl.h"
#include "core/common/inlined_containers.h"
#include "core/graph/basic_types.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/graph/onnx_protobuf.h"
#endif

namespace onnxruntime {

class GraphViewer;
class Node;

namespace QDQ {

struct NodeGroup;
class NodeGroupSelector;

// struct that provides a join between selector and op versions supported
struct OpVersionsAndSelector {
  using OpVersionsMap = std::unordered_map<std::string, std::vector<ONNX_NAMESPACE::OperatorSetVersion>>;

  OpVersionsAndSelector(const OpVersionsMap& ops_and_versions_in,
                        std::unique_ptr<NodeGroupSelector> selector_in)
      : op_versions_map{ops_and_versions_in},
        selector{std::move(selector_in)} {}

  OpVersionsMap op_versions_map;
  std::unique_ptr<NodeGroupSelector> selector;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OpVersionsAndSelector);
};

// class that manages a set of node group selectors
class Selectors {
 public:
  Selectors() = default;

  // register a selector for the specified ops.
  void RegisterSelector(const OpVersionsAndSelector::OpVersionsMap& ops_and_versions_in,
                        std::unique_ptr<NodeGroupSelector> selector_in);

  const InlinedHashSet<std::unique_ptr<OpVersionsAndSelector>>& SelectorsSet() const {
    return selectors_set_;
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Selectors);

 private:
  InlinedHashSet<std::unique_ptr<OpVersionsAndSelector>> selectors_set_;
};

// class that manages qdq node group selections
class SelectorManager {
 public:
  SelectorManager();

  // Methods that finds and returns a vector of QDQ::NodeGroup in a given graph
  // Can be used in QDQ support in different EPs
  std::vector<NodeGroup> GetQDQSelections(const GraphViewer& graph_viewer) const;

 private:
  Selectors qdq_selectors_;

  std::unordered_map<std::string, const OpVersionsAndSelector*> op_type_to_selectors_map_;

  void InitializeSelectorsMap();

  void CreateSelectors();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SelectorManager);
};

// Checks whether the provided DQ nodes are valid for forming a QDQ node group with the provided target node.
// Returns successful status if so, failed status with reason otherwise.
Status ValidateNodeGroupDQNodes(const GraphViewer& graph_viewer,
                                const Node& target_node,
                                gsl::span<const Node* const> dq_nodes);

}  // namespace QDQ
}  // namespace onnxruntime
