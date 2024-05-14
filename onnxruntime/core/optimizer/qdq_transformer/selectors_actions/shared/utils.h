// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/common/common.h"
#include "core/common/gsl.h"
#include "core/common/inlined_containers.h"
#include "core/framework/node_unit.h"
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

// Get all the nodes in the given graph_viewer as NodeUnits (SingleNode or QDQGroup)
// And return a map to quick query the NodeUnit which contains the given Node,
// Note, the value of the map is owned by the vector of std::unique_ptr<NodeUnit>
//
// TODO: The overall QDQ setup needs refactoring to separate out generic functionality from optimizer specific
// functionality.
// We currently have a bit of a mess with generic things like this to get all the node units being in the optimizer
// library whereas it should be able to be used by an EP with no dependency on optimizers.
std::pair<std::vector<std::unique_ptr<NodeUnit>>, std::unordered_map<const Node*, const NodeUnit*>>
GetAllNodeUnits(const GraphViewer& graph_viewer);

}  // namespace QDQ
}  // namespace onnxruntime
