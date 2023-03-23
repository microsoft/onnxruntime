// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>
#include <string>

#include "core/common/gsl.h"
#include "core/common/inlined_containers.h"
#include "core/graph/basic_types.h"
#include "core/graph/graph.h"

namespace onnxruntime {

class GraphViewer;
class Node;
class NodeArg;
class Path;

namespace QDQ {
struct NodeGroup;
}

// Definition of one input or output
// If the optional quant_param is present, then this is a quantized input,
// otherwise this is a regular input
struct NodeUnitIODef {
  // The quantization parameter, scale is manadatory, and zero_point is optional
  struct QuantParam {
    const NodeArg& scale;
    const NodeArg* zero_point{nullptr};
  };

  const NodeArg& node_arg;
  const std::optional<QuantParam> quant_param;
};

/**
@class NodeUnit
Class to represent a single node or a QDQ group of nodes, which will be used as a single unit.
*/
class NodeUnit {
 public:
  // NodeUnit type
  enum class Type : uint8_t {
    SingleNode,  // The NodeUnit contains a single node
    QDQGroup,    // The NodeUnit contain a QDQ group of nodes, such as "DQ->Sigmoid->Q"
  };

 public:
  explicit NodeUnit(const Node& node);
  explicit NodeUnit(const GraphViewer& graph_viewer, const QDQ::NodeGroup& node_group);

  Type UnitType() const noexcept { return type_; }

  gsl::span<const NodeUnitIODef> Inputs() const noexcept { return inputs_; }
  gsl::span<const NodeUnitIODef> Outputs() const noexcept { return outputs_; }

  const std::string& Domain() const noexcept;
  const std::string& OpType() const noexcept;
  const std::string& Name() const noexcept;
  int SinceVersion() const noexcept;
  NodeIndex Index() const noexcept;
  const Path& ModelPath() const noexcept;
  ProviderType GetExecutionProviderType() const noexcept;

  const Node& GetNode() const noexcept { return target_node_; }
  gsl::span<const gsl::not_null<const Node*>> GetDQNodes() const noexcept { return dq_nodes_; }
  gsl::span<const gsl::not_null<const Node*>> GetQNodes() const noexcept { return q_nodes_; }
  InlinedVector<gsl::not_null<const Node*>> GetAllNodesInGroup() const noexcept;

  Node::EdgeConstIterator OutputEdgesBegin(size_t index) const;
  Node::EdgeConstIterator OutputEdgesEnd(size_t index) const;

 private:
  InlinedVector<gsl::not_null<const Node*>> q_nodes_;   // q-nodes for this NodeUnit
  InlinedVector<gsl::not_null<const Node*>> dq_nodes_;  // dq nodes for this NodeUnit, not all inputs
  const Node& target_node_;
  const Type type_;

  InlinedVector<NodeUnitIODef> inputs_;
  InlinedVector<NodeUnitIODef> outputs_;

  // Initializing for a single Node
  void InitForSingleNode();
};

using NodeUnitHolder = InlinedVector<std::unique_ptr<NodeUnit>>;
using NodeToNodeUnitMap = InlinedHashMap<gsl::not_null<const Node*>, gsl::not_null<const NodeUnit*>>;
// Get all the nodes in the given graph_viewer as NodeUnits (SingleNode or QDQGroup)
// And return a map to quick query the NodeUnit which contains the given Node,
// Note, the values of the NodeToNodeUnitMap are owned by the NodeUnitHolder.
std::pair<NodeUnitHolder, NodeToNodeUnitMap> GetAllNodeUnits(const GraphViewer& graph_viewer);

}  // namespace onnxruntime
