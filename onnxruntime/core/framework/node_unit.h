// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// QDQ models require graph modification at runtime, so we know this infrastructure is not used in a minimal build
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include <string>
#include <optional>
#include <vector>

#include "core/graph/basic_types.h"
#include "core/graph/graph.h"

namespace onnxruntime {

class GraphViewer;
class Node;
class NodeArg;
class Path;

namespace QDQ {
// Struct to represent a DequantizeLinear -> Op -> QuantizeLinear node group
struct NodeGroup {
  std::vector<NodeIndex> dq_nodes;
  std::vector<NodeIndex> q_nodes;
  NodeIndex target_node;

  // Validator to check if the set of nodes can form a valid QDQ NodeGroup.
  // Checks target node is only consumer of each DQ, and that the outputs remain valid if the QDQ node group was to
  // be converted into a single node with a quantized operator.
  static Status CanCreateNodeGroup(const GraphViewer& graph_viewer,
                                   const Node& target_node,
                                   gsl::span<const Node* const> dq_nodes,
                                   gsl::span<const Node* const> q_nodes);
};
}  // namespace QDQ

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

  const std::vector<NodeUnitIODef>& Inputs() const noexcept { return inputs_; }
  const std::vector<NodeUnitIODef>& Outputs() const noexcept { return outputs_; }

  const std::string& Domain() const noexcept;
  const std::string& OpType() const noexcept;
  const std::string& Name() const noexcept;
  int SinceVersion() const noexcept;
  NodeIndex Index() const noexcept;
  const Path& ModelPath() const noexcept;
  ProviderType GetExecutionProviderType() const noexcept;

  const Node& GetNode() const noexcept { return target_node_; }
  const std::vector<const Node*>& GetDQNodes() const noexcept { return dq_nodes_; }
  const std::vector<const Node*>& GetQNodes() const noexcept { return q_nodes_; }
  std::vector<const Node*> GetAllNodesInGroup() const noexcept;

  /// Number of input edges to the logical node. For a QDQ node this is the count of input edges to the DQ nodes
  /// plus any other edges to the target node for inputs that are not via a DQ node.
  size_t InputEdgeCount() const { return input_edge_count_; }

  // output edges. src index is for outputs of the target node. dest index and node is for consumer of node unit
  // output. any Q nodes are hidden.
  Node::EdgeConstIterator OutputEdgesBegin() const;
  Node::EdgeConstIterator OutputEdgesEnd() const;

 private:
  // Initialization for a NodeUnit that contains a single node
  void InitForSingleNode();

  const std::vector<const Node*> dq_nodes_;  // dq nodes for this NodeUnit, not necessarily all inputs
  const Node& target_node_;
  const std::vector<const Node*> q_nodes_;  // q-nodes for this NodeUnit. not necessarily all outputs
  const Type type_;

  std::vector<NodeUnitIODef> inputs_;
  std::vector<NodeUnitIODef> outputs_;

  size_t input_edge_count_;  // total number of input edges

  // output edges, hiding any Q nodes involved. src_idx will be value from target node. only used for QDQ node group.
  Node::EdgeSet output_edges_;
};

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
