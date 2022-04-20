// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <optional>
#include <vector>

#include "core/graph/basic_types.h"

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
  const std::vector<const Node*>& GetOutputNodes() const noexcept { return output_nodes_; }

 private:
  const std::vector<const Node*> output_nodes_;  // all the nodes producing outputs for this NodeUnit
  const Node& target_node_;
  const Type type_;

  std::vector<NodeUnitIODef> inputs_;
  std::vector<NodeUnitIODef> outputs_;

  // Initializing for a single Node
  void InitForSingleNode();
};

// Get all the nodes in the given graph_viewer as NodeUnits (SingleNode or QDQGroup)
// And return a map to quick query the NodeUnit which contains the given Node,
// Note, the value of the map is owned by the vector of std::unique_ptr<NodeUnit>
std::pair<std::vector<std::unique_ptr<NodeUnit>>, std::unordered_map<const Node*, const NodeUnit*>>
GetAllNodeUnits(const GraphViewer& graph_viewer);

}  // namespace onnxruntime
