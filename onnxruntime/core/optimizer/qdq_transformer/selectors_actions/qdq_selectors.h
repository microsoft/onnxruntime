// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include "core/optimizer/selectors_actions/selector_action_transformer.h"

namespace onnxruntime {
class Graph;
class Node;

namespace QDQ {

// Struct to represent a DQ->Op->Q node group
struct NodeGroup {
  std::vector<NodeIndex> dq_nodes;
  std::vector<NodeIndex> q_nodes;
  NodeIndex target_node;
};

class NodeGroupSelector {
 public:
  // This is a QDQ Selectors only function, will return QDQ::NodeGroup instead of NodesToOptimizeIndices
  // Can be used in QDQ handling in EPs such as NNAPI
  std::optional<NodeGroup> GetQDQSelection(const GraphViewer& graph_viewer, const Node& node) const;

  virtual ~NodeGroupSelector() = default;

 protected:
  // base check that we have the expected number of QDQ inputs/outputs, and `node` isn't producing a graph output.
  // num_dq_inputs defaults to the number of inputs `node` has if not explicitly specified
  bool CheckQDQNodes(const GraphViewer& graph_viewer, const Node& node,
                     const std::vector<const Node*>& dq_nodes,
                     const std::vector<const Node*>& q_nodes,
                     int num_dq_inputs = -1,
                     bool is_empty_q_nodes_allowed = false) const;

 private:
  // derived classes should implement this check
  bool virtual Check(const GraphViewer& graph_viewer, const Node& node,
                     const std::vector<const Node*>& dq_nodes,
                     const std::vector<const Node*>& q_nodes) const = 0;
};

/*
 * NodeGroup selectors. These are general purpose and used in both the QDQ SelectorActionTransformer setup that the
 * CPU EP has, and directly in compiling EPs such as NNAPI and CoreML.
 */

// Single DQ -> node that does not change data -> Q.
// Zero point and scale are constant scalars and must match
class DropQDQNodeGroupSelector : public NodeGroupSelector {
  bool Check(const GraphViewer& graph_viewer, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;
};

// Single DQ -> node.
class DropDQNodeGroupSelector : public NodeGroupSelector {
  bool Check(const GraphViewer& graph_viewer, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;
};

// single input. default is to only support uint8.
class UnaryNodeGroupSelector : public NodeGroupSelector {
  bool Check(const GraphViewer& graph_viewer, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;
};

// 2 DQ nodes providing input -> node -> Q
class BinaryNodeGroupSelector : public NodeGroupSelector {
  bool Check(const GraphViewer& graph_viewer, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;
};

// Variadic DQ nodes -> node -> Q
class VariadicNodeGroupSelector : public NodeGroupSelector {
 private:
  bool Check(const GraphViewer& graph_viewer, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;
};

// DQ nodes for X, W and optionally B -> node -> Q
class ConvNodeGroupSelector : public NodeGroupSelector {
 public:
  // default to 'true'
  ConvNodeGroupSelector(bool int8_allowed = true) : int8_allowed_(int8_allowed) {}

 private:
  bool Check(const GraphViewer& graph_viewer, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;

  bool int8_allowed_;
};

class WhereNodeGroupSelector : public NodeGroupSelector {
 public:
  WhereNodeGroupSelector() = default;

 private:
  bool Check(const GraphViewer& graph_viewer, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;
};

// 2 DQ nodes for input -> node -> optional Q if QLinearMatMul, MatMulIntegerToFloat if not
// The lack of a trailing Q isn't really a QDQ node group, so we default support for that to off.
class MatMulNodeGroupSelector : public NodeGroupSelector {
 public:
  MatMulNodeGroupSelector(bool int8_allowed = true,
                          bool matmulintegertofloat_allowed = false)
      : int8_allowed_(int8_allowed),
        matmulintegertofloat_allowed_(matmulintegertofloat_allowed) {
  }

 private:
  bool Check(const GraphViewer& graph_viewer, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;
  bool int8_allowed_;
  bool matmulintegertofloat_allowed_;
};

// Input: DQ nodes for A, B and optional C
// Output: optional Q node for Y
class GemmNodeGroupSelector : public NodeGroupSelector {
 private:
  bool Check(const GraphViewer& graph_viewer, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;
};

// Input: DQ nodes for input, scale, and B
// Output: Q node for output
class InstanceAndLayerNormalizationNodeGroupSelector : public NodeGroupSelector {
 private:
  bool Check(const GraphViewer& graph_viewer, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;
};

// DQ nodes for X, W and optionally B, not used for mean, var -> node -> Q
class BatchNormalizationNodeGroupSelector : public NodeGroupSelector {
 public:
  // default to 'true'
  BatchNormalizationNodeGroupSelector(bool int8_allowed = true) : int8_allowed_(int8_allowed) {}

 private:
  bool Check(const GraphViewer& graph_viewer, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;

  bool int8_allowed_;
};

// 2 DQ nodes providing input -> node with bool output tensor.
// Example: Equal, Less, Greater.
class LogicalComparisonNodeGroupSelector : public NodeGroupSelector {
  bool Check(const GraphViewer& graph_viewer, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;
};

/*
 * NodeSelector instances for use in the QDQ::SelectorActionTransformer.
 */
// Base QDQ checker. Finds and provides the DQ and Q nodes to the operator specific checkers, as the QDQ optimizations
// always involve those nodes.
class BaseSelector : public NodeSelector {
 public:
  std::optional<NodesToOptimizeIndices> Select(const GraphViewer& graph_viewer, const Node& node) const override;

  // We std::move SelectorActionRegistry into the SelectorActionTransformer so this class needs to have a move ctor
  BaseSelector(BaseSelector&& rhs) noexcept
      : node_group_selector_{std::move(rhs.node_group_selector_)} {
  }

 protected:
  BaseSelector(std::unique_ptr<NodeGroupSelector> node_group_selector)
      : node_group_selector_{std::move(node_group_selector)} {}

  // override if you need to adjust the values in NodesToOptimize.
  // e.g. add entries for missing optional DQ inputs or set num_inputs to handle variadic inputs
  // Called post-Check, if Check returned `true`
  virtual void UpdateBuilder(NodesToOptimizeIndicesBuilder&) const {}

 private:
  std::unique_ptr<NodeGroupSelector> node_group_selector_;
};

class DropQDQNodesSelector : public BaseSelector {
 public:
  DropQDQNodesSelector() : BaseSelector(std::make_unique<DropQDQNodeGroupSelector>()) {}
};

class DropDQNodesSelector : public BaseSelector {
 public:
  DropDQNodesSelector() : BaseSelector(std::make_unique<DropDQNodeGroupSelector>()) {}
};

class UnarySelector : public BaseSelector {
 public:
  UnarySelector() : BaseSelector(std::make_unique<UnaryNodeGroupSelector>()) {}
};

class BinarySelector : public BaseSelector {
 public:
  BinarySelector() : BaseSelector(std::make_unique<BinaryNodeGroupSelector>()) {}
};

// Variadic DQ nodes -> node -> Q
class InputVariadicSelector : public BaseSelector {
 public:
  InputVariadicSelector() : BaseSelector(std::make_unique<VariadicNodeGroupSelector>()) {}

  void UpdateBuilder(NodesToOptimizeIndicesBuilder&) const override;
};

//  DQ -> node -> Variadic Q nodes
class OutputVariadicSelector : public BaseSelector {
 public:
  OutputVariadicSelector() : BaseSelector(std::make_unique<VariadicNodeGroupSelector>()) {}

  void UpdateBuilder(NodesToOptimizeIndicesBuilder&) const override;
};

// DQ nodes for X, W and optionally B -> node -> Q
class ConvSelector : public BaseSelector {
 public:
  ConvSelector(bool int8_allowed = false) : BaseSelector(std::make_unique<ConvNodeGroupSelector>(int8_allowed)) {}

  void UpdateBuilder(NodesToOptimizeIndicesBuilder&) const override;
};
class WhereSelector : public BaseSelector {
 public:
  WhereSelector() : BaseSelector(std::make_unique<WhereNodeGroupSelector>()) {}
};
// 2 DQ nodes for input -> node -> optional Q if QLinearMatMul, MatMulIntegerToFloat if not
class MatMulSelector : public BaseSelector {
 public:
  MatMulSelector(bool int8_allowed)
      : BaseSelector(std::make_unique<MatMulNodeGroupSelector>(int8_allowed, /*matmulintegertofloat_allowed*/ true)) {}
};

// Input: DQ nodes for A, B and optional C
// Output: optional Q node for Y
class GemmSelector : public BaseSelector {
 public:
  GemmSelector()
      : BaseSelector(std::make_unique<GemmNodeGroupSelector>()) {}

  void UpdateBuilder(NodesToOptimizeIndicesBuilder&) const override;
};

// Input: DQ nodes for input, scale, and B (bias)
// Output: Q node for output
class InstanceNormalizationSelector : public BaseSelector {
 public:
  InstanceNormalizationSelector()
      : BaseSelector(std::make_unique<InstanceAndLayerNormalizationNodeGroupSelector>()) {}
};

// DQ nodes for X, W and optionally B, (mean, var not required) -> node -> Q
class BatchNormalizationSelector : public BaseSelector {
 public:
  BatchNormalizationSelector(bool int8_allowed = false)
      : BaseSelector(std::make_unique<BatchNormalizationNodeGroupSelector>(int8_allowed)) {}
};

}  // namespace QDQ
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
