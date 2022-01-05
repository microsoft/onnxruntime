// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

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

// Base QDQ checker. Finds and provides the DQ and Q nodes to the operator specific checkers, as the QDQ optimizations
// always involve those nodes.
class BaseSelector : public NodeSelector {
 public:
  std::optional<NodesToOptimizeIndices> Select(const GraphViewer& graph_viewer, const Node& node) const override;

  // This is a QDQ Selectors only function, will return QDQ::NodeGroup instead of NodesToOptimizeIndices
  // Can be used in QDQ handling in EPs such as NNAPI
  std::optional<NodeGroup> GetQDQSelection(const GraphViewer& graph_viewer, const Node& node) const;

 protected:
  BaseSelector() = default;

  // base check that we have the expected number of QDQ inputs/outputs, and `node` isn't producing a graph output.
  // num_dq_inputs defaults to the number of inputs `node` has if not explicitly specified
  bool CheckQDQNodes(const GraphViewer& graph_viewer, const Node& node,
                     const std::vector<const Node*>& dq_nodes,
                     const std::vector<const Node*>& q_nodes,
                     int num_dq_inputs = -1) const;

 private:
  // derived classes should implement this check
  bool virtual Check(const GraphViewer& graph_viewer, const Node& node,
                     const std::vector<const Node*>& dq_nodes,
                     const std::vector<const Node*>& q_nodes) const = 0;

  // override if you need to adjust the values in NodesToOptimize.
  // e.g. add entries for missing optional DQ inputs or set num_inputs to handle variadic inputs
  // Called post-Check, if Check returned `true`
  virtual void UpdateBuilder(NodesToOptimizeIndicesBuilder&) const {}
};

// Single DQ -> node that does not change data -> Q.
// Zero point and scale are constant scalars and must match
class DropDQDNodesSelector : public BaseSelector {
 private:
  bool Check(const GraphViewer& graph_viewer, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;
};

// single input. default is to only support uint8.
class UnarySelector : public BaseSelector {
  bool Check(const GraphViewer& graph_viewer, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;

};

// 2 DQ nodes providing input -> node -> Q
class BinarySelector : public BaseSelector {
  bool Check(const GraphViewer& graph_viewer, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;
};

// Variadic DQ nodes -> node -> Q
class VariadicSelector : public BaseSelector {
 public:
  void UpdateBuilder(NodesToOptimizeIndicesBuilder&) const override;

 private:
  bool Check(const GraphViewer& graph_viewer, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;
};

// DQ nodes for X, W and optionally B -> node -> Q
class ConvSelector : public BaseSelector {
 public:
  ConvSelector(bool int8_allowed = false) : int8_allowed_(int8_allowed) {}

  void UpdateBuilder(NodesToOptimizeIndicesBuilder&) const override;

 private:
  bool Check(const GraphViewer& graph_viewer, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;

  bool int8_allowed_;
};

// 2 DQ nodes for input -> node -> optional Q if QLinearMatMul, MatMulIntegerToFloat if not
class MatMulSelector : public BaseSelector {
 public:
  MatMulSelector(bool int8_allowed = false) : int8_allowed_(int8_allowed) {}

 private:
  bool Check(const GraphViewer& graph_viewer, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;
  bool int8_allowed_;
};
}  // namespace QDQ
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
