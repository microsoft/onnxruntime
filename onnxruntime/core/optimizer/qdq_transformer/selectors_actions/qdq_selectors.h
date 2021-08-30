// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include "core/optimizer/selectors_actions/selector_action_transformer.h"

namespace onnxruntime {
class Graph;
class Node;

namespace QDQ {
// Base QDQ checker. Finds and provides the DQ and Q nodes to the operator specific checkers, as the QDQ optimizations
// always involve those nodes.
class BaseSelector : public NodeSelector {
 public:
  bool Select(Graph& graph, const Node& node, std::unique_ptr<NodesToOptimize>& selection) const override;

 protected:
  BaseSelector() = default;

  // base check that we have the expected number of QDQ inputs/outputs, and `node` isn't producing a graph output.
  // num_dq_inputs defaults to the number of inputs `node` has if not explicitly specified
  bool CheckQDQNodes(const Graph& graph, const Node& node,
                     const std::vector<const Node*>& dq_nodes,
                     const std::vector<const Node*>& q_nodes,
                     int num_dq_inputs = -1) const;

 private:
  // derived classes should implement this check
  bool virtual Check(const Graph& graph, const Node& node,
                     const std::vector<const Node*>& dq_nodes,
                     const std::vector<const Node*>& q_nodes) const = 0;

  // override if you need to adjust the values in NodesToOptimize.
  // e.g. add entries for missing optional DQ inputs or set num_inputs to handle variadic inputs
  // Called post-Check, if Check returned `true`
  virtual void UpdateBuilder(NodesToOptimizeBuilder&) const {}
};

// Single DQ -> node that does not change data -> Q.
// Zero point and scale are constant scalars and must match
class DropDQDNodesSelector : public BaseSelector {
 private:
  bool Check(const Graph& graph, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;
};

// single input. default is to only support uint8.
class UnarySelector : public BaseSelector {
 public:
  UnarySelector(bool int8_allowed = false) : int8_allowed_{int8_allowed} {}

 private:
  bool Check(const Graph& graph, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;

  bool int8_allowed_;
};

// 2 DQ nodes providing input -> node -> Q
class BinarySelector : public BaseSelector {
  bool Check(const Graph& graph, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;
};

// Variadic DQ nodes -> node -> Q
class VariadicSelector : public BaseSelector {
  bool Check(const Graph& graph, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;
  void UpdateBuilder(NodesToOptimizeBuilder&) const override;
};

// DQ nodes for X, W and optionally B -> node -> Q
class ConvSelector : public BaseSelector {
  bool Check(const Graph& graph, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;

  void UpdateBuilder(NodesToOptimizeBuilder&) const override;
};

// 2 DQ nodes for input -> node -> optional Q if QLinearMatMul, MatMulIntegerToFloat if not
class MatMulSelector : public BaseSelector {
  bool Check(const Graph& graph, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;
};
}  // namespace QDQ
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
