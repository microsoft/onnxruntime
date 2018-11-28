// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/graph/graph.h"

namespace onnxruntime {

/**
@class GraphEditor
The API for Graph rewrite rules.
*/
class GraphEditor {
 public:
  explicit GraphEditor(Graph& graph) noexcept : graph_{graph} {}

  /** Add a node to this Graph */
  Node& AddNode(const std::string& name,
                const std::string& op_type,
                const std::string& description,
                const std::vector<NodeArg*>& input_args,
                const std::vector<NodeArg*>& output_args,
                const std::string& domain = "") {
    return graph_.AddNode(name, op_type, description,
                          input_args, output_args, nullptr, domain);
  }

  /** Copy an existing node into this Graph. */
  Node& AddNode(const Node& other) {
    return graph_.AddNode(other);
  }

  /** Remove a node from this Graph. */
  bool RemoveNode(NodeIndex node_index) {
    return graph_.RemoveNode(node_index);
  }

  /** Add a control edge between two Nodes in this Graph
  The <dst> node does not consume any data output by <src>, so there is no input/output edge between them, 
  but dst must executed after src so a control edge is required.
  @param src NodeIndex from the Graph of the Node which must execute first.
  @param dst NodeIndex from the Graph of the Node which must execute after src.
  */
  bool AddControlEdge(NodeIndex src, NodeIndex dst) {
    return graph_.AddControlEdge(src, dst);
  }

  /** Resolve the Graph.
  @returns Status indicating success or providing error information.
  @remarks Resolve must be called after modifying the Graph is completed. */
  common::Status Resolve() {
    return graph_.Resolve();
  }

 private:
  ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphEditor);

  Graph& graph_;
};

/**
@class RewriteRule 

The base class for a rewrite rule. A rewrite rule represents a semantics-preserving
transformation of a computation-graph. It can be used to represent, for example,
the elimination of operators that serve as no-ops (for example, dropout during
inference), as well as inlining of "function" definitions or the dual (replacing
a complex expression by an equivalent function-call). Unlike the more general
IGraphTransformer, a rewrite-rule is applied at a single node, representing the
root of an expression that is rewritten.
*/
class RewriteRule {
 public:
  RewriteRule(const std::string& name, const std::string& desc)
      : name_(name), desc_(desc) {
  }

  virtual ~RewriteRule() = default;

  /** Gets the name of this rewrite rule. */
  const std::string& Name() const noexcept {
    return name_;
  }

  /** Gets the description of this rewrite rule. */
  const std::string& Description() const noexcept {
    return desc_;
  }

  /** Checks if the condition of the rule is satisfied, and if so applies the rule.
  @param graph_editor The GraphEditor.
  @param node The Node to apply the rewrite to.
  @param[out] modified Set to indicate whether the node was modified or not.
  @returns Status indicating success or providing error information */
  common::Status CheckConditionAndApply(GraphEditor& graph_editor, Node& node, bool& modified) {
    return SatisfyCondition(node) ? Apply(graph_editor, node, modified) : Status::OK();
  }

 private:
  ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RewriteRule);

  const std::string name_;
  const std::string desc_;

  /** Check if the Node satisfies a condition.
  The rewrite rule is applied if the condition function returns true. This can include
  a more complex pattern matching (conditions on the ascending or descending nodes of the
  node for which this rule was triggered) or some other properties of the nodes. */
  virtual bool SatisfyCondition(const Node& node) = 0;

  /**
  Apply the rewrite rule to a specific node.
  The transformation happens in-place. The return-value of node may be different
  from the input-value due to rewriting.
  The value of "modified" indicates if the graph was modified or not. */
  virtual common::Status Apply(GraphEditor& graph_editor, Node& node, bool& modified) = 0;
};
}  // namespace onnxruntime
