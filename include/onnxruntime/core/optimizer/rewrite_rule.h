// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime {

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
  @param[in] graph The Graph.
  @param[in] node The Node to apply the rewrite to.
  @param[out] modified Set to indicate whether the node was modified or not.
  @param[out] deleted Set to indicate if the node was deleted. 
  @returns Status indicating success or providing error information */
  common::Status CheckConditionAndApply(Graph& graph, Node& node, bool& modified, bool& deleted) {
    return SatisfyCondition(graph, node) ? Apply(graph, node, modified, deleted) : Status::OK();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RewriteRule);

  const std::string name_;
  const std::string desc_;

  /** Check if the Node of the given Graph satisfies a condition.
  The rewrite rule is applied if the condition function returns true. This can include
  a more complex pattern matching (conditions on the ascending or descending nodes of the
  node for which this rule was triggered) or some other properties of the nodes. */
  virtual bool SatisfyCondition(const Graph& graph, const Node& node) = 0;

  /** Returns true if the op type of the node is compatible with this rewrite rule. */
  virtual bool OpTypeCondition(const Node& node) = 0;

  /**
  Apply the rewrite rule to a specific node.
  The transformation happens in-place. The return-value of node may be different
  from the input-value due to rewriting.
  The value of "modified" indicates if the graph was modified or not. 
  The value of "deleted" indicates if the node was deleted or not. */
  virtual common::Status Apply(Graph& graph, Node& node, bool& modified, bool& deleted) = 0;
};
}  // namespace onnxruntime
