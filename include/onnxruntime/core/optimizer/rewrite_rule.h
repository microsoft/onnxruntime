// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime {

/**
@class RewriteRule 

The base class for a rewrite rule. A rewrite rule represents a semantics-preserving
transformation of a computation graph. It can be used to represent, for example,
the elimination of operators that serve as no-ops (e.g., dropout during
inference), as well as inlining of "function" definitions or the dual (replacing
a complex expression by an equivalent function-call). Unlike the more general
IGraphTransformer, a rewrite rule is applied at a single node, representing the
root of an expression that is rewritten.

When creating a new rewrite rule, two main function have to be implemented: SatisfyCondition and Apply.
- SatisfyCondition determines whether the rule will be triggered, and can include multiple condition checks.
It is advisable to add the more selective checks first, because those will lead to discarding fast rules that 
cannot be applied on a node. One such check is the predefined OpTypeCondition check that all rules should 
implement, and which determines whether the op type of the node is compatible with the rule (e.g., to trigger 
Unsqueeze elimination, the op type of the node has to be Unsqueeze). Additional conditions should be added
in the AdditionalConditions method.
- Apply is the actual body of the rule that will be executed if the checks in SatisfyCondition are passed
successfully. Note that additional, more complex checks can be included in the Apply if putting them in the
SatisfyCondition would lead to duplicate work (e.g., when we make a check on a Node attribute, but we need
that attribute to execute the rule too).

In general, simple fast checks are a better fit for SatisfyCondition, whereas more complex ones can be 
added in the Apply.
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
  node for which this rule was triggered) or some other properties of the nodes. 
  At the moment each rule should implement the predefined OpTypeCondition check. If there
  are additional checks required, they should be added in the AdditionalConditions check. */
  bool SatisfyCondition(const Graph& graph, const Node& node) {
    return OpTypeCondition(node) && AdditionalConditions(graph, node);
  }

  /** Returns true if the op type of the node is compatible with this rewrite rule. */
  virtual bool OpTypeCondition(const Node& node) = 0;

  /** Conditions other than the ones related to the op type of the node. */
  virtual bool AdditionalConditions(const Graph& /*graph*/, const Node& /*node*/) {
    return true;
  }

  /**
  Apply the rewrite rule to a specific node.
  The transformation happens in-place. The return-value of node may be different
  from the input-value due to rewriting.
  The value of "modified" indicates if the graph was modified or not. 
  The value of "deleted" indicates if the node was deleted or not. */
  virtual common::Status Apply(Graph& graph, Node& node, bool& modified, bool& deleted) = 0;
};
}  // namespace onnxruntime
