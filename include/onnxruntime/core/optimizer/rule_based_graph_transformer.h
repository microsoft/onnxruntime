// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@class RuleBasedGraphTransformer

Rule-based graph transformer that provides an API to register rewrite rules 
and an API to apply all applicable rules to a Graph.

Represents an IGraphTransformer determined by a set of rewrite rules.
The transformer will apply all the rewrite rules iteratively as determined by the underlying rewriting strategy.
Several rewriting-strategies are possible when traversing the graph and applying rewrite rules, 
each with different trade offs. At the moment, we define one that performs top-down traversal of nodes.

@TODO: Is a bottom-up traversal more efficient?
@TODO: Is it worth adding the max number of passes a rule should be applied for?
@TODO: We need to define a contract about whether a rewrite rule is allowed to leave
       the graph in an inconsistent state (this will determine when and where we will be
       calling Graph::resolve().
*/
class RuleBasedGraphTransformer : public GraphTransformer {
 public:
  RuleBasedGraphTransformer(const std::string& name, const std::string& desc)
      : GraphTransformer(name, desc) {}

  /**
  Register a rewrite rule in this transformer.
  */
  Status Register(std::unique_ptr<RewriteRule> rule);

  /**
  Gets the list of registered rewrite rules in this rule-based transformer.
  @returns a reference to the vector containing all the registered rewrite rules.
  */
  const std::vector<std::unique_ptr<RewriteRule>>& GetRewriteRules() const {
    return rules_;
  }

 protected:
  /** Apply the given set of rewrite rules on the Node of this Graph.
  @param[in] graph The Graph.
  @param[in] node The Node to apply the rules to.
  @param[in] rules The vector of RewriteRules that will be applied to the Node.
  @param[out] modified Set to indicate whether the node was modified or not.
  @param[out] deleted Set to indicate if the node was deleted.
  @returns Status indicating success or providing error information */
  common::Status ApplyRulesOnNode(Graph& graph, Node& node,
                                  const std::vector<std::unique_ptr<RewriteRule>>& rules,
                                  bool& modified, bool& deleted) const;

 private:
  std::vector<std::unique_ptr<RewriteRule>> rules_;

  // Performs a single top-down traversal of the graph and applies all registered rules.
  common::Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override;
};

}  // namespace onnxruntime
