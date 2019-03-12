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

Rule based graph transformer that provides an API to register rewrite rules, 
and an API to apply all applicable rules to a Graph.

Represents an IGraphTransformer determined by a set of rewrite-rules.
The transformer will apply all the rewrite-rules iteratively as determined by the underlying rewriting-strategy.
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
  Register a rewriting rule.

  @TODO (revisit needed): Using OpSignature* here will ask that OpSignature should be stored globally. 
  Otherwise, there will be multiple addresses/pointers for the same operator or function. 
  To avoid this, we may use OpSignature ID as the key, which should be name_domain_version.
  We will use the string type instead of the OpSchema for now. We should probably add a version as well.
  */
  Status Register(const std::string& op_type, std::unique_ptr<RewriteRule> rule);

  /** Check if the given op_type has any rules registered for it 
  @returns true if there are rules registered for this op_type.*/
  bool HasRules(const std::string& op_type) const {
    return op_to_rules_.find(op_type) != op_to_rules_.cend();
  }

  /**
  Gets the rewrite rules for the given op_type.
  @returns a pointer to the vector containing all the rewrite rules registered for op_type if found. nullptr
  otherwise.
  */
  const std::vector<std::unique_ptr<RewriteRule>>* GetRewriteRules(const std::string& op_type) const {
    auto entry = op_to_rules_.find(op_type);
    if (entry != op_to_rules_.cend())
      return &entry->second;

    return nullptr;
  }

 private:
  using RewriteRuleSet = std::unordered_map<std::string, std::vector<std::unique_ptr<RewriteRule>>>;

  RewriteRuleSet op_to_rules_;
};

/**
@class TopDownRuleBasedTransformer

This is a rule-based Graph transformer that applies rules by performing top-down passes of the Graph.
*/
class TopDownRuleBasedTransformer : public RuleBasedGraphTransformer {
 public:
  TopDownRuleBasedTransformer(const std::string& name, const std::string& desc)
      : RuleBasedGraphTransformer(name, desc) {}

 private:
  // Performs a single top-down traversal of the graph and applies all registered rules.
  common::Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override;
};

}  // namespace onnxruntime
