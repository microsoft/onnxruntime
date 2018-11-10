// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/graph/graph.h"
#include "core/graph/rewrite_rule.h"

namespace onnxruntime {

// A graph transformer interface. A graph transformer transforms a graph in-place.
class GraphTransformer {
 public:
  GraphTransformer(const std::string& name, const std::string& desc)
      : name_(name), desc_(desc) {
  }

  virtual ~GraphTransformer() = default;

  // The name of this graph transformer.
  const std::string& Name() const noexcept {
    return name_;
  }

  // An description of this graph transformer.
  const std::string& Description() const noexcept {
    return desc_;
  }

  // Apply <*this> transformation to a specific graph.
  // Transformation happens in place.
  // The return value of "modified" indicates if the graph was modified or not.
  virtual ::onnxruntime::common::Status Apply(Graph& graph, bool& modified) const = 0;

 private:
  ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphTransformer);

  const std::string name_;
  const std::string desc_;
};

// Rule based graph transformer.
// It provides API to register rewrite rules, and API to apply for
// all applicable rules against one graph.

// Represents a IGraphTransformer determined by a set of rewrite-rules.
// The transformer will apply all the rewrite-rules iteratively as
// determined by the underlying rewriting-strategy.
// Several rewriting-strategies are possible when traversing the graph and applying
// rewrite rules, each with different tradeoffs. At the moment, we define one
// that performs top-down traversal of nodes.
// TODO: Is a bottom-up traversal more efficient?
// TODO: Is it worth adding the max number of passes a rule should be applied for?
// TODO: We need to define a contract about whether a rewrite rule is allowed to leave
// the graph in an inconsistent state (this will determine when and where we will be
// calling resolve().
class RuleBasedGraphTransformer : public GraphTransformer {
 public:
  RuleBasedGraphTransformer(const std::string& name, const std::string& desc) : GraphTransformer(name, desc) {}

  // Register a rewriting rule.
  // TODO (revisit needed): Using OpSignature* here will ask that OpSignature
  // should be stored globally. Otherwise, there will be multiple addresses/pointers
  // for the same operator or function. To avoid this, we may use OpSignature ID
  // as the key, which should be name_domain_version.
  // We will use the string type instead of the OpSchema for now. We should probably
  // add a version as well.
  Status Register(const std::string& op_type, std::unique_ptr<RewriteRule> rule);

  // Returns true if there are rules registered for this op_type.
  bool HasRules(const std::string& op_type) const {
    return op_to_rules_.count(op_type) > 0;
  }

  // Returns a reference to the vector that contains all rewrite rules registered
  // for this operator. It assumes that there are registered rules, therefore HasRules
  // should be called before.
  const std::vector<std::unique_ptr<RewriteRule>>& GetRewriteRules(const std::string& op_type) const {
    return op_to_rules_.at(op_type);
  }

 private:
  using RewriteRuleSet = std::unordered_map<std::string, std::vector<std::unique_ptr<RewriteRule>>>;

  RewriteRuleSet op_to_rules_;
};

// This is a rule-based graph transformer that applies rules by performing top-down passes of the graph.
class TopDownRuleBasedTransformer : public RuleBasedGraphTransformer {
 public:
  TopDownRuleBasedTransformer(const std::string& name, const std::string& desc) : RuleBasedGraphTransformer(name, desc) {}

  // Performs a single top-down traversal of the graph and applies all registered rules.
  ::onnxruntime::common::Status Apply(Graph&, bool&) const override;
};

}  // namespace onnxruntime
