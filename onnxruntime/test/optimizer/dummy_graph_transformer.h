// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {
namespace test {

// Dummy graph transformer that does nothing, but just sets the modified value
class DummyGraphTransformer : public GraphTransformer {
 public:
  DummyGraphTransformer(const std::string& name) noexcept : GraphTransformer(name),
                                                            transformer_invoked_(false) {}

  bool IsTransformerInvoked() const {
    return transformer_invoked_;
  }

 private:
  mutable bool transformer_invoked_;

  Status ApplyImpl(Graph& /*graph*/, bool& /*modified*/, int /*graph_level*/, const logging::Logger&) const override {
    transformer_invoked_ = true;
    return Status::OK();
  }
};

// Dummy graph transformer that does nothing, but just sets the modified value
// This is currently used to test custom transformer selection feature
class DummyRewriteRule : public RewriteRule {
 public:
  DummyRewriteRule(const std::string& name) noexcept : RewriteRule(name), rewrite_rule_invoked_(false) {}

  bool IsRewriteRuleInvoked() const {
    return rewrite_rule_invoked_;
  }

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return std::vector<std::string>();
  }

 private:
  mutable bool rewrite_rule_invoked_;

  bool SatisfyCondition(const Graph& /*graph*/, const Node& /*node*/, const logging::Logger& /*logger*/) const override {
    return true;
  }

  Status Apply(Graph& /*graph*/, Node& /*node*/, RewriteRuleEffect& /*rule_effect*/,
               const logging::Logger& /*logger*/) const override {
    rewrite_rule_invoked_ = true;
    return Status::OK();
  }
};

}  // namespace test
}  // namespace onnxruntime
