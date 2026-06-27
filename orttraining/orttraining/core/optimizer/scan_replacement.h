#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {
class ScanReplacement : public RewriteRule {
 public:
  ScanReplacement() noexcept : RewriteRule("ScanReplacement") {}
  std::vector<std::string> TargetOpTypes() const noexcept override { return {"Scan"}; }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;
  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};
}  // namespace onnxruntime
