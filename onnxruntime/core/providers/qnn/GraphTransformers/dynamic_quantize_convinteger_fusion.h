// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class DynamicQuantizeConvInteger
Fuse DynamicQuantizeLinear + ConvInteger and following cast and mul to Dequantze + Conv
*/
class DynamicQuantizeConvIntegerFusion : public RewriteRule {
 public:
  DynamicQuantizeConvIntegerFusion() noexcept
      : RewriteRule("DynamicQuantizeConvIntegerFusion") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"ConvInteger"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const onnxruntime::logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const onnxruntime::logging::Logger& logger) const override;
};

}  // namespace onnxruntime
