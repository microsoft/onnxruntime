// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_TORCH_INTEROP

#pragma once

#include <string>
#include <vector>
#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
This transformer is to add schema supplementary for PythonOp.

Currently, add memory reuse output to input map as an attribute, if users registered alias input function
in `OrtTorchFunctionPool`.
*/

class PythonOpRewriter : public RewriteRule {
 public:
  PythonOpRewriter() noexcept : RewriteRule("PythonOpRewriter") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"PythonOp"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
#endif
