// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// gather_internal_unused_output_removal.h

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class GatherInternalUnusedOutputRemoval

Rewrite rule that replaces the GatherInternal with Gather when the extra
outputs computed by GatherInternal are not needed.

This happens when the GatherInternal node is not part of the backward graph
and hence is not connected to a GatherGrad node. So the extra precomputed outputs
are not needed to be computed.

     data    indices
       \       /
        \     /
     GatherInternal
       ||||||||| (nine output)
          ...

is rewritten so that GatherInternal is replaced by Gather (which has a single output)

     data    indices
       \       /
        \     /
         Gather
           | (single output)
          ...

*/
class GatherInternalUnusedOutputRemoval : public RewriteRule {
 public:
  GatherInternalUnusedOutputRemoval() noexcept : RewriteRule("GatherInternalUnusedOutputRemoval") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"GatherInternal"};
  }

 private:
  bool SatisfyCondition(const Graph& graph,
                        const Node& node,
                        const logging::Logger& logger) const override;

  Status Apply(Graph& graph,
               Node& node,
               RewriteRuleEffect& rule_effect,
               const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
