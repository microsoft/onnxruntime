// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/selectors_actions/selector_action_transformer.h"

namespace onnxruntime {

class ConvActivationFusion : public GraphTransformer {
 public:
  ConvActivationFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("ConvActivationFusion", compatible_execution_providers) {}

 private:
  Status ApplyImpl(onnxruntime::Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

class ConvActivationFusion2 : public SelectorActionTransformer {
 public:
  ConvActivationFusion2(const std::unordered_set<std::string>& compatible_execution_providers = {},
                        const SatApplyContextVariant& apply_context = {});
};

}  // namespace onnxruntime
