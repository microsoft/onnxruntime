// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/selectors_actions/selector_action_transformer.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

// Performs node fusions with MatMulNBits.
// Currently supports these fusions:
// - MatMulNBits + Add -> MatMulNBits with bias input
class MatMulNBitsFusion : public SelectorActionTransformer {
 public:
  MatMulNBitsFusion(const InlinedHashSet<std::string_view>& compatible_eps = {},
                    const SatApplyContextVariant& apply_context = {});

  SelectorActionRegistry CreateSelectorActionRegistry() const;
};

#if !defined(ORT_MINIMAL_BUILD)
class LoraMulAddFusion final : public GraphTransformer {
 public:
  explicit LoraMulAddFusion(const InlinedHashSet<std::string_view>& compatible_eps)
      : GraphTransformer("LoraMulAddFusion", compatible_eps) {}

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LoraMulAddFusion);

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level,
                   const logging::Logger& logger) const override;
};
#endif

}  // namespace onnxruntime
