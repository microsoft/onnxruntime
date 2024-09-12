// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/selectors_actions/selector_action_transformer.h"

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

}  // namespace onnxruntime
