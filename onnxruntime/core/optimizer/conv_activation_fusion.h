// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/selectors_actions/selector_action_transformer.h"

namespace onnxruntime {

class ConvActivationFusion : public SelectorActionTransformer {
 public:
  ConvActivationFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {},
                       const SatApplyContextVariant& apply_context = {});
};

}  // namespace onnxruntime
