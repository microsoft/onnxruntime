// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/selectors_actions/selector_action_transformer.h"

namespace onnxruntime {

/**
@Class ConvAddActivationFusion

Transformer that optimizes the graph by using NCHW nodes and a more general version of convaddrelu.
This Fusion pattern is used to fuse Conv Add Activation together from different branch, The reason
is that we assume the graph would be executed by sequential executor. then the orders of branch running doesn't matter
*/
class ConvAddActivationFusion : public SelectorActionTransformer {
 public:
  ConvAddActivationFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {},
                          const SatApplyContextVariant& apply_context = {});
};

}  // namespace onnxruntime
