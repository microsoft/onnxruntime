// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/selectors_actions/selector_action_transformer.h"

namespace onnxruntime {

class ConvActivationFusion : public SelectorActionTransformer {
 public:
  explicit ConvActivationFusion(const InlinedHashSet <std::string_view> &compatible_execution_providers = {},
                                std::shared_ptr<KernelRegistry> cpu_kernel_registry = {},
                                const SatApplyContextVariant &apply_context = {}) noexcept;
};

}  // namespace onnxruntime
