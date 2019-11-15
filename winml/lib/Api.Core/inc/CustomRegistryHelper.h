// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "core/providers/dml/DmlExecutionProvider/src/AbiCustomRegistry.h"

namespace Windows::AI::MachineLearning::Adapter {

inline std::list<std::shared_ptr<onnxruntime::CustomRegistry>>
GetLotusCustomRegistries(
    IMLOperatorRegistry* registry) {
  if (registry != nullptr) {
    // Down-cast to the concrete type.
    // The only supported input is the AbiCustomRegistry type.
    // Other implementations of IMLOperatorRegistry are forbidden.
    auto abi_custom_registry =
        static_cast<AbiCustomRegistry*>(registry);

    // Get the ORT registry
    return abi_custom_registry->GetRegistries();
  }

  return {};
}

}  // namespace Windows::AI::MachineLearning::Adapter