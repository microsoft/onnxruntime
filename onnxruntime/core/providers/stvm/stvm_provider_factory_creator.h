// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef STVM_PROVIDER_FACTORY_CREATOR_H
#define STVM_PROVIDER_FACTORY_CREATOR_H

#include <memory>

#include "core/providers/providers.h"
#include "core/providers/stvm/stvm_execution_provider_info.h"

namespace onnxruntime {

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Stvm(const StvmExecutionProviderInfo& info);

}  // namespace onnxruntime

#endif  // STVM_PROVIDER_FACTORY_CREATOR_H
