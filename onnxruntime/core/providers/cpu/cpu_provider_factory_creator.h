// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/providers.h"

namespace onnxruntime {

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CPU(const CPUExecutionProviderInfo& info);

}  // namespace onnxruntime
