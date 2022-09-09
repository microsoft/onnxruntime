// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"

namespace onnxruntime {
// defined in provider_bridge_ort.cc
struct DnnlProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(int use_arena);
};
}  // namespace onnxruntime
