// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"

namespace onnxruntime {
namespace tvm {
struct TvmEPOptions;
}

struct TVMProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(const tvm::TvmEPOptions& options);
  static std::shared_ptr<IExecutionProviderFactory> Create(const char* params);
};
}  // namespace onnxruntime
