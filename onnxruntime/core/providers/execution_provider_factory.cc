// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/providers.h"

#include <memory>
#include "core/framework/execution_provider.h"

namespace onnxruntime {
std::unique_ptr<IExecutionProvider> IExecutionProviderFactory::CreateProvider(
    const OrtSessionOptions& /*session_options*/, const OrtLogger& /*session_logger*/) {
  return CreateProvider();
}
}  // namespace onnxruntime
