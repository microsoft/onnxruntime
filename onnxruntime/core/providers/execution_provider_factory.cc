// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/providers.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {
std::unique_ptr<IExecutionProvider> IExecutionProviderFactory::CreateProvider(const OrtSessionOptions* /*session_options*/,
                                                                              const OrtLogger* /*logger*/) {
  return CreateProvider();
}
}  // namespace onnxruntime
