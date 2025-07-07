// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/framework/provider_options.h"
#include "core/providers/providers.h"

namespace onnxruntime {
struct SessionOptions;

// Defined in core/session/provider_bridge_ort.cc if built as a shared library (default build config).
// Defined in core/providers/qnn-abi/qnn_provider_factory.cc if built as a static library.
// The preprocessor macro `BUILD_QNN_EP_STATIC_LIB` is defined and set to 1 if QNN is built as a static library.
struct QNNProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(const ProviderOptions& provider_options_map,
                                                           const SessionOptions* session_options);
};
}  // namespace onnxruntime