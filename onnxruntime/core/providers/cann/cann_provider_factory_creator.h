// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"

struct OrtCANNProviderOptions;

namespace onnxruntime {
struct CannProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(const OrtCANNProviderOptions* provider_options);
};
}  // namespace onnxruntime
