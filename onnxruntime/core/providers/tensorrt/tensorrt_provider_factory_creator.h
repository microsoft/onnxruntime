// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"

struct OrtTensorRTProviderOptions;
struct OrtTensorRTProviderOptionsV2;

namespace onnxruntime {
// defined in provider_bridge_ort.cc
struct TensorrtProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(int device_id);
  static std::shared_ptr<IExecutionProviderFactory> Create(const OrtTensorRTProviderOptions* provider_options);
  static std::shared_ptr<IExecutionProviderFactory> Create(const OrtTensorRTProviderOptionsV2* provider_options);
};
}  // namespace onnxruntime
