// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"

struct OrtOpenVINOProviderOptions;
struct OrtOpenVINOProviderOptionsV2;

namespace onnxruntime {
// defined in provider_bridge_ort.cc
struct OpenVINOProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(const OrtOpenVINOProviderOptions* provider_options);
  static std::shared_ptr<IExecutionProviderFactory> Create(const OrtOpenVINOProviderOptionsV2* provider_options);
};
}  // namespace onnxruntime
