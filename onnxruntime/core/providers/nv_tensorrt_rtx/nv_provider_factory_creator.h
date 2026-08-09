// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/framework/provider_options.h"
#include "core/providers/providers.h"

namespace onnxruntime {
struct SessionOptions;

// defined in provider_bridge_ort.cc
struct NvProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(int device_id);
  static std::shared_ptr<IExecutionProviderFactory> Create(const ProviderOptions& provider_options_map,
                                                           const SessionOptions* session_options);
};
}  // namespace onnxruntime
