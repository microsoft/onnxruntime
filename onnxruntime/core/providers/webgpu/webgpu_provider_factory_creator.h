// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/framework/provider_options.h"
#include "core/providers/providers.h"

#include "core/providers/webgpu/webgpu_provider_options.h"

struct OrtDataTransferImpl;

namespace onnxruntime {
struct ConfigOptions;
class WebGpuExecutionProvider;

struct WebGpuProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(const ConfigOptions& config_options);
};

// C API to create data transfer for WebGPU EP with lazy initialization.
// A null EP selects independent environment recording; otherwise the transfer uses the EP's recording.
// Caller takes ownership of the returned OrtDataTransferImpl*.
OrtDataTransferImpl* OrtWebGpuCreateDataTransfer(int context_id = 0, WebGpuExecutionProvider* ep = nullptr);

}  // namespace onnxruntime
