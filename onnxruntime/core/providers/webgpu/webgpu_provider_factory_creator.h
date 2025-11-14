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

struct WebGpuProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(const ConfigOptions& config_options);
};

// C API to create data transfer for WebGPU EP
// Returns nullptr if WebGPU context (context_id=0) doesn't exist yet
// Caller takes ownership of the returned OrtDataTransferImpl*
OrtDataTransferImpl* OrtWebGpuCreateDataTransfer(int context_id);

}  // namespace onnxruntime
