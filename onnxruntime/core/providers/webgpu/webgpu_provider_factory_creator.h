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
  static std::shared_ptr<IExecutionProviderFactory> Create(const ConfigOptions& config_options,
                                                           bool enable_profiling = false);
  static std::shared_ptr<IExecutionProviderFactory> CreateForTesting(
      const ConfigOptions& config_options, uint64_t max_storage_buffer_binding_size);
};

// C API to create data transfer for WebGPU EP with lazy initialization
// Context will be determined from tensors during the first CopyTensors call
// Caller takes ownership of the returned OrtDataTransferImpl*
OrtDataTransferImpl* OrtWebGpuCreateDataTransfer(int context_id = 0);

}  // namespace onnxruntime
