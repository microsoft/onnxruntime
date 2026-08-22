// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/framework/provider_options.h"
#include "core/providers/providers.h"

#include "core/providers/webgpu/webgpu_provider_options.h"

struct OrtDataTransferImpl;
struct OrtKeyValuePairs;

namespace onnxruntime {
struct ConfigOptions;

webgpu::WebGpuDeviceConfig ParseWebGpuDeviceConfig(const char* enable_robustness,
                                                   const char* enable_zero_buffer);
webgpu::WebGpuDeviceConfig ParseWebGpuDeviceConfig(const OrtKeyValuePairs& environment_options);

struct WebGpuProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(
      const ConfigOptions& config_options,
      const webgpu::WebGpuDeviceConfig& device_config = {});
};

// C API to create data transfer for WebGPU EP with lazy initialization
// Context will be determined from tensors during the first CopyTensors call
// Caller takes ownership of the returned OrtDataTransferImpl*
OrtDataTransferImpl* OrtWebGpuCreateDataTransfer(
    int context_id,
    const webgpu::WebGpuDeviceConfig& device_config);

}  // namespace onnxruntime
