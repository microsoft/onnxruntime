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
  static std::shared_ptr<IExecutionProviderFactory> CreateForTesting(
      const ConfigOptions& config_options, uint64_t max_storage_buffer_binding_size);
};

// The caller owns the returned transfer. An EP-bound transfer uses that EP's current recording and
// buffer manager; the EP must outlive the transfer. Without an EP, the default Env context is acquired
// lazily on the first copy, so registration does not determine the Session's device configuration.
OrtDataTransferImpl* OrtWebGpuCreateDataTransfer(int context_id = 0, WebGpuExecutionProvider* ep = nullptr);

}  // namespace onnxruntime
