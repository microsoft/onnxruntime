// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <memory>

#include "core/framework/execution_provider.h"
#include "core/framework/config_options.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "test/providers/webgpu/internal_test_api.h"

namespace onnxruntime::test {

// Loads the private test plugin for cases that need implementation-only setup.
// Providers returned by MakeEp must be destroyed before this object.
class WebGpuTestPlugin {
 public:
  WebGpuTestPlugin();
  ~WebGpuTestPlugin();
  std::unique_ptr<IExecutionProvider> MakeEp(const ConfigOptions& options = {},
                                             const WebGpuExecutionProviderTestOptions& test_options = {}) const;
  WebGpuTestCapabilities Inspect(const IExecutionProvider& ep) const;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WebGpuTestPlugin);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

WebGpuTestPlugin& GetWebGpuTestPlugin();

}  // namespace onnxruntime::test
