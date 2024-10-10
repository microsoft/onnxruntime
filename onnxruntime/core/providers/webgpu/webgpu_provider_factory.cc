// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <charconv>

#include "core/framework/error_code_helper.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"

namespace onnxruntime {

struct WebGpuProviderFactory : IExecutionProviderFactory {
  WebGpuProviderFactory() {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<WebGpuExecutionProvider>();
  }
};

std::shared_ptr<IExecutionProviderFactory> WebGpuProviderFactoryCreator::Create(const ConfigOptions&) {
  return std::make_shared<WebGpuProviderFactory>();
}

}  // namespace onnxruntime
