// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/tensorrt/tensorrt_provider_factory.h"
#include <atomic>
#include "tensorrt_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {

struct TensorrtProviderFactory : IExecutionProviderFactory {
  TensorrtProviderFactory(int device_id) : device_id_(device_id) {}
  ~TensorrtProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  int device_id_;
};

std::unique_ptr<IExecutionProvider> TensorrtProviderFactory::CreateProvider() {
  TensorrtExecutionProviderInfo info;
  info.device_id = device_id_;
  return std::make_unique<TensorrtExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(int device_id) {
  return std::make_shared<onnxruntime::TensorrtProviderFactory>(device_id);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Tensorrt, _In_ OrtSessionOptions* options, int device_id) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Tensorrt(device_id));
  return nullptr;
}

