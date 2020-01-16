// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/tensorrt/tensorrt_provider_factory.h"
#include <atomic>
#include "tensorrt_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {

struct TensorrtProviderFactory : IExecutionProviderFactory {
  TensorrtProviderFactory(int device_id, bool use_cuda_arena)
      : device_id_(device_id), use_cuda_arena_(use_cuda_arena) {}
  ~TensorrtProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  int device_id_;
  bool use_cuda_arena_;
};

std::unique_ptr<IExecutionProvider> TensorrtProviderFactory::CreateProvider() {
  TensorrtExecutionProviderInfo info;
  info.device_id = device_id_;
  info.use_cuda_arena = use_cuda_arena_;
  return onnxruntime::make_unique<TensorrtExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(int device_id,
                                                                                   bool enable_cuda_mem_arena = true) {
  return std::make_shared<onnxruntime::TensorrtProviderFactory>(device_id, enable_cuda_mem_arena);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Tensorrt, _In_ OrtSessionOptions* options, int device_id) {
  options->provider_factories.push_back(
      onnxruntime::CreateExecutionProviderFactory_Tensorrt(device_id, options->value.enable_cuda_mem_arena));
  return nullptr;
}
