// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_provider_factory.h"
#include <atomic>
#include "cuda_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {

struct CUDAProviderFactory : IExecutionProviderFactory {
  CUDAProviderFactory(int device_id, bool enable_compiler) : device_id_(device_id), enable_compiler_(enable_compiler) {}
  ~CUDAProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  int device_id_;
  bool enable_compiler_;
};

std::unique_ptr<IExecutionProvider> CUDAProviderFactory::CreateProvider() {
  CUDAExecutionProviderInfo info;
  info.device_id = device_id_;
  bool enable_compiler = enable_compiler_;
  return std::make_unique<CUDAExecutionProvider>(info, enable_compiler);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CUDA(int device_id, bool enable_compiler = false) {
  return std::make_shared<onnxruntime::CUDAProviderFactory>(device_id, enable_compiler);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, int device_id, bool enable_compiler) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_CUDA(device_id, enable_compiler));
  return nullptr;
}
