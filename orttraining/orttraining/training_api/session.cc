// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_api/include/session.h"
#include "core/session/inference_session.h"

#ifdef USE_CUDA

#include "core/providers/cuda/cuda_execution_provider_info.h"
#include "core/providers/cuda/cuda_provider_factory.h"
#include "core/providers/cuda/cuda_provider_factory_creator.h"
#include "core/providers/cuda/cuda_provider_options.h"

namespace onnxruntime {

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cuda(const OrtCUDAProviderOptionsV2* provider_options);

}  // namespace onnxruntime

#endif

namespace onnxruntime {
namespace training {
namespace api {

SessionOptions& GetSessionOptions() {
  static onnxruntime::SessionOptions session_option = onnxruntime::SessionOptions();
  return session_option;
}

typedef std::unordered_map<std::string, std::shared_ptr<IExecutionProvider>> ExecutionProviderMapType;

ExecutionProviderMapType& GetRegisteredExecutionProviders() {
  static ExecutionProviderMapType execution_provider_map;
  return execution_provider_map;
}

#ifdef USE_CUDA

void AtExitCUDAEPHanlder() {
  auto& execution_providers = GetRegisteredExecutionProviders();
  execution_providers.clear();
}

void SetExecutionProvider(OrtCUDAProviderOptionsV2* cuda_options) {
  constexpr const char* CUDA = "Cuda";
  auto& execution_providers = GetRegisteredExecutionProviders();
  if (execution_providers.find(CUDA) == execution_providers.end()) {
    std::shared_ptr<IExecutionProviderFactory> factory = onnxruntime::CreateExecutionProviderFactory_Cuda(cuda_options);
    ORT_ENFORCE(factory, "SetExecutionProvider.CreateExecutionProviderFactory_Cuda: Failed to load shared library");
    auto provider = std::move(factory->CreateProvider());
    std::shared_ptr<IExecutionProvider> cuda_ep = std::move(provider);
    execution_providers.insert({CUDA, cuda_ep});
  }

  static bool is_ready = false;
  if (!is_ready) {
    is_ready = true;
    // Clean up cached execution provider before CUDA driver shutting down.
    // Here we clean up all execution provider (not limited to CUDA EP), which is fine.
    std::atexit(AtExitCUDAEPHanlder);
  }
}

#endif

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
