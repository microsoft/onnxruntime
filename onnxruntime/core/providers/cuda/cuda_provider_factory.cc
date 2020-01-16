// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_provider_factory.h"
#include <atomic>
#include "cuda_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {

struct CUDAProviderFactory : IExecutionProviderFactory {
  CUDAProviderFactory(int device_id, bool use_cuda_arena, bool use_cpu_arena)
      : device_id_(device_id), use_cuda_arena_(use_cuda_arena), use_cpu_arena_(use_cpu_arena) {}
  ~CUDAProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  int device_id_;
  bool use_cuda_arena_;
  bool use_cpu_arena_;
};

std::unique_ptr<IExecutionProvider> CUDAProviderFactory::CreateProvider() {
  CUDAExecutionProviderInfo info;
  info.device_id = device_id_;
  info.use_cuda_arena = use_cuda_arena_;
  info.use_cpu_arena = use_cpu_arena_;

  return onnxruntime::make_unique<CUDAExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CUDA(int device_id,
                                                                               bool enable_cuda_mem_arena = true,
                                                                               bool enable_cpu_mem_arena = true) {
  return std::make_shared<onnxruntime::CUDAProviderFactory>(device_id, enable_cuda_mem_arena, enable_cpu_mem_arena);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, int device_id) {
  options->provider_factories.push_back(
      onnxruntime::CreateExecutionProviderFactory_CUDA(device_id,
                                                       options->value.enable_cuda_mem_arena,
                                                       options->value.enable_cpu_mem_arena));
  return nullptr;
}
