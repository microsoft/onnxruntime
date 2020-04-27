// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_provider_factory.h"
#include <atomic>
#include "core/graph/onnx_protobuf.h"
#include "cuda_execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/framework/bfc_arena.h"

using namespace onnxruntime;

namespace onnxruntime {

struct CUDAProviderFactory : IExecutionProviderFactory {
  CUDAProviderFactory(OrtDevice::DeviceId device_id,
                      size_t cuda_mem_limit = std::numeric_limits<size_t>::max(),
                      ArenaExtendStrategy arena_extend_strategy = ArenaExtendStrategy::kNextPowerOfTwo) 
      : device_id_(device_id), 
        cuda_mem_limit_(cuda_mem_limit), 
        arena_extend_strategy_(arena_extend_strategy) {}
  ~CUDAProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  OrtDevice::DeviceId device_id_;
  size_t cuda_mem_limit_;
  ArenaExtendStrategy arena_extend_strategy_;
};

std::unique_ptr<IExecutionProvider> CUDAProviderFactory::CreateProvider() {
  CUDAExecutionProviderInfo info;
  info.device_id = device_id_;
  info.cuda_mem_limit = cuda_mem_limit_;
  info.arena_extend_strategy = arena_extend_strategy_;
  return onnxruntime::make_unique<CUDAExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CUDA(OrtDevice::DeviceId device_id,
                                                                               size_t cuda_mem_limit = std::numeric_limits<size_t>::max(),
                                                                               ArenaExtendStrategy arena_extend_strategy = ArenaExtendStrategy::kNextPowerOfTwo) {
  return std::make_shared<onnxruntime::CUDAProviderFactory>(device_id, cuda_mem_limit, arena_extend_strategy);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, int device_id) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_CUDA(static_cast<OrtDevice::DeviceId>(device_id)));
  return nullptr;
}
