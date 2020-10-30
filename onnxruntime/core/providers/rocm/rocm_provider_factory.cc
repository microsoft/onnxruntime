// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/rocm_provider_factory.h"
#include <atomic>
#include "core/graph/onnx_protobuf.h"
#include "rocm_execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/framework/bfc_arena.h"

using namespace onnxruntime;

namespace onnxruntime {

struct HIPProviderFactory : IExecutionProviderFactory {
  HIPProviderFactory(OrtDevice::DeviceId device_id,
                      size_t hip_mem_limit = std::numeric_limits<size_t>::max(),
                      ArenaExtendStrategy arena_extend_strategy = ArenaExtendStrategy::kNextPowerOfTwo) 
      : device_id_(device_id), 
        hip_mem_limit_(hip_mem_limit), 
        arena_extend_strategy_(arena_extend_strategy) {}
  ~HIPProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  OrtDevice::DeviceId device_id_;
  size_t hip_mem_limit_;
  ArenaExtendStrategy arena_extend_strategy_;
};

std::unique_ptr<IExecutionProvider> HIPProviderFactory::CreateProvider() {
  ROCMExecutionProviderInfo info;
  info.device_id = device_id_;
  info.hip_mem_limit = hip_mem_limit_;
  info.arena_extend_strategy = arena_extend_strategy_;
  return onnxruntime::make_unique<ROCMExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ROCM(OrtDevice::DeviceId device_id,
                                                                               size_t hip_mem_limit = std::numeric_limits<size_t>::max(),
                                                                               ArenaExtendStrategy arena_extend_strategy = ArenaExtendStrategy::kNextPowerOfTwo) {
  return std::make_shared<HIPProviderFactory>(device_id, hip_mem_limit, arena_extend_strategy);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_ROCM, _In_ OrtSessionOptions* options, int device_id) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_ROCM(static_cast<OrtDevice::DeviceId>(device_id)));
  return nullptr;
}
