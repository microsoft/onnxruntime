// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/cpu_provider_factory_creator.h"
#include "core/providers/cpu/cpu_provider_factory.h"

#include <memory>

#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

struct CpuProviderFactory : IExecutionProviderFactory {
  CpuProviderFactory(bool create_arena) : create_arena_(create_arena) {}
  ~CpuProviderFactory() override = default;
  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  bool create_arena_;
};

std::unique_ptr<IExecutionProvider> CpuProviderFactory::CreateProvider() {
  CPUExecutionProviderInfo info;
  info.create_arena = create_arena_;
  return std::make_unique<CPUExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CPU(int use_arena) {
  return std::make_shared<onnxruntime::CpuProviderFactory>(use_arena != 0);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_CPU, _In_ OrtSessionOptions* options, int use_arena) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_CPU(use_arena));
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::CreateCpuMemoryInfo, enum OrtAllocatorType type, enum OrtMemType mem_type,
                    _Outptr_ OrtMemoryInfo** out) {
  *out = new OrtMemoryInfo(onnxruntime::CPU, type, OrtDevice(), 0, mem_type);
  return nullptr;
}
