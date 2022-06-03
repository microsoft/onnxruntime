// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/cpu_provider_factory_creator.h"
#include "core/providers/cpu/cpu_provider_factory.h"

#include <memory>

#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

struct CpuProviderFactory : IExecutionProviderFactory {
  CpuProviderFactory(const CPUExecutionProviderInfo& info) : info(info) {}
  ~CpuProviderFactory() override = default;
  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  CPUExecutionProviderInfo info;
};

std::unique_ptr<IExecutionProvider> CpuProviderFactory::CreateProvider() {
  return std::make_unique<CPUExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CPU(const CPUExecutionProviderInfo& info) {
  return std::make_shared<onnxruntime::CpuProviderFactory>(info);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_CPU, _In_ OrtSessionOptions* options, int use_arena) {
  const bool use_fixed_point_requant_on_arm64 =
      options->value.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigFixedPointRequantOnARM64, "0") == "1";
  options->provider_factories.push_back(
      onnxruntime::CreateExecutionProviderFactory_CPU(
          onnxruntime::CPUExecutionProviderInfo(use_arena, use_fixed_point_requant_on_arm64)));
  return nullptr;
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26409)
#endif
ORT_API_STATUS_IMPL(OrtApis::CreateCpuMemoryInfo, enum OrtAllocatorType type, enum OrtMemType mem_type,
                    _Outptr_ OrtMemoryInfo** out) {
  *out = new OrtMemoryInfo(onnxruntime::CPU, type, OrtDevice(), 0, mem_type);
  return nullptr;
}
