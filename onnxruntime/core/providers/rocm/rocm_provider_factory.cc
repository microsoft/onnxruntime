// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/rocm_provider_factory_creator.h"
#include "core/providers/rocm/rocm_provider_factory.h"

#include <memory>

#include "gsl/gsl"

#include "core/common/make_unique.h"
#include "core/providers/rocm/rocm_execution_provider.h"
#include "core/providers/rocm/rocm_execution_provider_info.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {

struct HIPProviderFactory : IExecutionProviderFactory {
  HIPProviderFactory(const ROCMExecutionProviderInfo& info)
      : info_{info} {}
  ~HIPProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  ROCMExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> HIPProviderFactory::CreateProvider() {
  return onnxruntime::make_unique<ROCMExecutionProvider>(info_);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ROCM(const ROCMExecutionProviderInfo& info) {
  return std::make_shared<HIPProviderFactory>(info);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_ROCM, _In_ OrtSessionOptions* options, int device_id) {
  ROCMExecutionProviderInfo info{};
  info.device_id = gsl::narrow<OrtDevice::DeviceId>(device_id);

  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_ROCM(info));

  return nullptr;
}
