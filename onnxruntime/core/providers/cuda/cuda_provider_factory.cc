// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_provider_factory_creator.h"
#include "core/providers/cuda/cuda_provider_factory.h"

#include <memory>

#include "gsl/gsl"

#include "core/common/make_unique.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"

using namespace onnxruntime;

namespace onnxruntime {

struct CUDAProviderFactory : IExecutionProviderFactory {
  CUDAProviderFactory(const CUDAExecutionProviderInfo& info)
      : info_{info} {}
  ~CUDAProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  CUDAExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> CUDAProviderFactory::CreateProvider() {
  return onnxruntime::make_unique<CUDAExecutionProvider>(info_);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CUDA(const CUDAExecutionProviderInfo& info) {
  return std::make_shared<onnxruntime::CUDAProviderFactory>(info);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, int device_id) {
  CUDAExecutionProviderInfo info{};
  info.device_id = gsl::narrow<OrtDevice::DeviceId>(device_id);

  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_CUDA(info));

  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_CUDA,
                    _In_ OrtSessionOptions* options, _In_ const OrtCUDAProviderOptions* cuda_options) {
  CUDAExecutionProviderInfo info{};
  info.device_id = gsl::narrow<OrtDevice::DeviceId>(cuda_options->device_id);
  info.cuda_mem_limit = cuda_options->cuda_mem_limit;
  info.arena_extend_strategy = static_cast<onnxruntime::ArenaExtendStrategy>(cuda_options->arena_extend_strategy);
  info.cudnn_conv_algo_search = cuda_options->cudnn_conv_algo_search;
  info.do_copy_in_default_stream = cuda_options->do_copy_in_default_stream;

  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_CUDA(info));

  return nullptr;
}
