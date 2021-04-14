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
  info.gpu_mem_limit = cuda_options->gpu_mem_limit;
  info.arena_extend_strategy = static_cast<onnxruntime::ArenaExtendStrategy>(cuda_options->arena_extend_strategy);
  info.cudnn_conv_algo_search = cuda_options->cudnn_conv_algo_search;
  info.do_copy_in_default_stream = cuda_options->do_copy_in_default_stream;
  info.has_user_compute_stream = cuda_options->has_user_compute_stream;
  info.user_compute_stream = cuda_options->user_compute_stream;
  info.default_memory_arena_cfg = cuda_options->default_memory_arena_cfg;
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_CUDA(info));

  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::SetCurrentGpuDeviceId, _In_ int device_id) {
  int num_devices;
  auto cuda_err = cudaGetDeviceCount(&num_devices);
  if (cuda_err != cudaSuccess) {
    return CreateStatus(ORT_FAIL, "Failed to set device id since cudaGetDeviceCount failed.");
  }

  if (device_id >= num_devices) {
    std::ostringstream ostr;
    ostr << "Invalid device id. Device id should be less than total number of devices (" << num_devices << ")";
    return CreateStatus(ORT_INVALID_ARGUMENT, ostr.str().c_str());
  }

  cuda_err = cudaSetDevice(device_id);
  if (cuda_err != cudaSuccess) {
    return CreateStatus(ORT_FAIL, "Failed to set device id.");
  }
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::GetCurrentGpuDeviceId, _In_ int* device_id) {
  auto cuda_err = cudaGetDevice(device_id);
  if (cuda_err != cudaSuccess) {
    return CreateStatus(ORT_FAIL, "Failed to get device id.");
  }
  return nullptr;
}
