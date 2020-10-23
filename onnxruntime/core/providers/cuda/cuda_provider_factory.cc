// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_provider_factory.h"
#include <atomic>
#include "core/graph/onnx_protobuf.h"
#include "cuda_execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"
#include "core/framework/bfc_arena.h"

using namespace onnxruntime;

namespace onnxruntime {

struct CUDAProviderFactory : IExecutionProviderFactory {
  CUDAProviderFactory(OrtDevice::DeviceId device_id,
                      size_t cuda_mem_limit = std::numeric_limits<size_t>::max(),
                      ArenaExtendStrategy arena_extend_strategy = ArenaExtendStrategy::kNextPowerOfTwo,
                      OrtCudnnConvAlgoSearch cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE,
                      bool do_copy_in_default_stream = true) 
      : device_id_(device_id), 
        cuda_mem_limit_(cuda_mem_limit), 
        arena_extend_strategy_(arena_extend_strategy),
        cudnn_conv_algo_search_(cudnn_conv_algo_search),
        do_copy_in_default_stream_(do_copy_in_default_stream) {}
  ~CUDAProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  OrtDevice::DeviceId device_id_;
  size_t cuda_mem_limit_;
  ArenaExtendStrategy arena_extend_strategy_;
  OrtCudnnConvAlgoSearch cudnn_conv_algo_search_;
  bool do_copy_in_default_stream_;
};

std::unique_ptr<IExecutionProvider> CUDAProviderFactory::CreateProvider() {
  CUDAExecutionProviderInfo info;
  info.device_id = device_id_;
  info.cuda_mem_limit = cuda_mem_limit_;
  info.arena_extend_strategy = arena_extend_strategy_;
  info.cudnn_conv_algo = cudnn_conv_algo_search_;
  info.do_copy_in_default_stream = do_copy_in_default_stream_;
  return onnxruntime::make_unique<CUDAExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CUDA(OrtDevice::DeviceId device_id,
                                                                               OrtCudnnConvAlgoSearch cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE,
                                                                               size_t cuda_mem_limit = std::numeric_limits<size_t>::max(),
                                                                               ArenaExtendStrategy arena_extend_strategy = ArenaExtendStrategy::kNextPowerOfTwo,
                                                                               bool do_copy_in_default_stream = true) {
  return std::make_shared<onnxruntime::CUDAProviderFactory>(device_id, cuda_mem_limit, arena_extend_strategy, cudnn_conv_algo_search, do_copy_in_default_stream);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, int device_id){
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_CUDA(static_cast<OrtDevice::DeviceId>(device_id)));
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::OrtSessionOptionsAppendExecutionProvider_CUDA,
                    _In_ OrtSessionOptions* options, _In_ OrtCUDAProviderOptions* cuda_options) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_CUDA(static_cast<OrtDevice::DeviceId>(cuda_options->device_id),
                                            cuda_options->cudnn_conv_algo_search, cuda_options->cuda_mem_limit, 
                                            static_cast<onnxruntime::ArenaExtendStrategy>(cuda_options->arena_extend_strategy),
                                            cuda_options->do_copy_in_default_stream));
  return nullptr;
}
