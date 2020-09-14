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
                      ArenaExtendStrategy arena_extend_strategy = ArenaExtendStrategy::kNextPowerOfTwo,
                      int cudnn_conv_algo_search = 1) 
      : device_id_(device_id), 
        cuda_mem_limit_(cuda_mem_limit), 
        arena_extend_strategy_(arena_extend_strategy),
        cudnn_conv_algo_search_(cudnn_conv_algo_search){}
  ~CUDAProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  OrtDevice::DeviceId device_id_;
  size_t cuda_mem_limit_;
  ArenaExtendStrategy arena_extend_strategy_;
  int cudnn_conv_algo_search_;
};

std::unique_ptr<IExecutionProvider> CUDAProviderFactory::CreateProvider() {
  CUDAExecutionProviderInfo info;
  info.device_id = device_id_;
  info.cuda_mem_limit = cuda_mem_limit_;
  info.arena_extend_strategy = arena_extend_strategy_;
  info.cudnn_conv_algo = cudnn_conv_algo_search_;
  return onnxruntime::make_unique<CUDAExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CUDA(OrtDevice::DeviceId device_id,
                                                                               int cudnn_conv_algo_search = 0,
                                                                               size_t cuda_mem_limit = std::numeric_limits<size_t>::max(),
                                                                               ArenaExtendStrategy arena_extend_strategy = ArenaExtendStrategy::kNextPowerOfTwo) {
  return std::make_shared<onnxruntime::CUDAProviderFactory>(device_id, cuda_mem_limit, arena_extend_strategy, cudnn_conv_algo_search);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, int device_id){
  return OrtSessionOptionsAppendExecutionProvider_CUDA_CONV_ALGO(options, device_id, 0);
}

//conv_algo parameter:
//      0: expensive exhaustive benchmarking using cudnnFindConvolutionForwardAlgorithmEx
//      1: lightweight heuristic based search using cudnnGetConvolutionForwardAlgorithm_v7
//      2: default algorithm using CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_CUDA_CONV_ALGO, _In_ OrtSessionOptions* options, int device_id, int conv_algo) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_CUDA(static_cast<OrtDevice::DeviceId>(device_id), conv_algo));
  return nullptr;
}
