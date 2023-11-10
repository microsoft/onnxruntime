// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "contrib_ops/cuda/transformers/greedy_search.h"
#include "contrib_ops/cuda/transformers/generation_device_helper.h"
#include "contrib_ops/cuda/transformers/dump_cuda_tensor.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    GreedySearch,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)    // 'input_ids' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 1)    // 'max_length' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 2)    // 'min_length' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 3)    // 'repetition_penalty' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 6)    // 'custom_attention_mask' needs to be on CPU
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)  // 'sequences' output on CPU
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<MLFloat16>()}),
    GreedySearch);

transformers::CudaTensorConsoleDumper g_cuda_dumper_greedysearch;

GreedySearch::GreedySearch(const OpKernelInfo& info)
    : onnxruntime::contrib::transformers::GreedySearch(info) {
  SetDeviceHelpers(GenerationCudaDeviceHelper::AddToFeeds,
                   GenerationCudaDeviceHelper::TopK,
                   GenerationCudaDeviceHelper::DeviceCopy<float>,
                   GenerationCudaDeviceHelper::GreedySearchProcessLogits<float>,
                   GenerationCudaDeviceHelper::GreedySearchProcessLogits<MLFloat16>,
                   GenerationCudaDeviceHelper::InitGreedyState<float>,
                   GenerationCudaDeviceHelper::InitGreedyState<MLFloat16>);

#ifndef USE_ROCM
  SetDeviceHelpers_Cuda(GenerationCudaDeviceHelper::ReorderPastState);
#endif

  SetDeviceHelpers_Gpt(GenerationCudaDeviceHelper::UpdateGptFeeds<float>,
                       GenerationCudaDeviceHelper::UpdateGptFeeds<MLFloat16>);

  SetConsoleDumper(&g_cuda_dumper_greedysearch);

  cuda_device_prop_ = &reinterpret_cast<const CUDAExecutionProvider*>(info.GetExecutionProvider())->GetDeviceProp();

  cuda_device_arch_ = static_cast<const cudaDeviceProp*>(cuda_device_prop_)->major * 100 +
                      static_cast<const cudaDeviceProp*>(cuda_device_prop_)->minor * 10;
}

Status GreedySearch::ComputeInternal(OpKernelContext* context) const {
  return onnxruntime::contrib::transformers::GreedySearch::Compute(context);
}

Status GreedySearch::Compute(OpKernelContext* context) const {
  auto s = ComputeInternal(context);

  if (s.IsOK()) {
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDA error ", cudaGetErrorName(err), ":", cudaGetErrorString(err));
    }
  }

  return s;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
