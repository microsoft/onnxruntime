// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "contrib_ops/cuda/transformers/beam_search.h"
#include "contrib_ops/cuda/transformers/generation_device_helper.h"
#include "contrib_ops/cuda/transformers/dump_cuda_tensor.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    BeamSearch,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)    // 'input_ids' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 1)    // 'max_length' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 2)    // 'min_length' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 3)    // 'num_beams' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 4)    // 'num_return_sequences' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 5)    // 'length_penalty' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 6)    // 'repetition_penalty' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 9)    // 'attention_mask' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 10)   // 'decoder_input_ids' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 11)   // 'logits_processor' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 12)   // 'input_images' needs to be on CPU
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)  // 'sequences' output on CPU
        .OutputMemoryType(OrtMemTypeCPUOutput, 1)  // 'sequences_scores' output on CPU
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<MLFloat16>()}),
    BeamSearch);

ONNX_OPERATOR_KERNEL_EX(
    WhisperBeamSearch,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)    // 'input_ids' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 1)    // 'max_length' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 2)    // 'min_length' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 3)    // 'num_beams' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 4)    // 'num_return_sequences' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 5)    // 'length_penalty' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 6)    // 'repetition_penalty' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 9)    // 'attention_mask' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 10)   // 'decoder_input_ids' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 11)   // 'logits_processor' needs to be on CPU
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)  // 'sequences' output on CPU
        .OutputMemoryType(OrtMemTypeCPUOutput, 1)  // 'sequences_scores' output on CPU
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<MLFloat16>()}),
    WhisperBeamSearch);

transformers::CudaTensorConsoleDumper g_cuda_dumper;

BeamSearch::BeamSearch(const OpKernelInfo& info)
    : onnxruntime::contrib::transformers::BeamSearch(info) {
  SetDeviceHelpers(GenerationCudaDeviceHelper::AddToFeeds,
                   GenerationCudaDeviceHelper::TopK,
                   GenerationCudaDeviceHelper::DeviceCopy<float>,
                   GenerationCudaDeviceHelper::DeviceCopy<int32_t>,
                   GenerationCudaDeviceHelper::ProcessLogits<float>,
                   GenerationCudaDeviceHelper::ProcessLogits<MLFloat16>,
                   GenerationCudaDeviceHelper::InitBeamState<float>,
                   GenerationCudaDeviceHelper::InitBeamState<MLFloat16>,
                   GenerationCudaDeviceHelper::CreateBeamScorer);

#ifndef USE_ROCM
  SetDeviceHelpers_Cuda(GenerationCudaDeviceHelper::ReorderPastState, GenerationCudaDeviceHelper::InitCacheIndir);
#endif

  SetDeviceHelpers_Gpt(GenerationCudaDeviceHelper::UpdateGptFeeds<float>,
                       GenerationCudaDeviceHelper::UpdateGptFeeds<MLFloat16>);

  SetDeviceHelpers_EncoderDecoder(GenerationCudaDeviceHelper::UpdateDecoderFeeds<float>,
                                  GenerationCudaDeviceHelper::UpdateDecoderFeeds<MLFloat16>,
                                  GenerationCudaDeviceHelper::ExpandBuffer<int32_t>,
                                  GenerationCudaDeviceHelper::ExpandBuffer<float>,
                                  GenerationCudaDeviceHelper::ExpandBuffer<MLFloat16>,
                                  GenerationCudaDeviceHelper::UpdateDecoderCrossQK,
                                  GenerationCudaDeviceHelper::FinalizeDecoderCrossQK);

  SetConsoleDumper(&g_cuda_dumper);

#ifndef USE_ROCM
  cuda_device_prop_ = &reinterpret_cast<const CUDAExecutionProvider*>(info.GetExecutionProvider())->GetDeviceProp();

  cuda_device_arch_ = static_cast<const cudaDeviceProp*>(cuda_device_prop_)->major * 100 +
                      static_cast<const cudaDeviceProp*>(cuda_device_prop_)->minor * 10;
#endif
}

Status BeamSearch::ComputeInternal(OpKernelContext* context) const {
  return onnxruntime::contrib::transformers::BeamSearch::Compute(context);
}

Status BeamSearch::Compute(OpKernelContext* context) const {
  auto s = ComputeInternal(context);

  if (s.IsOK()) {
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDA error ", cudaGetErrorName(err), ":", cudaGetErrorString(err));
    }
  }

  return s;
}

WhisperBeamSearch::WhisperBeamSearch(const OpKernelInfo& info)
    : onnxruntime::contrib::transformers::WhisperBeamSearch(info) {
  SetDeviceHelpers(GenerationCudaDeviceHelper::AddToFeeds,
                   GenerationCudaDeviceHelper::TopK,
                   GenerationCudaDeviceHelper::DeviceCopy<float>,
                   GenerationCudaDeviceHelper::DeviceCopy<int32_t>,
                   GenerationCudaDeviceHelper::ProcessLogits<float>,
                   GenerationCudaDeviceHelper::ProcessLogits<MLFloat16>,
                   GenerationCudaDeviceHelper::InitBeamState<float>,
                   GenerationCudaDeviceHelper::InitBeamState<MLFloat16>,
                   GenerationCudaDeviceHelper::CreateBeamScorer);

#ifndef USE_ROCM
  SetDeviceHelpers_Cuda(GenerationCudaDeviceHelper::ReorderPastState, GenerationCudaDeviceHelper::InitCacheIndir);
#endif

  SetDeviceHelpers_Gpt(GenerationCudaDeviceHelper::UpdateGptFeeds<float>,
                       GenerationCudaDeviceHelper::UpdateGptFeeds<MLFloat16>);

  SetDeviceHelpers_EncoderDecoder(GenerationCudaDeviceHelper::UpdateDecoderFeeds<float>,
                                  GenerationCudaDeviceHelper::UpdateDecoderFeeds<MLFloat16>,
                                  GenerationCudaDeviceHelper::ExpandBuffer<int32_t>,
                                  GenerationCudaDeviceHelper::ExpandBuffer<float>,
                                  GenerationCudaDeviceHelper::ExpandBuffer<MLFloat16>,
                                  GenerationCudaDeviceHelper::UpdateDecoderCrossQK,
                                  GenerationCudaDeviceHelper::FinalizeDecoderCrossQK);

  SetConsoleDumper(&g_cuda_dumper);

#ifndef USE_ROCM
  cuda_device_prop_ = &reinterpret_cast<const CUDAExecutionProvider*>(info.GetExecutionProvider())->GetDeviceProp();

  cuda_device_arch_ = static_cast<const cudaDeviceProp*>(cuda_device_prop_)->major * 100 +
                      static_cast<const cudaDeviceProp*>(cuda_device_prop_)->minor * 10;
#endif
}

Status WhisperBeamSearch::ComputeInternal(OpKernelContext* context) const {
  return onnxruntime::contrib::transformers::WhisperBeamSearch::Compute(context);
}

Status WhisperBeamSearch::Compute(OpKernelContext* context) const {
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
