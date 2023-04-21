// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/rocm_execution_provider.h"
#include "contrib_ops/rocm/transformers/beam_search.h"
#include "contrib_ops/rocm/transformers/dump_rocm_tensor.h"
#include "contrib_ops/rocm/transformers/generation_device_helper.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

ONNX_OPERATOR_KERNEL_EX(
    BeamSearch,
    kMSDomain,
    1,
    kRocmExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)    // 'input_ids' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 1)    // 'max_length' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 2)    // 'min_length' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 3)    // 'num_beams' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 4)    // 'num_return_sequences' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 5)    // 'length_penalty' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 6)    // 'repetition_penalty' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 9)    // 'attention_mask' needs to be on CPU
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)  // 'sequences' output on CPU
        .OutputMemoryType(OrtMemTypeCPUOutput, 1)  // 'sequences_scores' output on CPU
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<MLFloat16>()}),
    BeamSearch);

transformers::HipTensorConsoleDumper g_rocm_dumper;

BeamSearch::BeamSearch(const OpKernelInfo& info)
    : onnxruntime::contrib::transformers::BeamSearch(info) {
  SetDeviceHelpers(GenerationCudaDeviceHelper::ReorderPastState,
                   GenerationCudaDeviceHelper::AddToFeeds,
                   GenerationCudaDeviceHelper::TopK,
                   GenerationCudaDeviceHelper::DeviceCopy<float>,
                   GenerationCudaDeviceHelper::DeviceCopy<int32_t>,
                   GenerationCudaDeviceHelper::ProcessLogits<float>,
                   GenerationCudaDeviceHelper::ProcessLogits<MLFloat16>,
                   GenerationCudaDeviceHelper::InitBeamState<float>,
                   GenerationCudaDeviceHelper::InitBeamState<MLFloat16>);

  SetDeviceHelpers_Gpt(GenerationCudaDeviceHelper::UpdateGptFeeds<float>,
                       GenerationCudaDeviceHelper::UpdateGptFeeds<MLFloat16>);

  SetDeviceHelpers_EncoderDecoder(GenerationCudaDeviceHelper::UpdateDecoderFeeds<float>,
                                  GenerationCudaDeviceHelper::UpdateDecoderFeeds<MLFloat16>,
                                  GenerationCudaDeviceHelper::ExpandBuffer<int32_t>,
                                  GenerationCudaDeviceHelper::ExpandBuffer<float>,
                                  GenerationCudaDeviceHelper::ExpandBuffer<MLFloat16>);

  SetConsoleDumper(&g_rocm_dumper);
}

Status BeamSearch::ComputeInternal(OpKernelContext* context) const {
  return onnxruntime::contrib::transformers::BeamSearch::Compute(context);
}

Status BeamSearch::Compute(OpKernelContext* context) const {
  auto s = ComputeInternal(context);

  if (s.IsOK()) {
    auto err = hipGetLastError();
    if (err != hipSuccess) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "ROCM error ", hipGetErrorName(err), ":", hipGetErrorString(err));
    }
  }

  return s;
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
