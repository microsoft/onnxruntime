// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "ngram_repeat_block.h"
#include "ngram_repeat_block_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    NGramRepeatBlock,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("Tid", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NGramRepeatBlock);

using namespace ONNX_NAMESPACE;

NGramRepeatBlock::NGramRepeatBlock(const OpKernelInfo& info) : CudaKernel(info) {
  ORT_ENFORCE(info.GetAttr<int64_t>("ngram_size", &ngram_size_).IsOK());
}

Status NGramRepeatBlock::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input_ids = context->Input<Tensor>(0);
  const Tensor* scores = context->Input<Tensor>(1);
  Tensor* output = context->Output(0, scores->Shape());

  const auto* scores_source = static_cast<const float*>(scores->DataRaw());
  auto* scores_target = static_cast<float*>(output->MutableDataRaw());
  if (scores_source != scores_target) {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(scores_target, scores_source, scores->Shape().Size() * sizeof(float), cudaMemcpyDeviceToDevice, Stream()));
  }

  const auto& input_ids_dims = input_ids->Shape().GetDims();
  const auto& scores_dims = scores->Shape().GetDims();
  ORT_ENFORCE(input_ids_dims.size() == 2);
  ORT_ENFORCE(scores_dims.size() == 2);
  int64_t batch_size = input_ids_dims[0];
  int64_t cur_len = input_ids_dims[1];
  ORT_ENFORCE(scores_dims[0] == batch_size);
  int64_t vocab_size = scores_dims[1];

  if (cur_len + 1 < ngram_size_ || ngram_size_ <= 0) {
    return Status::OK();
  }

  const auto* input_ids_data = static_cast<const int64_t*>(input_ids->DataRaw(input_ids->DataType()));

  NGramRepeatBlockImpl(
      Stream(),
      input_ids_data,
      scores_target,
      gsl::narrow_cast<int>(batch_size),
      gsl::narrow_cast<int>(cur_len - 1),
      gsl::narrow_cast<int>(cur_len),
      gsl::narrow_cast<int>(vocab_size),
      gsl::narrow_cast<int>(1),
      gsl::narrow_cast<int>(ngram_size_));

  return Status::OK();
}

}  //namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
