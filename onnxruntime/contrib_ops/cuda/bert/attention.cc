// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention.h"
#include "core/framework/tensorprotoutils.h"
#include "attention_impl.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Attention,                                                  \
      kOnnxDomain,                                                \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Attention);

REGISTER_KERNEL_TYPED(float)

Attention::Attention(const OpKernelInfo& info) : CudaKernel(info) {
  int64_t numHeads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &numHeads).IsOK() && numHeads > 0);
  numHeads_ = static_cast<int>(numHeads);

  int64_t headSize = 0;
  ORT_ENFORCE(info.GetAttr("head_size", &headSize).IsOK() && headSize > 0);
  headSize_ = static_cast<int>(headSize);

  int64_t batchSize = 0;
  ORT_ENFORCE(info.GetAttr("batch_size", &batchSize).IsOK() && batchSize > 0);
  batchSize_ = static_cast<int>(batchSize);

  int64_t sequenceLength = 0;
  ORT_ENFORCE(info.GetAttr("sequence_length", &sequenceLength).IsOK() && sequenceLength > 0);
  sequenceLength_ = static_cast<int>(sequenceLength);

  //TODO: derive word size from input tensor type
  const size_t wordSize = 4;
  size_t wordSpaceSize = getAttentionWorkspaceSize(wordSize, batchSize_, numHeads_, headSize_, sequenceLength_);
  workSpace_ = GetScratchBuffer<void>(wordSpaceSize);
}

Status Attention::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);

  // Output has only one dimension: (B * S * N * H)
  const int outputSize = batchSize_ * sequenceLength_ * numHeads_ * headSize_;

  // Input has only one dimension: (B * S * 3 * N * H)
  const auto dims = input->Shape().GetDims();
  if (dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "input is expected to have 1 dimension, got ", dims.size());
  }

  if (static_cast<int>(dims[0]) != 3 * outputSize)
  {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "input dimension does not match with node attributes");
  }

  // Attention mask (1 or 0) for each word in each sequence: shape (B, S)
  const Tensor* mask = context->Input<Tensor>(1);
  const auto mask_dims = mask->Shape().GetDims();
  if (mask_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "mask is expected to have 2 dimensions, got ", mask_dims.size());
  }
  if (mask_dims[0] != batchSize_ || mask_dims[1] != sequenceLength_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "mask dimensions does not match input ");
  }

  std::vector<int64_t> out_dims(1, outputSize);
  TensorShape outputShape(out_dims);
  Tensor* output = context->Output(0, outputShape);

  cublasHandle_t cublas = CublasHandle();

  launchAttentionKernel(
      input->template Data<float>(),
      mask->template Data<int>(),
      output->template MutableData<float>(),
      batchSize_,
      sequenceLength_,
      numHeads_,
      headSize_,
      workSpace_.get(),
      cublas
      );

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
