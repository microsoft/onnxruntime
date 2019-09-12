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
      Attention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)


template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : CudaKernel(info) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = static_cast<int>(num_heads);

  int64_t head_size = 0;
  ORT_ENFORCE(info.GetAttr("head_size", &head_size).IsOK() && head_size > 0);
  head_size_ = static_cast<int>(head_size);

  int64_t batch_size = 0;
  ORT_ENFORCE(info.GetAttr("batch_size", &batch_size).IsOK() && batch_size > 0);
  batch_size_ = static_cast<int>(batch_size);

  int64_t sequence_length = 0;
  ORT_ENFORCE(info.GetAttr("sequence_length", &sequence_length).IsOK() && sequence_length > 0);
  sequence_length_ = static_cast<int>(sequence_length);

  const size_t element_size = sizeof(T);
  size_t wordSpaceSize = getAttentionWorkspaceSize(element_size, batch_size_, num_heads_, head_size_, sequence_length_);
  word_space_ = GetScratchBuffer<void>(wordSpaceSize);
}

template <typename T>
Status Attention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);

  // Input dimensions: (B, S, 3 * N * H)
  // Output dimensions: (B, S, N * H)

  const auto dims = input->Shape().GetDims();
  if (dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "input is expected to have 3 dimensions, got ", dims.size());
  }

  if (static_cast<int>(dims[0]) != batch_size_ 
      || static_cast<int>(dims[1]) != sequence_length_ 
      || static_cast<int>(dims[2]) != 3 * num_heads_ * head_size_)
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
  if (mask_dims[0] != batch_size_ || mask_dims[1] != sequence_length_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "mask dimensions does not match input ");
  }

  std::vector<int64_t> out_dims;
  out_dims.reserve(3);
  out_dims.push_back(dims[0]);
  out_dims.push_back(dims[1]);
  out_dims.push_back(dims[2] / 3);
  TensorShape output_shape(out_dims);
  Tensor* output = context->Output(0, output_shape);

  cublasHandle_t cublas = CublasHandle();
  const size_t element_size = sizeof(T);
  launchAttentionKernel(
      input->template Data<T>(),
      mask->template Data<int>(),
      output->template MutableData<T>(),
      batch_size_,
      sequence_length_,
      num_heads_,
      head_size_,
      word_space_.get(),
      cublas,
      element_size
      );

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
