// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/framework/tensorprotoutils.h"
#include "onnx/defs/tensor_proto_util.h"
#include "sequence_pooling.h"
#include "sequence_pooling_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      SequencePooling,                                            \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SequencePooling<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

using namespace ONNX_NAMESPACE;

template <typename T>
SequencePooling<T>::SequencePooling(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
}

template <typename T>
Status SequencePooling<T>::ComputeInternal(OpKernelContext* ctx) const {
  // get inputs tensors and data
  const Tensor* input_tensor = context->Input<Tensor>(0);
  const Tensor* sentence_lengthes_tensor = context->Input<Tensor>(1);
  const int64_t* sentence_lengthes_data = sentence_lengthes_tensor->template Data<int64_t>();

  // shape info
  const auto& input_shape = input_tensor->Shape().GetDims();
  const int batch_size = static_cast<int>(input_shape[0]);
  const int sequence_length_for_split = static_cast<int>(input_shape[1]);
  const int hidden_size = static_cast<int>(input_shape[2]);
  const auto& sentence_lengthes_shape = sentence_lengthes_tensor->Shape().GetDims();
  const int num_sequences = static_cast<int>(sentence_lengthes_shape[1]);

  // check inputs
  int sentence_lengthes_prefixsum[batch_size * num_sequences];
  for (int batch = 0; batch < batch_size; ++batch) {
    const int offset(batch * num_sequences);
    std::partial_sum(sentence_lengthes_data + offset, sentence_lengthes_data + offset + num_sequences, sentence_length_prefixsum.begin() + offset);
    ORT_ENFORCE(sentence_length_prefixsum[offset + num_sequences - 1] == sequence_length_for_split);
  }

  // initialize outputs
  TensorShape output_shape({batch_size, num_sequences, hidden_size});
  Tensor* output_tensor(context->Output(0, output_shape));
  Tensor* masks_tensor(context->Output(1, sentence_lengthes_tensor->Shape()));

  for (int i = 0; i < batch_size * num_sequences; i++) {
    *masks_data++ = 1;
  }

  size_t element_size = sizeof(T);

  if (!LaunchSequencePoolingKernel(
          output_tensor->template MutableData<T>(),
          input->template Data<T>(),
          sentence_lengthes_prefixsum,
          batch_size,
          hidden_size,
          num_sequences,
          element_size)) {
    // Get last error to reset it to cudaSuccess.
    CUDA_CALL(cudaGetLastError());
    return Status(common::ONNXRUNTIME, common::FAIL);
  }

  return Status::OK();
}

}  //namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
