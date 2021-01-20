// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sequence_pooling.h"

namespace onnxruntime {
namespace contrib {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      SequencePooling,                                            \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SequencePooling<T>);

REGISTER_KERNEL_TYPED(float)

template <typename T>
SequencePooling<T>::SequencePooling(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
}

template <typename T>
Status SequencePooling<T>::Compute(OpKernelContext* context) const {
  // get inputs tensors and data
  const Tensor* input_tensor = context->Input<Tensor>(0);
  const T* input_data = input_tensor->template Data<T>();
  const Tensor* sentence_lengthes_tensor = context->Input<Tensor>(1);
  //const int32_t* sentence_lengthes_data = sentence_lengthes_tensor->template Data<int32_t>();

  // shape info
  const auto& input_shape = input_tensor->Shape().GetDims();
  const int batch_size = static_cast<int>(input_shape[0]);
  //const int sequence_length_for_split = static_cast<int>(input_shape[1]);
  const int hidden_size = static_cast<int>(input_shape[2]);

  const auto& sentence_lengthes_shape = sentence_lengthes_tensor->Shape().GetDims();
  const int num_sequences = static_cast<int>(sentence_lengthes_shape[1]);

  // initialize outputs
  TensorShape output_shape({batch_size, num_sequences, hidden_size});
  Tensor* output_tensor(context->Output(0, output_shape));
  T* output_data = output_tensor->template MutableData<T>();

  Tensor* masks_tensor(context->Output(1, sentence_lengthes_tensor->Shape()));
  T* masks_data = masks_tensor->template MutableData<T>();

  // assign values to outputs
  memcpy(output_data, input_data, batch_size * num_sequences * hidden_size * sizeof(T));
  for (int i = 0; i < batch_size * num_sequences; i++) {
    *masks_data++ = 1;
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
