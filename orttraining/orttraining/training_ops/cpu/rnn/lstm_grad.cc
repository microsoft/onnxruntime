// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/rnn/lstm_grad.h"
#include "orttraining/training_ops/cpu/rnn/lstm_grad_compute.h"

namespace onnxruntime::contrib {

#define REGISTER_LSTMGRAD_KERNEL_TYPED(T)                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      LSTMGrad,                                                   \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      LSTMGrad<T>);

REGISTER_LSTMGRAD_KERNEL_TYPED(float)

template <typename T>
Status LSTMGrad<T>::Compute(OpKernelContext* context) const {
  const auto lstmgrad_inputs = lstm::LSTMGradInputs<T>(context, attributes_.num_directions, attributes_.hidden_size);
  auto lstmgrad_outputs = lstm::LSTMGradOutputs<T>(context, attributes_.num_directions, lstmgrad_inputs.shape.sequence_length,
                                                   lstmgrad_inputs.shape.batch_size, attributes_.hidden_size,
                                                   lstmgrad_inputs.shape.input_size);

  // Allocator in case we need to allocate memory for a nullptr input/output
  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  lstm::LSTMGradImpl<T> lstm_cell(lstmgrad_inputs.shape.sequence_length,
                                  lstmgrad_inputs.shape.batch_size,
                                  attributes_.hidden_size,
                                  lstmgrad_inputs.shape.input_size,
                                  context->GetOperatorThreadPool(),
                                  alloc);

  lstm_cell.ComputeGradient(lstmgrad_inputs, lstmgrad_outputs);

  return Status::OK();
}

}  // namespace onnxruntime::contrib
