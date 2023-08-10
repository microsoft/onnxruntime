// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/rnn/gru_grad.h"
#include "orttraining/training_ops/cpu/rnn/gru_grad_compute.h"

namespace onnxruntime::contrib {

#define REGISTER_GRUGRAD_KERNEL_TYPED(T)                          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      GRUGrad,                                                    \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GRUGrad<T>);

REGISTER_GRUGRAD_KERNEL_TYPED(float)

template <typename T>
Status GRUGrad<T>::Compute(OpKernelContext* context) const {
  const auto grugrad_inputs = gru::GRUGradInputs<T>(context, attributes_.num_directions, attributes_.hidden_size);
  auto grugrad_outputs = gru::GRUGradOutputs<T>(context, attributes_.num_directions, grugrad_inputs.shape.sequence_length,
                                                grugrad_inputs.shape.batch_size, attributes_.hidden_size,
                                                grugrad_inputs.shape.input_size);

  // Allocator in case we need to allocate memory for a nullptr input/output
  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  gru::GRUGradImpl<T> gru_cell(grugrad_inputs.shape.sequence_length,
                               grugrad_inputs.shape.batch_size,
                               attributes_.hidden_size,
                               grugrad_inputs.shape.input_size,
                               attributes_.linear_before_reset,
                               context->GetOperatorThreadPool(),
                               alloc);

  gru_cell.ComputeGradient(grugrad_inputs, grugrad_outputs);

  return Status::OK();
}

}  // namespace onnxruntime::contrib
