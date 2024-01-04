// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/rnn/gru.h"

#include "core/providers/cpu/rnn/deep_cpu_gru.h"

namespace onnxruntime::contrib {

#define REGISTER_GRUTRAINING_KERNEL_TYPED(T)                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      GRUTraining,                                                \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GRUTraining<T>);

REGISTER_GRUTRAINING_KERNEL_TYPED(float)

template <typename T>
Status GRUTraining<T>::Compute(OpKernelContext* context) const {
  const auto gru_inputs = gru::GRUInputs<T>(context, attributes_.num_directions, attributes_.hidden_size);
  auto gru_outputs = gru::GRUOutputs<T>(context, attributes_.num_directions, gru_inputs.shape.sequence_length,
                                        gru_inputs.shape.batch_size, attributes_.hidden_size);

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  detail::UniDirectionalGru<T> gru(alloc,
                                   gru_inputs.shape.sequence_length,
                                   gru_inputs.shape.batch_size,
                                   gru_inputs.shape.input_size,
                                   attributes_.hidden_size,
                                   attributes_.linear_before_reset,
                                   attributes_.direction,
                                   gru_inputs.bias,
                                   gru_inputs.initial_hidden_state,
                                   attributes_.activation_funcs.Entries()[0],
                                   attributes_.activation_funcs.Entries()[1],
                                   attributes_.clip,
                                   context->GetOperatorThreadPool(),
                                   true /*training_mode*/);
  gru.Compute(gru_inputs.input,
              gru_inputs.sequence_lengths,
              attributes_.num_directions,
              gru_inputs.weights,
              gru_inputs.recurrence_weights_zr,
              gru_inputs.recurrence_weights_h,
              gru_outputs.all_hidden_states,
              gru_outputs.final_hidden_state,
              gru_outputs.zrh);

  return Status::OK();
}

}  // namespace onnxruntime::contrib
