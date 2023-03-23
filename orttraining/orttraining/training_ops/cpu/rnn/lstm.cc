// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/rnn/lstm.h"
#include "core/providers/cpu/rnn/uni_directional_lstm.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"

namespace onnxruntime::contrib {

#define REGISTER_LSTMTRAINING_KERNEL_TYPED(T)                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      LSTMTraining,                                               \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      LSTMTraining<T>);

REGISTER_LSTMTRAINING_KERNEL_TYPED(float)

template <typename T>
Status LSTMTraining<T>::Compute(OpKernelContext* context) const {
  const auto lstm_inputs = lstm::LSTMInputs<T>(context, attributes_.num_directions, attributes_.hidden_size);
  auto lstm_outputs = lstm::LSTMOutputs<T>(context, attributes_.num_directions, lstm_inputs.shape.sequence_length,
                                           lstm_inputs.shape.batch_size, attributes_.hidden_size);

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  lstm::UniDirectionalLstm<T> lstm(alloc,
                                   context->Logger(),
                                   lstm_inputs.shape.sequence_length,
                                   lstm_inputs.shape.batch_size,
                                   lstm_inputs.shape.input_size,
                                   attributes_.hidden_size,
                                   attributes_.direction,
                                   attributes_.input_forget,
                                   lstm_inputs.bias,
                                   lstm_inputs.peephole_weights,
                                   lstm_inputs.initial_hidden_state,
                                   lstm_inputs.initial_cell_state,
                                   attributes_.activation_funcs.Entries()[0],
                                   attributes_.activation_funcs.Entries()[1],
                                   attributes_.activation_funcs.Entries()[2],
                                   attributes_.clip,
                                   context->GetOperatorThreadPool(),
                                   true);

  lstm.Compute(lstm_inputs.input,
               lstm_inputs.sequence_lengths,
               attributes_.num_directions,
               lstm_inputs.weights,
               lstm_inputs.recurrence_weights,
               lstm_outputs.all_hidden_states,
               lstm_outputs.final_hidden_state,
               lstm_outputs.final_cell_state,
               lstm_outputs.all_cell_states,
               lstm_outputs.iofc);

  return Status::OK();
}

}  // namespace onnxruntime::contrib
