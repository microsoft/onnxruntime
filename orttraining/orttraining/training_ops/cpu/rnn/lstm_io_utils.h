// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"

namespace onnxruntime::lstm {

struct LSTMAttributes {
  LSTMAttributes(const OpKernelInfo& info);

  LSTMAttributes() = delete;

  rnn::detail::Direction direction;
  int num_directions;
  rnn::detail::ActivationFuncs activation_funcs;
  float clip;
  int input_forget;
  int hidden_size;
};

struct InputShape {
  int sequence_length;
  int batch_size;
  int input_size;
};

template <typename T>
struct LSTMInputs {
  LSTMInputs(OpKernelContext* context, const int directions, const int hidden_size);

  LSTMInputs() = delete;

  gsl::span<const T> input;
  InputShape shape;
  rnn::detail::GemmWeights<T> weights;
  rnn::detail::GemmWeights<T> recurrence_weights;
  gsl::span<const T> bias;
  gsl::span<const int> sequence_lengths;
  gsl::span<const T> initial_hidden_state;
  gsl::span<const T> initial_cell_state;
  gsl::span<const T> peephole_weights;
};

template <typename T>
struct LSTMOutputs {
  LSTMOutputs(OpKernelContext* context, const int directions, const int sequence_length,
              const int batch_size, const int hidden_size);

  LSTMOutputs() = delete;

  gsl::span<T> all_hidden_states;
  gsl::span<T> final_hidden_state;
  gsl::span<T> final_cell_state;
  gsl::span<T> all_cell_states;
  gsl::span<T> iofc;

 private:
  IAllocatorUniquePtr<T> hall_ptr_;
  IAllocatorUniquePtr<T> h_final_ptr_;
  IAllocatorUniquePtr<T> c_final_ptr_;
  IAllocatorUniquePtr<T> call_ptr_;
  IAllocatorUniquePtr<T> iofc_ptr_;
};

template <typename T>
struct LSTMGradInputs {
  LSTMGradInputs(OpKernelContext* context, const int directions, const int hidden_size);

  LSTMGradInputs() = delete;

  gsl::span<const T> input;
  InputShape shape;
  gsl::span<const T> weights;
  gsl::span<const T> recurrence_weights;
  gsl::span<const int> sequence_lengths;
  gsl::span<const T> initial_hidden_state;
  gsl::span<const T> initial_cell_state;
  gsl::span<const T> all_hidden_states;
  gsl::span<const T> all_cell_states;
  gsl::span<const T> iofc;
  gsl::span<const T> grad_all_hidden_states;
  gsl::span<const T> grad_final_hidden_state;
  gsl::span<const T> grad_final_cell_state;

 private:
  IAllocatorUniquePtr<T> initial_cell_state_ptr_;
  IAllocatorUniquePtr<T> initial_hidden_state_ptr_;
  IAllocatorUniquePtr<T> grad_final_cell_state_ptr_;
  IAllocatorUniquePtr<T> grad_final_hidden_state_ptr_;
};

template <typename T>
struct LSTMGradOutputs {
  LSTMGradOutputs(OpKernelContext* context, const int directions, const int sequence_length,
                  const int batch_size, const int hidden_size, const int input_size);

  LSTMGradOutputs() = delete;

  gsl::span<T> grad_input;
  gsl::span<T> grad_weights;
  gsl::span<T> grad_recurrence_weights;
  gsl::span<T> grad_bias;
  gsl::span<T> grad_initial_cell_state;
  gsl::span<T> grad_initial_hidden_state;
  gsl::span<T> grad_peephole_weights;

 private:
  IAllocatorUniquePtr<T> grad_initial_cell_state_ptr_;
  IAllocatorUniquePtr<T> grad_initial_hidden_state_ptr_;
};

}  // namespace onnxruntime::lstm
