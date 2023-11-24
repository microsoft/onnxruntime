// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"

namespace onnxruntime::gru {

struct GRUAttributes {
  GRUAttributes(const OpKernelInfo& info);

  GRUAttributes() = delete;

  rnn::detail::Direction direction;
  int num_directions;
  rnn::detail::ActivationFuncs activation_funcs;
  float clip;
  int linear_before_reset;
  int hidden_size;
};

struct InputShape {
  int sequence_length;
  int batch_size;
  int input_size;
};

template <typename T>
struct GRUInputs {
  GRUInputs(OpKernelContext* context, const int directions, const int hidden_size);

  GRUInputs() = delete;

  gsl::span<const T> input;
  InputShape shape;
  rnn::detail::GemmWeights<T> weights;
  rnn::detail::GemmWeights<T> recurrence_weights_zr;
  rnn::detail::GemmWeights<T> recurrence_weights_h;
  gsl::span<const T> bias;
  gsl::span<const int> sequence_lengths;
  gsl::span<const T> initial_hidden_state;
};

template <typename T>
struct GRUOutputs {
  GRUOutputs(OpKernelContext* context, const int directions, const int sequence_length,
             const int batch_size, const int hidden_size);

  GRUOutputs() = delete;

  gsl::span<T> all_hidden_states;
  gsl::span<T> final_hidden_state;
  gsl::span<T> zrh;

 private:
  IAllocatorUniquePtr<T> hall_ptr_;
  IAllocatorUniquePtr<T> h_final_ptr_;
  IAllocatorUniquePtr<T> zrh_ptr_;
};

template <typename T>
struct GRUGradInputs {
  GRUGradInputs(OpKernelContext* context, const int directions, const int hidden_size);

  GRUGradInputs() = delete;

  gsl::span<const T> input;
  InputShape shape;
  gsl::span<const T> weights;
  gsl::span<const T> recurrence_weights;
  gsl::span<const T> bias;
  gsl::span<const int> sequence_lengths;
  gsl::span<const T> initial_hidden_state;
  gsl::span<const T> all_hidden_states;
  gsl::span<const T> zrh;
  gsl::span<const T> grad_all_hidden_states;
  gsl::span<const T> grad_final_hidden_state;

 private:
  IAllocatorUniquePtr<T> initial_hidden_state_ptr_;
  IAllocatorUniquePtr<T> grad_final_hidden_state_ptr_;
};

template <typename T>
struct GRUGradOutputs {
  GRUGradOutputs(OpKernelContext* context, const int directions, const int sequence_length,
                 const int batch_size, const int hidden_size, const int input_size);

  GRUGradOutputs() = delete;

  gsl::span<T> grad_input;
  gsl::span<T> grad_weights;
  gsl::span<T> grad_recurrence_weights;
  gsl::span<T> grad_bias;
  gsl::span<T> grad_initial_hidden_state;

 private:
  IAllocatorUniquePtr<T> grad_initial_hidden_state_ptr_;
};

}  // namespace onnxruntime::gru
