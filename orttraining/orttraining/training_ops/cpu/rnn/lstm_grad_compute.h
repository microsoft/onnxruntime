// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/rnn/rnn_helpers.h"
#include "orttraining/training_ops/cpu/rnn/lstm_io_utils.h"

namespace onnxruntime::lstm {

template <typename T>
class LSTMGradImpl {
 public:
  LSTMGradImpl(size_t sequence_length, size_t batch_size, size_t hidden_size, size_t input_size,
               concurrency::ThreadPool* thread_pool, AllocatorPtr allocator);

  void ComputeGradient(const LSTMGradInputs<T>& inputs, LSTMGradOutputs<T>& outputs);

 private:
  const size_t sequence_length_;
  const size_t batch_size_;
  const size_t hidden_size_;
  const size_t input_size_;
  concurrency::ThreadPool* thread_pool_;
  AllocatorPtr allocator_;
  IAllocatorUniquePtr<T> grad_a_ptr_;
  gsl::span<T> grad_a_span_;
  IAllocatorUniquePtr<T> grad_Ct2_ptr_;
  gsl::span<T> grad_Ct2_span_;
  IAllocatorUniquePtr<T> grad_W_ptr_;
  gsl::span<T> grad_W_span_;
  IAllocatorUniquePtr<T> grad_R_ptr_;
  gsl::span<T> grad_R_span_;
};

}  // namespace onnxruntime::lstm
