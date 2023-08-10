// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "orttraining/training_ops/cpu/rnn/gru_io_utils.h"

namespace onnxruntime::gru {

template <typename T>
class GRUGradImpl {
 public:
  GRUGradImpl(int sequence_length, int batch_size, int hidden_size, int input_size,
              bool linear_before_reset, concurrency::ThreadPool* thread_pool,
              AllocatorPtr allocator);

  void ComputeGradient(const GRUGradInputs<T>& inputs, GRUGradOutputs<T>& outputs);

 private:
  const int sequence_length_;
  const int batch_size_;
  const int hidden_size_;
  const int input_size_;
  const bool linear_before_reset_;
  concurrency::ThreadPool* thread_pool_;
  const AllocatorPtr allocator_;
  IAllocatorUniquePtr<T> grad_a_ptr_;
  gsl::span<T> grad_a_span_;
  IAllocatorUniquePtr<T> rt_factor_ptr_;
  gsl::span<T> rt_factor_span_;
  IAllocatorUniquePtr<T> grad_W_ptr_;
  gsl::span<T> grad_W_span_;
  IAllocatorUniquePtr<T> grad_R_ptr_;
  gsl::span<T> grad_R_span_;
};

}  // namespace onnxruntime::gru
