// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"

#include "attention_mechanism.h"

namespace onnxruntime {
namespace contrib {

// Please refer to: https://arxiv.org/pdf/1409.0473.pdf
template <typename T>
class BahdanauAttention : public IAttentionMechanism<T> {
 public:
  BahdanauAttention(
      AllocatorPtr allocator,
      const logging::Logger& logger,
      int batch_size,
      int max_memory_step,
      int memory_depth,
      int query_depth,
      int attn_depth,
      bool normalize, concurrency::ThreadPool* threadpool);

  void SetWeights(
      const gsl::span<const T>& attn_weights,
      const gsl::span<const T>& query_layer_weights,
      const gsl::span<const T>& memory_layer_weights);

  ~BahdanauAttention() override = default;

  void PrepareMemory(
      const gsl::span<const T>& memory,
      const gsl::span<const int>& memory_sequence_lengths) override;

  void Compute(
      const gsl::span<const T>& queries,
      const gsl::span<const T>& prev_alignment,
      const gsl::span<T>& output,
      const gsl::span<T>& alignment) const override;

  const gsl::span<const T> Values() const override;

  const gsl::span<const T> Keys() const override;

  int GetMaxMemorySteps() const override;

  bool NeedPrevAlignment() const override;

 private:
  AllocatorPtr allocator_;
  const logging::Logger& logger_;

  int batch_size_;
  int max_memory_steps_;
  int memory_depth_;
  int query_depth_;
  int attn_depth_;

  gsl::span<const T> attention_v_;
  gsl::span<const T> query_layer_weights_;
  gsl::span<const T> memory_layer_weights_;

  IAllocatorUniquePtr<T> keys_ptr_;
  gsl::span<T> keys_;

  IAllocatorUniquePtr<T> values_ptr_;
  gsl::span<T> values_;

  IAllocatorUniquePtr<T> processed_query_ptr_;
  gsl::span<T> processed_query_;

  IAllocatorUniquePtr<int> mem_seq_lengths_ptr_;
  gsl::span<int> mem_seq_lengths_;

  bool normalize_;
  concurrency::ThreadPool* ttp_;
};

}  // namespace contrib
}  // namespace onnxruntime
