// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "attention_mechanism.h"

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/allocator.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class AttentionWrapper {
 public:
  AttentionWrapper(AllocatorPtr allocator,
                   const logging::Logger& logger,
                   int batch_size,
                   int attn_context_depth,  // it is also the depth of the memory
                   int attn_layer_depth,
                   int inner_cell_hidden_size,
                   bool has_attn_layer,
                   const IAttentionMechanism<T>& attention_mechanism, concurrency::ThreadPool* threadpool);

  virtual ~AttentionWrapper() = default;

  // Calculation based on output of the inner wrapped rnn_cell.
  void ProcessOutput(const gsl::span<const T>& rnn_cell_state);

  gsl::span<const T> GetAttnStates() const;

  void SetWeights(const gsl::span<const T>& wrapper_weights);

  // the size after attention layer or direct from attention context.
  int GetAttentionSize() const {
    return has_attn_layer_ ? attn_layer_depth_ : attn_context_depth_;
  }

  int GetAttentionContextSize() const {
    return attn_context_depth_;
  }

 private:
  AllocatorPtr allocator_;
  const logging::Logger& logger_;

  gsl::span<const T> attn_layer_cell_weights_;
  gsl::span<const T> attn_layer_attn_weights_;

  IAllocatorUniquePtr<T> attn_context_ptr_;
  gsl::span<T> attn_context_;

  IAllocatorUniquePtr<T> attn_states_ptr_;
  gsl::span<T> attn_states_;

  IAllocatorUniquePtr<T> prev_alignments_ptr_;
  gsl::span<T> prev_alignments_;

  IAllocatorUniquePtr<T> alignments_ptr_;
  gsl::span<T> alignments_;

  int batch_size_;
  int attn_context_depth_;
  int attn_layer_depth_;
  int inner_cell_hidden_size_;

  bool has_attn_layer_;

  const IAttentionMechanism<T>& attention_mechanism_;
  concurrency::ThreadPool* ttp_;
};

}  // namespace contrib
}  // namespace onnxruntime
