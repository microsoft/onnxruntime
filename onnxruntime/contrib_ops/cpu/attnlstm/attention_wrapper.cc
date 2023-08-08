// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention_wrapper.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"

#include <stdexcept>
#include <memory>

using onnxruntime::rnn::detail::Allocate;
// TODO: fix the warnings
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26451)
#endif
namespace onnxruntime {
namespace contrib {

template <typename T>
AttentionWrapper<T>::AttentionWrapper(AllocatorPtr alloc, const logging::Logger& logger,
                                      int batch_size, int attn_context_depth, int attn_layer_depth,
                                      int inner_cell_hidden_size, bool has_attn_layer,
                                      const IAttentionMechanism<T>& attention_mechanism, concurrency::ThreadPool* threadpool)
    : allocator_(alloc),
      logger_(logger),
      batch_size_(batch_size),
      attn_context_depth_(attn_context_depth),
      attn_layer_depth_(attn_layer_depth),
      inner_cell_hidden_size_(inner_cell_hidden_size),
      has_attn_layer_(has_attn_layer),
      attention_mechanism_(attention_mechanism),
      ttp_(threadpool) {
  auto mem_max_steps = attention_mechanism_.GetMaxMemorySteps();
  prev_alignments_ = Allocate(allocator_, batch_size_ * mem_max_steps, prev_alignments_ptr_, true);
  alignments_ = Allocate(allocator_, batch_size_ * mem_max_steps, alignments_ptr_, true);
  attn_context_ = Allocate(allocator_, batch_size_ * attn_context_depth, attn_context_ptr_, true);
  attn_states_ = (has_attn_layer_) ? Allocate(allocator_, batch_size_ * attn_layer_depth_, attn_states_ptr_, true) : attn_context_;
}

// rnn_cell_output is of [batch_size, rnn_cell_hidden_size]
template <typename T>
void AttentionWrapper<T>::ProcessOutput(const gsl::span<const T>& rnn_cell_output) {
  if (has_attn_layer_) {
    // rnn_cell_output * cell_weights, (part of the attention layer above the attention mechanism).
    math::GemmEx<T>(CblasNoTrans, CblasNoTrans,
                    batch_size_, attn_layer_depth_, inner_cell_hidden_size_, T{1.0},
                    rnn_cell_output.data(), inner_cell_hidden_size_,
                    attn_layer_cell_weights_.data(), attn_layer_depth_, T{0.0},
                    attn_states_.data(), attn_layer_depth_, ttp_);
  }

  // Get the context which is calculated within attention mechanism.
  attention_mechanism_.Compute(rnn_cell_output, prev_alignments_, attn_context_, alignments_);
  if (attention_mechanism_.NeedPrevAlignment()) {
    std::copy(alignments_.begin(), alignments_.end(), prev_alignments_.begin());
  }

  if (has_attn_layer_) {
    // concat([p_cell_output, context]) * stack([attn_layer_cell_weights_, attn_layer_attn_weights_]) =
    //      p_cell_output * attn_layer_cell_weights_ + context * attn_layer_attn_weights_
    //  The first part is calulated above. Here just add the later.
    math::GemmEx<T>(CblasNoTrans, CblasNoTrans,
                    batch_size_, attn_layer_depth_, attn_context_depth_, T{1.0},
                    attn_context_.data(), attn_context_depth_,
                    attn_layer_attn_weights_.data(), attn_layer_depth_, T{1.0},
                    attn_states_.data(), attn_layer_depth_, ttp_);
  }
}

template <typename T>
gsl::span<const T> AttentionWrapper<T>::GetAttnStates() const {
  return attn_states_;
}

template <typename T>
void AttentionWrapper<T>::SetWeights(const gsl::span<const T>& wrapper_weights) {
  has_attn_layer_ = !wrapper_weights.empty();

  if (has_attn_layer_) {
    // cell weight size and attn weight size in the attn layer
    size_t cws = inner_cell_hidden_size_ * attn_layer_depth_;
    size_t aws = attn_context_depth_ * attn_layer_depth_;
    attn_layer_cell_weights_ = wrapper_weights.subspan(0, cws);
    attn_layer_attn_weights_ = wrapper_weights.subspan(cws, aws);
  }
}

template class AttentionWrapper<float>;

}  // namespace contrib
}  // namespace onnxruntime
