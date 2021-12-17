// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bahdanau_attention.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"

#include <stdexcept>
#include <memory.h>

using onnxruntime::rnn::detail::Allocate;
//TODO: fix the warnings
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26451)
#endif
namespace onnxruntime {
namespace contrib {

template <typename T>
BahdanauAttention<T>::BahdanauAttention(AllocatorPtr allocator, const logging::Logger& logger,
                                        int batch_size, int max_memory_step, int memory_depth,
                                        int query_depth, int attn_depth, bool normalize, concurrency::ThreadPool* threadpool)
    : allocator_(allocator), logger_(logger), batch_size_(batch_size), max_memory_steps_(max_memory_step), memory_depth_(memory_depth), query_depth_(query_depth), attn_depth_(attn_depth), normalize_(normalize), ttp_(threadpool) {
  values_ = Allocate(allocator_, batch_size_ * max_memory_steps_ * memory_depth_, values_ptr_, true);
  keys_ = Allocate(allocator_, batch_size_ * max_memory_steps_ * attn_depth_, keys_ptr_, true);
  processed_query_ = Allocate(allocator_, batch_size_ * attn_depth_, processed_query_ptr_, true);
  mem_seq_lengths_ = Allocate(allocator_, batch_size_, mem_seq_lengths_ptr_, true);

  ORT_ENFORCE(!normalize_, "not support normalize yet.");
}

template <typename T>
void BahdanauAttention<T>::SetWeights(
    const gsl::span<const T>& attn_weights,
    const gsl::span<const T>& query_layer_weights,
    const gsl::span<const T>& memory_layer_weights) {
  attention_v_ = attn_weights;                   //[attn_depth_]
  query_layer_weights_ = query_layer_weights;    //[query_depth_, attn_depth_]
  memory_layer_weights_ = memory_layer_weights;  //[memory_depth_, attn_depth_]
}

template <typename T>
const gsl::span<const T> BahdanauAttention<T>::Values() const {
  return values_;
}

template <typename T>
const gsl::span<const T> BahdanauAttention<T>::Keys() const {
  return keys_;
}

template <typename T>
int BahdanauAttention<T>::GetMaxMemorySteps() const {
  return max_memory_steps_;
}

template <typename T>
bool BahdanauAttention<T>::NeedPrevAlignment() const {
  return false;
}

template <typename T>
void BahdanauAttention<T>::PrepareMemory(
    const gsl::span<const T>& memory,
    const gsl::span<const int>& memory_sequence_lengths) {
  std::copy(memory.cbegin(), memory.cend(), values_.begin());
  if (memory_sequence_lengths.empty()) {
    std::fill(mem_seq_lengths_.begin(), mem_seq_lengths_.end(), max_memory_steps_);
  } else {
    std::copy(memory_sequence_lengths.cbegin(), memory_sequence_lengths.cend(), mem_seq_lengths_.begin());
  }

  for (int b = 0; b < batch_size_; b++) {
    int mem_steps = mem_seq_lengths_[b];
    ORT_ENFORCE(mem_steps <= max_memory_steps_ && mem_steps > 0,
                "Real memory steps ", mem_steps, " is not in (0, ", max_memory_steps_, "]");
  }

  math::GemmEx<T>(CblasNoTrans, CblasNoTrans,
                  batch_size_ * max_memory_steps_, attn_depth_, memory_depth_, T{1.0},
                  memory.data(), memory_depth_,
                  memory_layer_weights_.data(), attn_depth_, T{0.0},
                  keys_.data(), attn_depth_, ttp_);
}

template <typename T>
static void SoftmaxInplace(const gsl::span<T>& alignments) {
  T* x = alignments.data();
  size_t len = alignments.size();

  double sum = 0.0;

  for (size_t i = 0; i < len; i++) {
    T e = exp(x[i]);
    sum += e;
    x[i] = e;
  }

  if (sum == 0.0) {
    for (size_t i = 0; i < len; i++) {
      x[i] = static_cast<T>(1.0 / len);
    }
  } else {
    for (size_t i = 0; i < len; i++) {
      x[i] = static_cast<T>(x[i] / sum);
    }
  }
}

/**
  * Args:
  *     queries: Tensor, shape `[batch_size_, query_depth_]` to compare to keys.
  *     keys_: Processed memory, shape `[batch_size_, max_memory_step_, attn_depth_]`.
  */
template <typename T>
void BahdanauAttention<T>::Compute(
    const gsl::span<const T>& queries,
    const gsl::span<const T>&,  // Not used by bahdanau attention
    const gsl::span<T>& output,
    const gsl::span<T>& aligns) const {
  //process query in dense query layer without bias
  math::GemmEx<T>(CblasNoTrans, CblasNoTrans,
                  batch_size_, attn_depth_, query_depth_, T{1.0},
                  queries.data(), query_depth_,
                  query_layer_weights_.data(), attn_depth_, T{0.0},
                  processed_query_.data(), attn_depth_, ttp_);

  std::fill(aligns.begin(), aligns.end(), T{});

  // return math_ops.reduce_sum(v * math_ops.tanh(keys + processed_query), [2])
  for (int b = 0; b < batch_size_; b++) {
    T* alignments = aligns.data() + b * max_memory_steps_;
    const T* keys = keys_.data() + b * max_memory_steps_ * attn_depth_;
    const T* query = processed_query_.data() + b * attn_depth_;

    int mem_steps = mem_seq_lengths_[b];
    for (int step = 0; step < mem_steps; step++) {
      const T* keys_on_step = keys + step * attn_depth_;
      T* dest = alignments + step;

      // reduce_sum(v * tanh(keys[step] + query)) on last dimension
      *dest = T(0.0);
      for (int i = 0; i < attn_depth_; i++) {
        *dest += (attention_v_[i] * tanh(keys_on_step[i] + query[i]));
      }
    }

    SoftmaxInplace(gsl::span<T>{alignments, gsl::narrow_cast<gsl::index>(mem_steps)});

    // Calculate the context
    auto outspan = output.subspan(b * memory_depth_);
    auto values = values_.subspan(b * max_memory_steps_ * memory_depth_);
    math::GemmEx<T>(CblasNoTrans, CblasNoTrans,
                    1, memory_depth_, max_memory_steps_, T{1.0},
                    alignments, max_memory_steps_,
                    values.data(), memory_depth_, T{0.0},
                    outspan.data(), memory_depth_, ttp_);
  }
}

template class BahdanauAttention<float>;

}  // namespace contrib
}  // namespace onnxruntime
