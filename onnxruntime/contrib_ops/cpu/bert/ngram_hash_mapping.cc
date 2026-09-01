// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/ngram_hash_mapping.h"

#include <cstdint>
#include <limits>

#include "contrib_ops/cpu/bert/engram_helper.h"
#include "core/common/narrow.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

#define REGISTER_NGRAM_HASH_TYPED(T)                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      NGramHashMapping,                                           \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .MayInplace(3, 1)                                       \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<T>()), \
      NGramHashMapping<T>);

REGISTER_NGRAM_HASH_TYPED(int32_t)
REGISTER_NGRAM_HASH_TYPED(int64_t)

#undef REGISTER_NGRAM_HASH_TYPED

template <typename T>
NGramHashMapping<T>::NGramHashMapping(const OpKernelInfo& info) : OpKernel(info) {
  ORT_ENFORCE(info.GetAttr<int64_t>("max_ngram_size", &max_ngram_size_).IsOK(),
              "max_ngram_size attribute is required");
  ORT_ENFORCE(info.GetAttr<int64_t>("n_head_per_ngram", &n_head_per_ngram_).IsOK(),
              "n_head_per_ngram attribute is required");
  int64_t pad_id = 0;
  ORT_ENFORCE(info.GetAttr<int64_t>("pad_id", &pad_id).IsOK(), "pad_id attribute is required");
  ORT_ENFORCE(max_ngram_size_ >= 2, "max_ngram_size must be at least 2");
  ORT_ENFORCE(n_head_per_ngram_ >= 1, "n_head_per_ngram must be positive");
  ORT_ENFORCE(pad_id >= static_cast<int64_t>(std::numeric_limits<T>::min()) &&
                  pad_id <= static_cast<int64_t>(std::numeric_limits<T>::max()),
              "pad_id is out of range for the input id type");
  pad_id_ = static_cast<T>(pad_id);
}

// Reads the id at right-aligned history slot `slot` of past_ids. Slots outside the provided history
// (or a missing past_ids) are positions before the start of the whole sequence, so they use pad_id.
template <typename T>
T NGramHashMapping<T>::HistoryId(const T* past_data, int64_t b, int64_t slot, int64_t state_length) const {
  if (past_data == nullptr || slot < 0 || slot >= state_length) {
    return pad_id_;
  }
  return past_data[b * state_length + slot];
}

template <typename T>
Status NGramHashMapping<T>::Compute(OpKernelContext* context) const {
  const Tensor* input_ids = context->Input<Tensor>(0);
  const Tensor* multipliers = context->Input<Tensor>(1);
  const Tensor* vocab_sizes = context->Input<Tensor>(2);
  const Tensor* past_ids = context->Input<Tensor>(3);

  const TensorShape& input_shape = input_ids->Shape();
  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 2, "input_ids must have rank 2");
  ORT_RETURN_IF_NOT(multipliers->Shape().NumDimensions() == 1 &&
                        multipliers->Shape()[0] == max_ngram_size_,
                    "multipliers must have shape (max_ngram_size)");
  const int64_t num_heads = (max_ngram_size_ - 1) * n_head_per_ngram_;
  ORT_RETURN_IF_NOT(vocab_sizes->Shape().NumDimensions() == 1 && vocab_sizes->Shape()[0] == num_heads,
                    "vocab_sizes must have shape ((max_ngram_size - 1) * n_head_per_ngram)");

  const int64_t batch_size = input_shape[0];
  const int64_t sequence_length = input_shape[1];
  // An n-gram window reaches this many positions before the current token.
  const int64_t state_length = max_ngram_size_ - 1;
  if (past_ids != nullptr) {
    ORT_RETURN_IF_NOT(past_ids->Shape() == TensorShape({batch_size, state_length}),
                      "past_ids must have shape (batch_size, max_ngram_size - 1)");
  }

  Tensor* output = context->Output(0, TensorShape({batch_size, sequence_length, num_heads}));
  Tensor* present_ids = context->Output(1, TensorShape({batch_size, state_length}));

  const T* input_data = input_ids->Data<T>();
  const T* multiplier_data = multipliers->Data<T>();
  const T* vocab_data = vocab_sizes->Data<T>();
  const T* past_data = past_ids == nullptr ? nullptr : past_ids->Data<T>();

  // A non-positive head vocabulary size has no meaningful modulo. Every EP guards the division to
  // avoid a device-side divide-by-zero, which turns the mistake into a constant hash id of 0 for that
  // head rather than a crash. That is a silent wrong answer, so validate it here where vocab_sizes is
  // already resident on the host and the check costs one pass over a tiny tensor.
  for (int64_t h = 0; h < num_heads; ++h) {
    ORT_RETURN_IF_NOT(vocab_data[h] > 0,
                      "vocab_sizes must be positive; entry ", h, " is ", static_cast<int64_t>(vocab_data[h]));
  }

  if (input_shape.Size() != 0) {
    T* output_data = output->MutableData<T>();
    const int64_t total = batch_size * sequence_length;
    ThreadPool::TryParallelFor(
        context->GetOperatorThreadPool(), narrow<ptrdiff_t>(total), static_cast<double>(max_ngram_size_ * n_head_per_ngram_),
        [&](ptrdiff_t begin, ptrdiff_t end) {
          for (int64_t linear = begin; linear < end; ++linear) {
            const int64_t t = linear % sequence_length;
            const int64_t b = linear / sequence_length;
            const int64_t input_base = b * sequence_length;
            const int64_t output_base = linear * num_heads;

            for (int64_t n = 2; n <= max_ngram_size_; ++n) {
              T mix = 0;
              for (int64_t k = 0; k < n; ++k) {
                const int64_t source_t = t - k;
                const T token = source_t >= 0 ? input_data[input_base + source_t]
                                              : HistoryId(past_data, b, state_length + source_t, state_length);
                const T product = engram_helper::WrappedMultiply(token, multiplier_data[k]);
                mix = k == 0 ? product : static_cast<T>(mix ^ product);
              }

              const int64_t ngram_offset = (n - 2) * n_head_per_ngram_;
              for (int64_t h = 0; h < n_head_per_ngram_; ++h) {
                const int64_t out_h = ngram_offset + h;
                // vocab_sizes was validated to be positive above, so the modulo is always well defined.
                output_data[output_base + out_h] = engram_helper::PositiveMod(mix, vocab_data[out_h]);
              }
            }
          }
        });
  }

  // present_ids is the right-aligned trailing window of (past_ids ++ input_ids), so it is well defined
  // even when this call is shorter than the window. It is written last because past_ids may share
  // its allocation, and the hash loop above still needs the original history. Within this loop the
  // aliased case is safe too: slot j writes index j and reads index j + sequence_length, so the walk
  // is strictly ahead of itself.
  if (present_ids != nullptr) {
    T* present_data = present_ids->MutableData<T>();
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t j = 0; j < state_length; ++j) {
        // Virtual position of slot j relative to the end of input_ids.
        const int64_t source_t = sequence_length - state_length + j;
        present_data[b * state_length + j] =
            source_t >= 0 ? input_data[b * sequence_length + source_t]
                          : HistoryId(past_data, b, state_length + source_t, state_length);
      }
    }
  }

  return Status::OK();
}

template class NGramHashMapping<int32_t>;
template class NGramHashMapping<int64_t>;

}  // namespace contrib
}  // namespace onnxruntime
