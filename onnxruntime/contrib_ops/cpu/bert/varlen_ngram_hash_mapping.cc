// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/varlen_ngram_hash_mapping.h"

#include <cstdint>
#include <limits>

#include "contrib_ops/cpu/bert/engram_helper.h"
#include "core/common/narrow.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

#define REGISTER_VARLEN_NGRAM_HASH_TYPED(T)                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      VarlenNGramHashMapping,                                           \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCpuExecutionProvider,                                            \
      KernelDefBuilder()                                                \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("S", DataTypeImpl::GetTensorType<int32_t>()), \
      VarlenNGramHashMapping<T>);

REGISTER_VARLEN_NGRAM_HASH_TYPED(int32_t)
REGISTER_VARLEN_NGRAM_HASH_TYPED(int64_t)

#undef REGISTER_VARLEN_NGRAM_HASH_TYPED

template <typename T>
VarlenNGramHashMapping<T>::VarlenNGramHashMapping(const OpKernelInfo& info) : OpKernel(info) {
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

// Reads the id at right-aligned history slot `slot` of past_ids for request `b`. Slots outside the
// provided history (or a missing past_ids) are positions before the start of the whole sequence, so
// they use pad_id.
template <typename T>
T VarlenNGramHashMapping<T>::HistoryId(const T* past_data, int64_t b, int64_t slot, int64_t state_length) const {
  if (past_data == nullptr || slot < 0 || slot >= state_length) {
    return pad_id_;
  }
  return past_data[b * state_length + slot];
}

template <typename T>
Status VarlenNGramHashMapping<T>::Compute(OpKernelContext* context) const {
  const Tensor* input_ids = context->Input<Tensor>(0);
  const Tensor* multipliers = context->Input<Tensor>(1);
  const Tensor* vocab_sizes = context->Input<Tensor>(2);
  const Tensor* cu_seqlens = context->Input<Tensor>(3);
  const Tensor* past_ids = context->Input<Tensor>(4);

  ORT_RETURN_IF_NOT(input_ids->Shape().NumDimensions() == 1, "input_ids must have rank 1 (total_tokens)");
  ORT_RETURN_IF_NOT(multipliers->Shape().NumDimensions() == 1 &&
                        multipliers->Shape()[0] == max_ngram_size_,
                    "multipliers must have shape (max_ngram_size)");
  int64_t num_heads = 0;
  ORT_RETURN_IF_NOT(engram_helper::TryMultiplyDims(max_ngram_size_ - 1, n_head_per_ngram_, num_heads),
                    "VarlenNGramHashMapping: (max_ngram_size - 1) * n_head_per_ngram overflows int64_t");
  ORT_RETURN_IF_NOT(vocab_sizes->Shape().NumDimensions() == 1 && vocab_sizes->Shape()[0] == num_heads,
                    "vocab_sizes must have shape ((max_ngram_size - 1) * n_head_per_ngram)");
  ORT_RETURN_IF_NOT(cu_seqlens->Shape().NumDimensions() == 1 && cu_seqlens->Shape()[0] >= 2,
                    "cumulative_sequence_length must have rank 1 with at least 2 elements");

  const int64_t total_tokens = input_ids->Shape()[0];
  const int64_t batch_size = cu_seqlens->Shape()[0] - 1;
  ORT_RETURN_IF_NOT(total_tokens >= batch_size,
                    "total_tokens must be at least batch_size because every request must contain a token");
  const int64_t state_length = max_ngram_size_ - 1;
  if (past_ids != nullptr) {
    ORT_RETURN_IF_NOT(past_ids->Shape() == TensorShape({batch_size, state_length}),
                      "past_ids must have shape (batch_size, max_ngram_size - 1)");
  }

  const int32_t* cu_data = cu_seqlens->Data<int32_t>();
  ORT_RETURN_IF_NOT(cu_data[0] == 0, "cumulative_sequence_length[0] must be 0");
  ORT_RETURN_IF_NOT(static_cast<int64_t>(cu_data[batch_size]) == total_tokens,
                    "cumulative_sequence_length[batch_size] must equal total_tokens");
  for (int64_t b = 0; b < batch_size; ++b) {
    ORT_RETURN_IF_NOT(cu_data[b] >= 0 && cu_data[b] < cu_data[b + 1],
                      "cumulative_sequence_length must be strictly increasing and non-negative "
                      "because every request must contain at least one token");
  }

  Tensor* output = context->Output(0, TensorShape({total_tokens, num_heads}));
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

  T* present_data = present_ids == nullptr ? nullptr : present_ids->MutableData<T>();
  T* output_data = total_tokens == 0 ? nullptr : output->MutableData<T>();

  ThreadPool::TryParallelFor(
      context->GetOperatorThreadPool(), narrow<ptrdiff_t>(batch_size),
      static_cast<double>((total_tokens > 0 ? total_tokens / std::max<int64_t>(batch_size, 1) : 1) *
                          max_ngram_size_ * n_head_per_ngram_),
      [&](ptrdiff_t begin, ptrdiff_t end) {
        for (int64_t b = begin; b < end; ++b) {
          const int64_t start = cu_data[b];
          const int64_t seq_end = cu_data[b + 1];
          const int64_t local_length = seq_end - start;

          // present_ids is the right-aligned trailing window of (past_ids ++ this request's tokens),
          // so it is well defined even when this call is shorter than the window. It never reads
          // across a sequence boundary into an adjacent packed request.
          if (present_data != nullptr) {
            for (int64_t j = 0; j < state_length; ++j) {
              const int64_t source_t = local_length - state_length + j;
              present_data[b * state_length + j] =
                  source_t >= 0 ? input_data[start + source_t]
                                : HistoryId(past_data, b, state_length + source_t, state_length);
            }
          }

          for (int64_t t = 0; t < local_length; ++t) {
            const int64_t output_base = (start + t) * num_heads;
            for (int64_t n = 2; n <= max_ngram_size_; ++n) {
              T mix = 0;
              for (int64_t k = 0; k < n; ++k) {
                const int64_t source_t = t - k;
                const T token = source_t >= 0 ? input_data[start + source_t]
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
        }
      });

  return Status::OK();
}

template class VarlenNGramHashMapping<int32_t>;
template class VarlenNGramHashMapping<int64_t>;

}  // namespace contrib
}  // namespace onnxruntime
