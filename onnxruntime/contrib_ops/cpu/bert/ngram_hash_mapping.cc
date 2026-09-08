// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/ngram_hash_mapping.h"

#include <cstdint>
#include <limits>
#include <vector>

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
  reset_on_eos_ = info.GetAttrOrDefault<int64_t>("reset_on_eos", 0);
}

template <typename T>
T NGramHashMapping<T>::HistoryId(const T* past_data, int64_t b, int64_t slot, int64_t state_length,
                                 T missing_history_value) const {
  if (past_data == nullptr || slot < 0 || slot >= state_length) {
    return missing_history_value;
  }
  return past_data[b * state_length + slot];
}

template <typename T>
Status NGramHashMapping<T>::Compute(OpKernelContext* context) const {
  const Tensor* input_ids = context->Input<Tensor>(0);
  const Tensor* multipliers = context->Input<Tensor>(1);
  const Tensor* vocab_sizes = context->Input<Tensor>(2);
  const Tensor* past_ids = context->Input<Tensor>(3);
  const Tensor* head_offsets = context->Input<Tensor>(4);
  const Tensor* eos_token_id = context->Input<Tensor>(5);
  const Tensor* segment_ids = context->Input<Tensor>(6);

  const TensorShape& input_shape = input_ids->Shape();
  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 2, "input_ids must have rank 2");
  ORT_RETURN_IF_NOT(multipliers->Shape().NumDimensions() == 1 &&
                        multipliers->Shape()[0] >= max_ngram_size_,
                    "multipliers must have shape (max_ngram_size)");
  const int64_t num_heads = (max_ngram_size_ - 1) * n_head_per_ngram_;
  ORT_RETURN_IF_NOT(vocab_sizes->Shape().NumDimensions() == 1 && vocab_sizes->Shape()[0] == num_heads,
                    "vocab_sizes must have shape ((max_ngram_size - 1) * n_head_per_ngram)");

  const int64_t batch_size = input_shape[0];
  const int64_t sequence_length = input_shape[1];
  const int64_t state_length = max_ngram_size_ - 1;
  if (past_ids != nullptr) {
    ORT_RETURN_IF_NOT(past_ids->Shape() == TensorShape({batch_size, state_length}),
                      "past_ids must have shape (batch_size, max_ngram_size - 1)");
  }
  if (head_offsets != nullptr) {
    ORT_RETURN_IF_NOT(head_offsets->Shape().NumDimensions() == 1 && head_offsets->Shape()[0] == num_heads,
                      "head_offsets must have shape ((max_ngram_size - 1) * n_head_per_ngram)");
  }
  if (eos_token_id != nullptr) {
    ORT_RETURN_IF_NOT(eos_token_id->Shape().Size() == 1, "eos_token_id must be a scalar");
  }
  if (segment_ids != nullptr) {
    ORT_RETURN_IF_NOT(segment_ids->Shape() == TensorShape({batch_size, sequence_length}),
                      "segment_ids must have shape (batch_size, sequence_length)");
  }

  Tensor* output = context->Output(0, TensorShape({batch_size, sequence_length, num_heads}));
  Tensor* present_ids = context->Output(1, TensorShape({batch_size, state_length}));

  const T* input_data = input_ids->Data<T>();
  const T* multiplier_data = multipliers->Data<T>();
  const T* vocab_data = vocab_sizes->Data<T>();
  const T* past_data = past_ids == nullptr ? nullptr : past_ids->Data<T>();
  const T* offset_data = head_offsets == nullptr ? nullptr : head_offsets->Data<T>();
  const int32_t* segment_data = segment_ids == nullptr ? nullptr : segment_ids->Data<int32_t>();

  for (int64_t h = 0; h < num_heads; ++h) {
    ORT_RETURN_IF_NOT(vocab_data[h] > 0,
                      "vocab_sizes must be positive; entry ", h, " is ", static_cast<int64_t>(vocab_data[h]));
  }

  const bool has_eos = eos_token_id != nullptr;
  const T eos_value = has_eos ? eos_token_id->Data<T>()[0] : pad_id_;
  const bool do_reset = reset_on_eos_ != 0 && has_eos;
  const int64_t combined_length = state_length + sequence_length;

  if (input_shape.Size() != 0) {
    T* output_data = output->MutableData<T>();
    ThreadPool::TryParallelFor(
        context->GetOperatorThreadPool(), narrow<ptrdiff_t>(batch_size),
        static_cast<double>(combined_length * max_ngram_size_),
        [&](ptrdiff_t begin, ptrdiff_t end) {
          std::vector<T> combined(static_cast<size_t>(combined_length));
          for (int64_t b = begin; b < end; ++b) {
            for (int64_t i = 0; i < state_length; ++i) {
              combined[static_cast<size_t>(i)] = HistoryId(past_data, b, i, state_length, eos_value);
            }
            for (int64_t t = 0; t < sequence_length; ++t) {
              combined[static_cast<size_t>(state_length + t)] = input_data[b * sequence_length + t];
            }

            int64_t last_reset = -1;
            for (int64_t idx = state_length; idx < combined_length; ++idx) {
              const int64_t t = idx - state_length;
              if (idx > 0) {
                const int64_t previous = idx - 1;
                bool boundary = do_reset && combined[static_cast<size_t>(previous)] == eos_value;
                if (segment_data != nullptr && t > 0 &&
                    segment_data[b * sequence_length + t] != segment_data[b * sequence_length + t - 1]) {
                  boundary = true;
                }
                if (boundary) {
                  last_reset = previous;
                }
              }

              const int64_t output_base = (b * sequence_length + t) * num_heads;
              for (int64_t n = 2; n <= max_ngram_size_; ++n) {
                T mix = 0;
                for (int64_t k = 0; k < n; ++k) {
                  const int64_t source = idx - k;
                  const T token = (last_reset >= source) ? eos_value : combined[static_cast<size_t>(source)];
                  const T product = engram_helper::WrappedMultiply(token, multiplier_data[k]);
                  mix = k == 0 ? product : static_cast<T>(mix ^ product);
                }

                const int64_t ngram_offset = (n - 2) * n_head_per_ngram_;
                for (int64_t h = 0; h < n_head_per_ngram_; ++h) {
                  const int64_t out_h = ngram_offset + h;
                  T result = engram_helper::PositiveMod(mix, vocab_data[out_h]);
                  if (offset_data != nullptr) {
                    result = static_cast<T>(result + offset_data[out_h]);
                  }
                  output_data[output_base + out_h] = result;
                }
              }
            }
          }
        });
  }

  if (present_ids != nullptr) {
    T* present_data = present_ids->MutableData<T>();
    for (int64_t b = 0; b < batch_size; ++b) {
      std::vector<T> present_row(static_cast<size_t>(state_length));
      for (int64_t j = 0; j < state_length; ++j) {
        const int64_t source_t = sequence_length - state_length + j;
        present_row[static_cast<size_t>(j)] =
            source_t >= 0 ? input_data[b * sequence_length + source_t]
                          : HistoryId(past_data, b, state_length + source_t, state_length, eos_value);
      }
      for (int64_t j = 0; j < state_length; ++j) {
        present_data[b * state_length + j] = present_row[static_cast<size_t>(j)];
      }
    }
  }

  return Status::OK();
}

template class NGramHashMapping<int32_t>;
template class NGramHashMapping<int64_t>;

}  // namespace contrib
}  // namespace onnxruntime
