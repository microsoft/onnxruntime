// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/ngram_hash_mapping.h"

#include <cstdint>
#include <limits>

#include "contrib_ops/cpu/bert/kernel_helper.h"
#include "core/common/narrow.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

#define REGISTER_NGRAM_HASH_TYPED(T)                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      NgramHashMapping,                                           \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<T>()), \
      NgramHashMapping<T>);

REGISTER_NGRAM_HASH_TYPED(int32_t)
REGISTER_NGRAM_HASH_TYPED(int64_t)

#undef REGISTER_NGRAM_HASH_TYPED

template <typename T>
NgramHashMapping<T>::NgramHashMapping(const OpKernelInfo& info) : OpKernel(info) {
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

template <typename T>
Status NgramHashMapping<T>::Compute(OpKernelContext* context) const {
  const Tensor* input_ids = context->Input<Tensor>(0);
  const Tensor* multipliers = context->Input<Tensor>(1);
  const Tensor* vocab_sizes = context->Input<Tensor>(2);

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
  Tensor* output = context->Output(0, TensorShape({batch_size, sequence_length, num_heads}));

  const T* input_data = input_ids->Data<T>();
  const T* multiplier_data = multipliers->Data<T>();
  const T* vocab_data = vocab_sizes->Data<T>();
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
              const T token = source_t < 0 ? pad_id_ : input_data[input_base + source_t];
              const T product = kernel_helper::WrappedMultiply(token, multiplier_data[k]);
              mix = k == 0 ? product : static_cast<T>(mix ^ product);
            }

            const int64_t ngram_offset = (n - 2) * n_head_per_ngram_;
            for (int64_t h = 0; h < n_head_per_ngram_; ++h) {
              const int64_t out_h = ngram_offset + h;
              const T mod = vocab_data[out_h];
              output_data[output_base + out_h] = mod <= 0 ? T{} : kernel_helper::PositiveMod(mix, mod);
            }
          }
        }
      });

  return Status::OK();
}

template class NgramHashMapping<int32_t>;
template class NgramHashMapping<int64_t>;

}  // namespace contrib
}  // namespace onnxruntime
